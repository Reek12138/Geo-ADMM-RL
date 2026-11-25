"""Check if early, bounded speed and altitude plans can resolve developing traffic."""

import argparse
from dataclasses import fields, replace
import json
from math import cos, prod
from pathlib import Path

import torch

from bluesky.rl_gpu import B737PerfConfig, TorchAirspaceState, TorchVecEnvConfig
from bluesky.rl_gpu.aero_torch import FT, KTS
from bluesky.rl_gpu.physics_torch import bearing_deg_torch, haversine_km_torch
from core.conflict_graph import cylinder_interval
from envs.bluesky_graph import METRES_PER_DEGREE, RADIANS, SCALE
from envs.bluesky_plan_b import GeoTorchBlueSky
from envs.bluesky_safety import single_axis_commands
from eval.bluesky_collision_trace import _component_plan
from eval.bluesky_pair import _checksum

PROFILES = ((0, 0), (0, -1), (0, 1), (-20, 0), (20, 0),
            (-20, -1), (-20, 1), (20, -1), (20, 1))
OUTCOMES = ('arrivals', 'collisions', 'exits', 'timeouts')


def _targets(state, cfg, profiles):
    speed, layer = profiles.unbind(dim=-1)
    return ((state.cas / KTS + speed).clamp(*cfg.cas_bounds_kts) * KTS,
            (state.alt / FT + layer * cfg.max_alt_delta_ft).clamp(*cfg.alt_bounds_ft) * FT)


def _controls(state, cfg, target_cas, target_alt):
    speed = ((target_cas - state.cas) / (cfg.max_speed_delta_kts * KTS)).clamp(-1, 1)
    altitude = ((target_alt - state.alt) / (cfg.max_alt_delta_ft * FT)).clamp(-1, 1)
    goal_heading = bearing_deg_torch(state.lat, state.lon, state.goal_lat, state.goal_lon)
    heading_error = (goal_heading - state.hdg + 180).remainder(360) - 180
    heading = (heading_error / cfg.max_hdg_delta_deg).clamp(-1, 1).masked_fill(state.swlnav, 0)
    return torch.stack((speed, altitude, heading), dim=-1)


def _positions(state, cfg):
    metre_lon = METRES_PER_DEGREE * cos(cfg.center_lat * RADIANS)
    east = (state.lon - cfg.center_lon) * metre_lon
    north = (state.lat - cfg.center_lat) * METRES_PER_DEGREE
    return torch.stack((east, north, state.alt), dim=-1)


def _preview(env, profiles, start_controls=None, *, single_axis=False):
    """Propagate all individual profiles and retain one-second trajectory chords."""
    env.reset()
    state = TorchAirspaceState(**{
        field.name: getattr(env.state, field.name).unsqueeze(-1).expand(-1, -1, len(profiles))
        for field in fields(TorchAirspaceState)
    })
    cfg = env.cfg
    sample_dt = 1.0
    physics_steps = round(sample_dt / cfg.sim_dt)
    samples_per_control = round(cfg.control_dt / sample_dt)
    if physics_steps * cfg.sim_dt != sample_dt or samples_per_control * sample_dt != cfg.control_dt:
        raise ValueError("Preview needs aligned physics, trajectory and control intervals")
    target_cas, target_alt = _targets(state, cfg, profiles)
    positions, active = [_positions(state, cfg)], [state.active]
    for decision in range(cfg.max_steps):
        controls = _controls(state, cfg, target_cas, target_alt)
        if start_controls is not None:
            controls[..., :2] *= (decision >= start_controls)[..., None, None]
        if single_axis:
            controls = single_axis_commands(controls)
        state = env._commands(state, controls)
        for _ in range(samples_per_control):
            state = env.core.step_physics(state, simdt=cfg.sim_dt, n_substeps=physics_steps)
            arrived = haversine_km_torch(state.lat, state.lon, state.goal_lat, state.goal_lon) < cfg.arrival_dist_km
            state = replace(state, active=state.active & ~arrived)
            positions.append(_positions(state, cfg))
            active.append(state.active)
        if not state.active.any():
            break
    return torch.stack(positions), torch.stack(active), sample_dt


def _activation(env, profiles, lead):
    cfg = env.cfg
    position, active, sample_dt = _preview(env, profiles[:1])
    horizontal = position.squeeze(-2)[..., :2]
    relative = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    flying = active.squeeze(-1)
    distinct = ~torch.eye(cfg.max_aircraft, dtype=torch.bool)
    overlap = (relative.square().sum(dim=-1) < (cfg.collision_h_km * 1000) ** 2) & distinct
    overlap &= flying.unsqueeze(-1) & flying.unsqueeze(-2)
    steps = torch.arange(len(position))[:, None, None]
    entry = torch.where(overlap.any(dim=-1), steps, len(position)).amin(dim=0)
    start = ((entry * sample_dt - lead).clamp_min(0) / cfg.control_dt).floor().long()
    return torch.where(entry < len(position), start, cfg.max_steps)


def _compatible(position, active, sample_dt, cfg):
    """Reject pair profiles with a concurrent horizontal and vertical violation."""
    _, aircraft, profiles, _ = position.shape
    shape = (aircraft, aircraft, profiles, profiles)
    conflicts = torch.zeros(shape, dtype=torch.bool, device=position.device)
    for start in range(0, len(position) - 1, 48):
        stop = min(start + 48, len(position) - 1)
        point = position[start:stop]
        velocity = (position[start + 1:stop + 1] - point) / sample_dt
        relative = point[:, :, None, :, None] - point[:, None, :, None, :]
        closing = velocity[:, :, None, :, None] - velocity[:, None, :, None, :]
        _, _, contact = cylinder_interval(relative, closing, cfg.collision_h_km * 1000,
                                           cfg.collision_v_ft * FT - 1e-4, sample_dt)
        flying = active[start:stop]
        conflicts |= (contact & flying[:, :, None, :, None] & flying[:, None, :, None, :]).any(dim=0)
    return ~conflicts


def _audit(cfg, perf, family, profiles, start_controls=None, *, single_axis=False, trace=None,
           detection_metres=SCALE.detection):
    env = GeoTorchBlueSky(cfg, family=family, perf=perf, audit_physics=True, detection_metres=detection_metres)
    graph = env.reset()
    speed, altitude = _targets(env.state, cfg, profiles)
    live = torch.ones(cfg.num_envs, dtype=torch.bool)
    seen = torch.zeros_like(graph[-1])
    first_decisions = torch.zeros(2, dtype=torch.long)
    totals = torch.zeros((cfg.num_envs, len(OUTCOMES)), dtype=torch.long)
    for decision in range(cfg.max_steps):
        control = _controls(env.state, cfg, speed, altitude)
        if start_controls is not None:
            control[..., :2] *= (decision >= start_controls).unsqueeze(-1)
        if single_axis:
            control = single_axis_commands(control)
        changed = control[..., :2].abs().amax(dim=-1) > .03
        first = live.unsqueeze(-1) & graph[-1] & changed & ~seen
        linked = graph[2].any(dim=-1)
        first_decisions += torch.stack(((first & linked).sum(), (first & ~linked).sum()))
        seen |= first
        if trace is not None:
            recording = graph[-1] & (control.abs().amax(dim=-1) > .03)
            worlds = torch.where(live & recording.any(dim=-1))[0]
            if worlds.numel():
                trace.append((*[value[worlds] for value in graph], control[worlds], worlds,
                              torch.full_like(worlds, decision)))
        graph, _, done, info = env.step(control)
        totals += torch.stack([info[name] for name in OUTCOMES], dim=-1) * live.unsqueeze(-1)
        live &= ~done
        if not live.any():
            break
    return ([dict(zip(OUTCOMES, values, strict=True)) for values in totals.tolist()],
            dict(zip(('linked', 'isolated'), first_decisions.tolist(), strict=True)))


def _write_teacher(path, records, outcomes, cfg, family):
    if not records:
        return 0
    merged = tuple(torch.cat(values) for values in zip(*records, strict=True))
    nodes, edges, adjacency, active, commands, world, decision = merged
    successful = torch.tensor([row['arrivals'] == cfg.max_aircraft for row in outcomes])
    keep = successful[world]
    if not keep.any():
        return 0
    torch.save({
        'graph': (nodes[keep], edges[keep], adjacency[keep], active[keep]),
        'commands': commands[keep].float(), 'world': world[keep], 'decision': decision[keep],
        'family': family, 'seed': cfg.seed,
    }, path)
    return int(keep.sum())


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    measured = json.loads(args.calibration.read_text())
    perf = B737PerfConfig(**{key: measured[key] for key in ('axmax_mps2', 'bank_def_rad', 'vs_def_mps')})
    families = {}
    options = PROFILES
    if args.wide_speed:
        options += tuple((speed, layer) for speed in (-35, 35) for layer in (0, -1, 1))
    if args.two_layer_targets:
        options += tuple((speed, layer) for layer in (-2, 2) for speed in (0, -20, 20))
    profiles = torch.tensor(options)
    if args.lead_seconds is not None and args.lead_seconds <= 0:
        raise ValueError("Lead time must be positive")
    if args.teacher_output is not None:
        if not args.single_axis:
            raise ValueError("Teacher commands must follow the policy's single-axis action space")
        args.teacher_output.mkdir(parents=True)
    for index, family in enumerate(args.families):
        cfg = TorchVecEnvConfig(num_envs=args.envs, max_aircraft=args.aircraft, sim_dt=.25, control_dt=6,
                               max_steps=args.max_steps, hdg_deadband_deg=.25, device='cpu',
                               dtype=torch.float64, seed=args.seed + 10_007 * index)
        start_controls = None
        if args.lead_seconds is not None:
            start_controls = _activation(GeoTorchBlueSky(cfg, family=family, perf=perf), profiles, args.lead_seconds)
        positions, active, step = _preview(GeoTorchBlueSky(cfg, family=family, perf=perf),
                                          profiles, start_controls, single_axis=args.single_axis)
        selected = torch.zeros((cfg.num_envs, cfg.max_aircraft), dtype=torch.long)
        search = []
        for world in range(cfg.num_envs):
            allowed = _compatible(positions[:, world], active[:, world], step, cfg)
            domains = [torch.nonzero(~row, as_tuple=False).flatten().tolist() for row in active[-1, world]]
            plan, attempts = _component_plan(allowed.tolist(), list(range(cfg.max_aircraft)), choices=domains)
            search.append({'feasible': plan is not None, 'assignments_examined': attempts,
                           'assignment_combinations': prod(map(len, domains))})
            if plan is not None:
                for aircraft, choice in plan.items():
                    selected[world, aircraft] = choice
        nominal, _ = _audit(cfg, perf, family, profiles[torch.zeros_like(selected)])
        records = [] if args.teacher_output is not None else None
        planned, context = _audit(cfg, perf, family, profiles[selected], start_controls, single_axis=args.single_axis,
                                  trace=records)
        families[family] = {'seed': cfg.seed, 'search': search, 'selected_profiles': profiles[selected].tolist(),
                            'route_recovery_only': nominal, 'early_profiles_with_route_recovery': planned,
                            'initial_profile_change_context': context,
                            'start_controls': start_controls.tolist() if start_controls is not None else None}
        if records is not None:
            output = args.teacher_output / f'{family}.pt'
            rows = _write_teacher(output, records, planned, cfg, family)
            families[family]['teacher_rows'] = rows
            if rows:
                families[family]['teacher_sha256'] = _checksum(output)
        print(json.dumps({'family': family, 'search': search, 'nominal': nominal, 'planned': planned,
                          'initial_profile_change_context': context}), flush=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps({'event_sampling_seconds': .25, 'aircraft': args.aircraft,
                                         'lead_seconds': args.lead_seconds,
                                         'single_axis': args.single_axis,
                                         'wide_speed': args.wide_speed,
                                         'two_layer_targets': args.two_layer_targets,
                                         'limits': {'cas_kts': cfg.cas_bounds_kts, 'altitude_ft': cfg.alt_bounds_ft},
                                         'families': families}, indent=2) + '\n')
        temporary.replace(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', default=['crossing4', 'crossing6', 'head_on'],
                        choices=('crossing3', 'crossing4', 'crossing6', 'head_on', 'yaw_plus15', 'yaw_minus15'))
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--aircraft', type=int, default=12)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=72119)
    parser.add_argument('--lead-seconds', type=float,
                        help='Delay speed and altitude changes until this long before nominal horizontal overlap')
    parser.add_argument('--single-axis', action='store_true', help='Use the policy action space of one axis per decision')
    parser.add_argument('--wide-speed', action='store_true', help='Include gradual 35 knot changes in candidate profiles')
    parser.add_argument('--two-layer-targets', action='store_true',
                        help='Allow two-layer targets approached in individual 300 ft steps')
    parser.add_argument('--teacher-output', type=Path, help='Save actions from collision-free rollouts')
    run(parser.parse_args())
