"""Trace native crossing collisions and compare pre-event forecasts at K=2 and K=4."""

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import json
from math import ceil, floor, isfinite
from pathlib import Path

import torch

from core.admm_layer import altitude_admm, speed_admm
from core.conflict_graph import _ball_interval, closest_approach, conflict_graph, connected_components, cylinder_interval
from core.native_vertical import DT, altitude_paths
from envs.bluesky_graph import SCALE
from envs.bluesky_safety import (
    CAS_DELTA, PhysicalState, _forecast_command, _overlap_vertical_gap, _vertical_difference, protected_command,
)
from envs.observations import LAYER
from eval.bluesky_pair import FAMILIES, _checksum, _load
from train.td3 import encode_action


def _native_vertical_clearance(altitude, target, vertical_speed, lo, hi):
    """Replay B737 vertical selection at 0.05 s, with horizontal motion held fixed."""
    dt, acceleration, cruise_rate = 0.05, 1.524, 7.62
    height = altitude.double().unsqueeze(-1).expand_as(target)
    speed = vertical_speed.double().unsqueeze(-1).expand_as(height)
    target = target.double()
    heights = [height]
    for _ in range(round(SCALE.horizon / dt)):
        delta = target - height
        threshold = 1.05 * dt * speed.abs().clamp_min(cruise_rate)
        changing = delta.abs() > threshold
        desired = changing * delta.sign() * cruise_rate
        error = desired - speed
        speed = torch.where(error.abs() > acceleration, speed + error.sign() * acceleration * dt, desired)
        height = torch.where(changing, (1e6 * (height + speed * dt)).round() / 1e6, target)
        heights.append(height)
        if (height == target).all():
            break

    path = torch.stack(heights)
    gaps = path.unsqueeze(-2).unsqueeze(-1) - path.unsqueeze(-3).unsqueeze(-2)
    times = dt * torch.arange(len(path), dtype=gaps.dtype, device=gaps.device)
    within = (times[:, None, None, None] >= lo) & (times[:, None, None, None] <= hi)

    def at(seconds):
        samples = seconds.double() / dt
        floor = samples.floor().long()
        shape = (1, *floor.shape, target.shape[-1], target.shape[-1])
        before = floor.clamp(0, len(path) - 1)[None, ..., None, None].expand(shape)
        after = (floor + 1).clamp(0, len(path) - 1)[None, ..., None, None].expand(shape)
        weight = (samples - floor)[..., None, None]
        return gaps.gather(0, before).squeeze(0).lerp(gaps.gather(0, after).squeeze(0), weight)

    start, end = at(lo), at(hi)
    lowest = torch.minimum(torch.minimum(start, end), gaps.masked_fill(~within[..., None, None], torch.inf).amin(0))
    highest = torch.maximum(torch.maximum(start, end), gaps.masked_fill(~within[..., None, None], -torch.inf).amax(0))
    return (lowest >= LAYER - 1e-4) | (highest <= -LAYER + 1e-4)


def _altitude_options(position, horizontal, selected, active, vertical_speed=None):
    """Compatibility of holding the selected altitude or changing by one layer."""
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    velocity = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    lo, hi, hit = _ball_interval(relative[..., :2], velocity, SCALE.horizontal_radius, SCALE.horizon)
    overlap = hit & active.unsqueeze(-1) & active.unsqueeze(-2)
    overlap &= ~torch.eye(active.shape[-1], dtype=torch.bool, device=active.device)
    altitude = position[..., 2]
    target = torch.stack((selected, altitude - LAYER, altitude + LAYER), dim=-1).clamp(0.0, 304.8)
    offset = (target - altitude.unsqueeze(-1)).clamp(-LAYER, LAYER)
    if vertical_speed is not None:
        clear = _native_vertical_clearance(altitude, altitude.unsqueeze(-1) + offset, vertical_speed, lo, hi)
    else:
        delta = offset.unsqueeze(-2).unsqueeze(-1) - offset.unsqueeze(-3).unsqueeze(-2)
        dz = relative[..., 2, None, None]
        start = dz + delta * (lo / 15.0).clamp(0, 1)[..., None, None]
        end = dz + delta * (hi / 15.0).clamp(0, 1)[..., None, None]
        clear = ((start >= LAYER - 1e-4) & (end >= LAYER - 1e-4)) | (
            (start <= -LAYER + 1e-4) & (end <= -LAYER + 1e-4)
        )
    return overlap, clear | ~overlap[..., None, None]


def _component_plan(allowed, members, *, choices=None):
    """Search only the component of the observed contact in the frozen forecast."""
    explored = 0

    def search(domains):
        nonlocal explored
        if not domains:
            return {}
        current = min(domains, key=lambda index: len(domains[index]))
        for choice in domains[current]:
            explored += 1
            if explored > 20_000:
                raise TimeoutError
            remaining = {index: [option for option in options if allowed[current][index][choice][option]]
                         for index, options in domains.items() if index != current}
            if any(not options for options in remaining.values()):
                continue
            assignment = search(remaining)
            if assignment is not None:
                return {current: choice, **assignment}
        return None

    domains = {index: [0, 1, 2] if choices is None else choices[index] for index in members}
    try:
        return search(domains), explored
    except TimeoutError:
        return None, explored


def _altitude_assignment(options, labels, env, i, j, overlap):
    if not overlap[env, i, j]:
        return {"status": "no_horizontal_overlap"}
    members = torch.where(labels[env] == labels[env, i])[0].tolist()
    assignment, explored = _component_plan(options[env].tolist(), members)
    status = "search_limit" if explored > 20_000 else "infeasible"
    if assignment is not None:
        status = "feasible"
    return {"status": status, "component_size": len(members), "examined_assignments": explored, "choices": assignment}


def _risk_clearances(state, action):
    planned = _forecast_command(state, action, dense=False)
    horizontal = planned.position[..., :2]
    velocity = planned.direction[..., :2] * planned.ground_speed.unsqueeze(-1)
    relative = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    change = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    for horizon in (60.0, 120.0):
        distance, time, _ = closest_approach(relative, change, horizon)
        at_cpa = _vertical_difference(planned, time).abs()
        gap = _overlap_vertical_gap(planned, relative, change, horizon, at_cpa)
        yield horizon, distance, at_cpa, gap


def _executed_pair_forecast(state, controls, env, pair):
    forecast = _forecast_command(state, controls, dense=False)
    point = state.position[env, pair, :2]
    velocity = forecast.direction[env, pair] * forecast.ground_speed[env, pair, None]
    relative, change = point[0] - point[1], velocity[0] - velocity[1]
    horizontal, closest_time, _ = closest_approach(relative, change, SCALE.horizon)
    entry, exit, overlap = _ball_interval(relative, change, SCALE.horizontal_radius, SCALE.horizon)
    result = dict(horizontal_min_m=float(horizontal), horizontal_cpa_s=float(closest_time),
                  horizontal_entry_s=float(entry) if overlap else None,
                  horizontal_exit_s=float(exit) if overlap else None, vertical_min_m=None)
    if overlap:
        first, last = ceil(float(entry) / DT), floor(float(exit) / DT)
        if first <= last:
            paths = altitude_paths(state.position[env, pair, 2], state.vertical_speed[env, pair],
                                   state.selected_altitude[env, pair], controls[env, pair, 1, None], last)
            result["vertical_min_m"] = float((paths[0, 0, first:] - paths[1, 0, first:]).abs().min())
    return result


def _classify(physical, active, adjacency, raw, tuning, contacts, *, altitude_oracle=False, native_vertical=False,
              risk_controls=None, detection_metres=SCALE.detection):
    state = PhysicalState(*physical)
    increment = CAS_DELTA * (state.tas / state.cas.clamp_min(1)).clamp(0.8, 2.0)
    desired_speed = state.ground_speed + increment * raw[..., 0]
    dual, rho, mobility, altitude_dual, altitude_rho = tuning
    speed = speed_admm(state.position, state.direction, desired_speed, active, adjacency,
                      minimum_speed=(state.ground_speed - increment).clamp_min(0),
                      maximum_speed=state.ground_speed + increment, horizon=SCALE.horizon,
                      radius=SCALE.horizontal_radius, margin=20, vertical_radius=LAYER,
                      vertical_rate=state.vertical_speed, rho=rho, mobility=mobility, warm_dual=dual, iterations=2)
    held = state.selected_altitude - state.position[..., 2]
    height = torch.where(raw[..., 1] != 0, LAYER * raw[..., 1], held)
    horizontal = state.direction[..., :2] * speed.speed.unsqueeze(-1)
    options = dict(minimum_altitude=0.0, maximum_altitude=304.8, iterations=2, rho=altitude_rho,
                   mobility=mobility, warm_dual=altitude_dual, colored_ties=True)
    k2 = altitude_admm(state.position, horizontal, height, active, **options)
    k4 = altitude_admm(state.position, horizontal, height, active, extra_rounds=2, **options)
    base_velocity = state.direction[..., :2] * state.ground_speed.unsqueeze(-1)
    if altitude_oracle:
        overlap, allowed = _altitude_options(state.position, horizontal, state.selected_altitude, active)
        labels = connected_components(overlap)
        velocity = torch.cat((base_velocity, state.vertical_speed.unsqueeze(-1)), dim=-1)
        wider_graph = conflict_graph(state.position, velocity, active, detection=50_000.0, speed_slack=SCALE.speed_unit)
        if native_vertical:
            _, allowed_native = _altitude_options(state.position, horizontal, state.selected_altitude, active,
                                                   vertical_speed=state.vertical_speed)
    if risk_controls is not None:
        forecasts = {"raw": tuple(_risk_clearances(state, raw)), "executed": tuple(_risk_clearances(state, risk_controls))}
    for row in contacts:
        env, (i, j) = row["env"], row["pair"]
        distance, time, _ = closest_approach(state.position[env, i, :2] - state.position[env, j, :2],
                                             base_velocity[env, i] - base_velocity[env, j], SCALE.horizon)
        upper = distance + time * (increment[env, i] + increment[env, j])
        predicted_vertical = abs(float(state.position[env, i, 2] - state.position[env, j, 2]
                                       + time * (state.vertical_speed[env, i] - state.vertical_speed[env, j])))
        row["speed_only_infeasible_bound"] = bool(upper < SCALE.horizontal_radius and predicted_vertical < LAYER)
        row["speed_k2_conflict"] = bool(speed.conflicts[env, i, j])
        row["altitude_k2_conflict"] = bool(k2.conflicts[env, i, j])
        row["altitude_adaptive_conflict"] = bool(k4.conflicts[env, i, j])
        row["extra_rounds_clear_prediction"] = row["altitude_k2_conflict"] and not row["altitude_adaptive_conflict"]
        row["descent_into_separation_band"] = (
            row["initial_vertical_m"] >= LAYER - 1e-4 and min(row["vertical_rates_mps"]) < -0.05
        )
        row["observed_by_conflict_graph"] = bool(adjacency[env, i, j])
        if risk_controls is not None:
            row["executed_pair_forecast"] = _executed_pair_forecast(state, risk_controls, env, (i, j))
            row["overlap_risk"] = {}
            for name, risk_by_horizon in forecasts.items():
                row["overlap_risk"][name] = {
                    str(int(horizon)): {"horizontal_min_m": float(distance[env, i, j]),
                                       "vertical_at_horizontal_cpa_m": float(cpa[env, i, j]),
                                       "vertical_during_horizontal_overlap_m": float(gap[env, i, j])}
                    for horizon, distance, cpa, gap in risk_by_horizon
                }
        if altitude_oracle:
            distance_now = (state.position[env, i, :2] - state.position[env, j, :2]).norm()
            row["observed_with_35km"] = bool(wider_graph.adjacency[env, i, j] & (distance_now <= 35_000.0))
            row["observed_with_50km"] = bool(wider_graph.adjacency[env, i, j])
            row["nominal_conflict_within_50km"] = bool(wider_graph.existing[env, i, j])
            row["potential_conflict_within_50km"] = bool(wider_graph.potential[env, i, j])
            row["nominal_horizontal_overlap_within_50km"] = bool(
                (distance_now <= 50_000.0) & (wider_graph.closest_horizontal[env, i, j] < SCALE.horizontal_radius)
            )
            relative = state.position[env, i] - state.position[env, j]
            relative_velocity = velocity[env, i] - velocity[env, j]
            _, _, nominal_hit = cylinder_interval(relative, relative_velocity, SCALE.horizontal_radius, LAYER, SCALE.horizon)
            end_altitude = relative[2] + SCALE.horizon * relative_velocity[2]
            minimum_vertical = torch.where(relative[2] * end_altitude <= 0, 0.0,
                                           torch.minimum(relative[2].abs(), end_altitude.abs()))
            reach = SCALE.horizontal_radius + 2 * SCALE.speed_unit * SCALE.horizon
            row["observation_geometry"] = {
                "distance_m": float(distance_now), "detection_m": detection_metres,
                "inside_detection": bool(distance_now <= detection_metres),
                "nominal_cylinder_hit_without_range_limit": bool(nominal_hit),
                "nominal_horizontal_min_m": float(distance),
                "nominal_horizontal_cpa_s": float(time),
                "minimum_vertical_in_horizon_m": float(minimum_vertical),
                "potential_horizontal_reachable": bool(distance < reach),
                "potential_vertical_reachable": bool(minimum_vertical < 2 * LAYER),
            }
            row["discrete_altitude"] = _altitude_assignment(allowed, labels, env, i, j, overlap)
            if native_vertical:
                row["discrete_altitude_with_inertia"] = _altitude_assignment(allowed_native, labels, env, i, j, overlap)


@torch.inference_mode()
def run(args):
    from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
    from envs.bluesky_plan_a import GeoNativeBlueSky

    layer_options = dict(early_layer_search=getattr(args, "early_layer_search", False),
                         exact_layer_forecast=getattr(args, "exact_layer_forecast", False),
                         batched_layer_margin=getattr(args, "batched_layer_margin", 0.))
    if not isfinite(layer_options["batched_layer_margin"]) or layer_options["batched_layer_margin"] < 0:
        raise ValueError("Layer margin must be finite and nonnegative")
    if any(layer_options.values()) and not args.batched_layer_search:
        raise ValueError("Layer forecast refinements require batched layer search")
    if args.without_projection and (args.batched_layer_search or args.joint_layer_search):
        raise ValueError("Layer search requires projection")
    if args.joint_layer_search and not args.batched_layer_search:
        raise ValueError("Three-aircraft layer search requires batched layer search")
    if not isfinite(args.joint_layer_reserve) or args.joint_layer_reserve < 0 or (
            args.joint_layer_reserve and not args.joint_layer_search):
        raise ValueError("The reserve must be nonnegative and requires three-aircraft layer search")
    plan = json.loads(args.layer_plan.read_text()) if args.layer_plan else None
    if plan is not None:
        start, end = plan["start_decision"], plan["end_decision"]
        offsets = {int(slot): offset for slot, offset in plan["initial_offsets"].items()}
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end:
            raise ValueError("The layer plan needs a valid inclusive control interval")
        if not offsets or any(not 0 <= slot < 12 or not -1 <= offset <= 1 for slot, offset in offsets.items()):
            raise ValueError("Layer overrides must identify aircraft slots and bounded altitude increments")
        if (args.family != plan["family"] or args.seed != plan["seed"]
                or _checksum(args.checkpoint) != plan["checkpoint_sha256"]
                or plan["projection"] != (not args.without_projection)
                or plan["batched_layer_search"] != args.batched_layer_search
                or plan["joint_layer_search"] != args.joint_layer_search
                or plan["joint_layer_reserve_m"] != args.joint_layer_reserve
                or any(plan.get("layer_safety_options", {}).get(key, False) != value
                       for key, value in layer_options.items())):
            raise ValueError("The layer plan must match the traced checkpoint and traffic")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    navdata = getattr(args, "navdata", None)
    if navdata is not None:
        cache = args.output / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(navdata)
    actor = _load(args.checkpoint, torch.device("cpu"))
    equivariant = getattr(args, "equivariant_actor", None)
    if equivariant is not None:
        actor.equivariant_actor = equivariant
    if plan is not None and plan.get("equivariant_actor", False) != actor.equivariant_actor:
        raise ValueError("The layer plan must use the same Actor forward mode")
    env = GeoNativeBlueSky(BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=12, sim_dt=.05, control_dt=6,
                                              hdg_deadband_deg=.25, seed=args.seed),
                           workdir=args.output, family=args.family,
                           detection_metres=actor.observation_profile.detection_metres,
                           geodesic_navigation=actor.observation_profile.geodesic_navigation,
                           cross_track_observation=actor.observation_profile.cross_track_observation,
                           far_conflict_observation=actor.observation_profile.far_conflict_observation,
                           selected_altitude_observation=actor.observation_profile.selected_altitude_observation)
    enable_rl_fast_mode(False, simdt=.05)
    graph = env.reset()
    summary = Counter()
    history = deque(maxlen=21)
    saved_override = False
    capture = set(getattr(args, "capture_decisions", ()))
    if any(decision < 0 or decision >= env.cfg.max_steps for decision in capture):
        raise ValueError("Capture decisions must fall within the simulated episode")
    with ExitStack() as files:
        log = files.enter_context((args.output / "collision-events.jsonl").open("w"))
        controls_log = (files.enter_context((args.output / "control-history.jsonl").open("w"))
                        if args.control_history else None)
        for decision in range(env.cfg.max_steps):
            policy = actor(*graph)
            action = encode_action(policy)
            tuning = policy.consensus_inputs
            physical, adjacency, active = env.physical, graph[2], graph[-1]
            control = action[..., 4:]
            if not args.without_projection:
                control, _ = protected_command(
                    action, physical, adjacency, active, tuning, 2, 0, directional_heading=True, extra_rounds=2,
                    batched_layer_search=args.batched_layer_search, joint_layer_search=args.joint_layer_search,
                    joint_layer_reserve=args.joint_layer_reserve, **layer_options,
                )
            if decision in capture:
                torch.save({"decision": decision, "graph": graph, "physical": physical,
                            "proposal": action, "original_commands": control},
                           args.output / f"decision-{decision:04d}.pt")
                capture.remove(decision)
            if plan is not None and start <= decision <= end:
                original_control = control
                control = original_control.clone()
                for slot, offset in offsets.items():
                    control[0, slot, 1] = offset if decision == start else 0.0
                if args.control_history and not saved_override and (control != original_control).any():
                    torch.save({
                        "decision": decision, "graph": graph, "physical": physical, "proposal": action,
                        "original_commands": original_control, "committed_commands": control,
                    }, args.output / "first-override-state.pt")
                    saved_override = True
            if controls_log is not None:
                slots = active[0].nonzero().flatten()
                height = physical[0][0, slots, 2]
                print(json.dumps({
                    "decision": decision, "slots": slots.tolist(), "xy_m": physical[0][0, slots, :2].tolist(),
                    "altitude_m": height.tolist(), "selected_altitude_m": physical[-1][0, slots].tolist(),
                    "vertical_rate_mps": physical[3][0, slots].tolist(),
                    "raw": action[0, slots, 4:].tolist(), "commands": control[0, slots].tolist(),
                }), file=controls_log)
            history.append((physical, active, adjacency, action[..., 4:], tuning, control))
            graph, _, done, events = env.step_native_audited(control.numpy(), track_contacts=True)
            contacts = events["contacts"]
            if contacts:
                diagnostics = {"altitude_oracle": args.altitude_oracle or args.native_vertical,
                               "native_vertical": args.native_vertical,
                               "detection_metres": actor.observation_profile.detection_metres}
                _classify(physical, active, adjacency, action[..., 4:], tuning, contacts,
                          risk_controls=control if args.overlap_risk_audit else None, **diagnostics)
                for contact in contacts:
                    contact["earlier_forecasts"] = []
                for decisions_back in (5, 10, 15, 20):
                    if len(history) <= decisions_back:
                        continue
                    *previous, previous_control = history[-1 - decisions_back]
                    forecasts = [{key: row[key] for key in ("env", "pair", "initial_vertical_m", "vertical_rates_mps")}
                                 for row in contacts]
                    _classify(*previous, forecasts, risk_controls=previous_control if args.overlap_risk_audit else None,
                              **diagnostics)
                    for contact, item in zip(contacts, forecasts, strict=True):
                        env_index, pair = contact["env"], contact["pair"]
                        item.pop("descent_into_separation_band")
                        earlier = {
                            "decision": decision - decisions_back,
                            "seconds_before_contact": decisions_back * env.cfg.control_dt + contact["frame"] * env.cfg.sim_dt,
                            "planned_commands": previous_control[env_index, pair].tolist(),
                            **{key: val for key, val in item.items() if isinstance(val, bool)},
                        }
                        for key in ("discrete_altitude", "discrete_altitude_with_inertia", "overlap_risk",
                                    "observation_geometry", "executed_pair_forecast"):
                            if key in item:
                                earlier[key] = item[key]
                        contact["earlier_forecasts"].append(earlier)
                for contact in contacts:
                    print(json.dumps({"decision": decision, **contact}), file=log)
                    summary["contact_pairs"] += 1
                    summary.update(key for key, value in contact.items() if isinstance(value, bool) and value)
            for key in ("collisions", "arrivals", "exits", "timeouts"):
                summary[key] += int(events[key].sum())
            if done[0]:
                break
    if capture:
        raise ValueError(f"Episode ended before capture decisions {sorted(capture)}")
    (args.output / "summary.json").write_text(json.dumps(dict(summary), indent=2) + "\n")
    (args.output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "sha256": _checksum(args.checkpoint), "family": args.family, "seed": args.seed,
        "sim_dt": env.cfg.sim_dt, "control_dt": env.cfg.control_dt, "projection": not args.without_projection,
        "batched_layer_search": args.batched_layer_search, "joint_layer_search": args.joint_layer_search,
        "joint_layer_reserve_m": args.joint_layer_reserve, "layer_safety_options": layer_options,
        "equivariant_actor": actor.equivariant_actor, "detection_metres": actor.observation_profile.detection_metres,
        "navdata_sha256": _checksum(navdata) if navdata is not None else None,
        "layer_intervention": plan,
    }, indent=2) + "\n")
    print(json.dumps(dict(summary)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--family", choices=FAMILIES, required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--altitude-oracle", action="store_true")
    cli.add_argument("--native-vertical", action="store_true", help="Also search using native vertical dynamics")
    cli.add_argument("--overlap-risk-audit", action="store_true")
    cli.add_argument("--without-projection", action="store_true")
    cli.add_argument("--batched-layer-search", action="store_true")
    cli.add_argument("--joint-layer-search", action="store_true")
    cli.add_argument("--joint-layer-reserve", type=float, default=0.0)
    cli.add_argument("--early-layer-search", action="store_true")
    cli.add_argument("--exact-layer-forecast", action="store_true")
    cli.add_argument("--batched-layer-margin", type=float, default=0.0)
    cli.add_argument("--equivariant-actor", action=argparse.BooleanOptionalAction, default=None)
    cli.add_argument("--navdata", type=Path)
    cli.add_argument("--control-history", action="store_true")
    cli.add_argument("--capture-decisions", type=int, nargs="+", default=[],
                     help="Save graph and physical state at these zero-based control decisions")
    cli.add_argument("--layer-plan", type=Path,
                     help="Replay explicitly recorded altitude targets on a diagnostic trajectory")
    run(cli.parse_args())
