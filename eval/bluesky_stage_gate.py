"""Six-family Stage 1 development gate on the calibrated Torch BlueSky backend."""

import argparse
import importlib.util
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from core.conflict_graph import conflict_graph
from envs.bluesky_graph import SCALE, with_far_horizontal_observation
from envs.bluesky_safety import protected_command
from eval.bluesky_pair import FAMILIES, _action_cost, _checksum, _load
from train.td3 import encode_action


@torch.inference_mode()
def run(args):
    device = torch.device("cuda:0" if args.device == "ppu" else "cpu")
    if args.device == "ppu" and importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    torch.set_num_threads(1)
    actor = _load(args.checkpoint, device)
    trained = actor.observation_profile
    detection = trained.detection_metres if args.detection_km is None else 1000 * args.detection_km
    if not isfinite(detection) or detection <= 0:
        raise ValueError("Detection distance must be positive")
    protection_detection = detection if args.protection_detection_km is None else 1000 * args.protection_detection_km
    if not isfinite(protection_detection) or protection_detection <= 0:
        raise ValueError("Protection distance must be positive")
    from bluesky.rl_gpu import B737PerfConfig, TorchVecEnvConfig
    from envs.bluesky_plan_b import GeoTorchBlueSky

    measured = json.loads(args.calibration.read_text())
    keys = ("axmax_mps2", "bank_def_rad", "vs_def_mps")
    perf = B737PerfConfig(**{key: measured[key] for key in keys})
    requested_geodesic = getattr(args, "geodesic_navigation", None)
    geodesic_navigation = trained.geodesic_navigation if requested_geodesic is None else requested_geodesic
    far_horizontal = getattr(args, "far_horizontal_observation", False)
    layer_fallback = getattr(args, "layer_fallback", False)
    batched_layer = getattr(args, "batched_layer_search", False)
    batched_layer_margin = getattr(args, "batched_layer_margin", 0.0)
    accelerated_layer = getattr(args, "accelerated_layer_forecast", False)
    preserve_imminent = getattr(args, "preserve_imminent_clearance", False)
    early_layer = getattr(args, "early_layer_search", False)
    exact_layer = getattr(args, "exact_layer_forecast", False)
    joint_layer = getattr(args, "joint_layer_search", False)
    joint_reserve = getattr(args, "joint_layer_reserve", 0.0)
    native_tolerance = getattr(args, "native_contact_tolerance", False)
    if not isfinite(batched_layer_margin) or batched_layer_margin < 0 or (batched_layer_margin and not batched_layer):
        raise ValueError("Vertical clearance margin must be finite and nonnegative and requires batched layer search")
    if (accelerated_layer or preserve_imminent or early_layer or exact_layer or joint_layer) and not batched_layer:
        raise ValueError("Layer forecast refinements require batched layer search")
    if not isfinite(joint_reserve) or joint_reserve < 0 or (joint_reserve and not joint_layer):
        raise ValueError("The reserve must be nonnegative and requires three-aircraft search")
    if batched_layer and (layer_fallback or args.without_projection):
        raise ValueError("Batched discrete layer search requires projection and runs independently of the CPU solver")
    project = protected_command
    if layer_fallback:
        if device.type != "cpu" or args.without_projection:
            raise ValueError("Discrete layer evaluation requires CPU with projection enabled")
        from eval.bluesky_layer_fallback import protected_with_discrete_layers
        project = protected_with_discrete_layers
    projection_options = {
        "batched_layer_search": True, "batched_layer_margin": batched_layer_margin,
        "accelerated_layer_forecast": accelerated_layer,
        "preserve_imminent_clearance": preserve_imminent,
        "early_layer_search": early_layer, "exact_layer_forecast": exact_layer,
        "joint_layer_search": joint_layer,
        "joint_layer_reserve": joint_reserve,
    } if batched_layer else {}
    diagnostic_range = detection != trained.detection_metres or protection_detection != trained.detection_metres
    diagnostic = diagnostic_range or geodesic_navigation != trained.geodesic_navigation or far_horizontal
    report = {"checkpoint": str(args.checkpoint), "sha256": _checksum(args.checkpoint),
              "event_sampling_seconds": args.sim_dt, "first_seed": args.seed, "families": {},
              "projection_enabled": not args.without_projection, "detection_metres": detection,
              "protection_detection_metres": protection_detection,
              "geodesic_navigation": geodesic_navigation,
              "cross_track_observation": trained.cross_track_observation,
              "far_conflict_observation": trained.far_conflict_observation,
              "selected_altitude_observation": trained.selected_altitude_observation,
              "far_horizontal_observation": far_horizontal,
              "diagnostic_projection": layer_fallback or batched_layer,
              "batched_layer_search": batched_layer,
              "batched_layer_margin_m": batched_layer_margin,
              "accelerated_layer_forecast": accelerated_layer,
              "preserve_imminent_clearance": preserve_imminent,
              "early_layer_search": early_layer, "exact_layer_forecast": exact_layer,
              "joint_layer_search": joint_layer,
              "joint_layer_reserve_m": joint_reserve,
              "diagnostic_contact_tolerance": native_tolerance,
              "diagnostic_observation_range": diagnostic_range,
              "diagnostic_observation_profile": diagnostic}
    start = perf_counter()
    fields = ("arrivals", "collisions", "exits", "timeouts")
    for index, family in enumerate(FAMILIES):
        cfg = TorchVecEnvConfig(num_envs=args.envs, max_aircraft=args.aircraft, sim_dt=args.sim_dt,
                               control_dt=6, max_steps=args.max_steps, hdg_deadband_deg=0.25,
                               device=str(device), seed=args.seed + 10_007 * index, dtype=torch.float64)
        env = GeoTorchBlueSky(cfg, family=family, perf=perf, audit_physics=True, detection_metres=detection,
                              geodesic_navigation=geodesic_navigation,
                              cross_track_observation=trained.cross_track_observation,
                              far_conflict_observation=trained.far_conflict_observation,
                              selected_altitude_observation=trained.selected_altitude_observation,
                              native_contact_tolerance=native_tolerance)
        graph = env.reset()
        live = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
        totals = torch.zeros(4, dtype=torch.long, device=device)
        total_cost = torch.zeros((), device=device)
        interventions = torch.zeros((), dtype=torch.long, device=device)
        for _ in range(args.max_steps):
            observation = with_far_horizontal_observation(graph, env.physical, detection) if far_horizontal else graph
            policy = actor(*observation)
            action = encode_action(policy)
            commands = action[..., 4:]
            if not args.without_projection:
                tuning = policy.consensus_inputs
                adjacency = graph[2]
                if protection_detection != detection:
                    position, direction, ground_speed, vertical, *_ = env.physical
                    velocity = torch.cat((direction[..., :2] * ground_speed.unsqueeze(-1), vertical.unsqueeze(-1)),
                                         dim=-1)
                    adjacency = conflict_graph(position, velocity, graph[-1], horizon=SCALE.horizon,
                                               detection=protection_detection, horizontal_radius=SCALE.horizontal_radius,
                                               speed_slack=SCALE.speed_unit).adjacency
                commands, _ = project(action, env.physical, adjacency, graph[3], tuning, 2, 0,
                                      directional_heading=True, extra_rounds=2, **projection_options)
            active = graph[-1] & live.unsqueeze(-1)
            total_cost += _action_cost(commands).masked_fill(~active, 0.0).sum()
            interventions += (((commands - action[..., 4:]).square().sum(dim=-1) > 1e-5) & active).sum()
            graph, _, done, info = env.step(commands)
            totals += torch.stack([(info[key] * live).sum() for key in fields])
            live &= ~done
            if not live.any():
                break
        report["families"][family] = dict(zip(fields, totals.tolist(), strict=True))
        report["families"][family]["aircraft"] = cfg.num_envs * cfg.max_aircraft
        report["families"][family]["action_cost_per_aircraft"] = float(total_cost) / (cfg.num_envs * cfg.max_aircraft)
        report["families"][family]["admm_interventions"] = int(interventions)
        print(json.dumps({"family": family, **report["families"][family]}), flush=True)
    aircraft = sum(x["aircraft"] for x in report["families"].values())
    arrival_rate = sum(x["arrivals"] for x in report["families"].values()) / aircraft
    collision_rate = sum(x["collisions"] for x in report["families"].values()) / aircraft
    report.update(arrival_rate=arrival_rate, collision_rate=collision_rate, seconds=perf_counter() - start,
                  eligible_for_stage2=not args.without_projection and not diagnostic
                  and not (layer_fallback or batched_layer or native_tolerance)
                  and arrival_rate > 0.90 and collision_rate < 0.03)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--aircraft", type=int, default=12)
    parser.add_argument("--seed", type=int, default=43209)
    parser.add_argument("--sim-dt", type=float, default=0.25)
    parser.add_argument("--detection-km", type=float,
                        help="Override the observation range for a development diagnostic")
    parser.add_argument("--protection-detection-km", type=float,
                        help="Override the projection graph range in a development diagnostic")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", choices=("ppu", "cpu"), default="ppu")
    parser.add_argument("--without-projection", action="store_true")
    parser.add_argument("--geodesic-navigation", action=argparse.BooleanOptionalAction, default=None,
                        help="Use a geodesic goal bearing in the diagnostic actor observation")
    parser.add_argument("--far-horizontal-observation", action="store_true",
                        help="Add one early horizontal conflict candidate per aircraft to diagnostic actor attention")
    parser.add_argument("--layer-fallback", action="store_true",
                        help="Use CPU discrete altitude search for projection diagnostics")
    parser.add_argument("--batched-layer-search", action="store_true",
                        help="Use batched discrete altitude search for projection diagnostics")
    parser.add_argument("--native-contact-tolerance", action="store_true",
                        help="Use native numeric tolerance at the 300 ft boundary for diagnostics")
    parser.add_argument("--batched-layer-margin", type=float, default=0.0,
                        help="Prefer this vertical clearance in metres beyond the nominal 300 ft separation")
    parser.add_argument("--accelerated-layer-forecast", action="store_true",
                        help="Check the acceleration and capture phases in the batched layer forecast")
    parser.add_argument("--preserve-imminent-clearance", action="store_true",
                        help="Preserve vertical clearance for existing unresolved pairs within 30 seconds")
    parser.add_argument("--early-layer-search", action="store_true")
    parser.add_argument("--exact-layer-forecast", action="store_true")
    parser.add_argument("--joint-layer-search", action="store_true",
                        help="Try a joint three-aircraft repair after pair search")
    parser.add_argument("--joint-layer-reserve", type=float, default=0.0,
                        help="Retain this existing vertical buffer in metres during three-aircraft repair")
    run(parser.parse_args())
