"""Evaluate Geo-ADMM checkpoints on paired, native 0.05 s BlueSky traffic."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from math import ceil, isclose, isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter
from warnings import warn

import torch

from envs.bluesky_graph import ObservationProfile
from envs.bluesky_traffic import HELD_TOPOLOGIES
from envs.bluesky_safety import protected_command
from envs.bluesky_workers import NativeWorkers, SEPARATION_FIELDS, WorkerConfig
from eval.admm_communication import PACKET_SEED_OFFSET, DelayedPairExchange, PairExchange
from eval.admm_gap_mpc import plan_gap_maneuver
from eval.graph_statistics import GraphStatistics
from models.graphformer import Actor
from train.td3 import encode_action


FAMILIES = ("crossing3", "crossing4", "crossing6", "head_on", "yaw_plus15", "yaw_minus15")


def _action_cost(commands):
    return (commands.abs() * commands.new_tensor([1.0, 2.0, 4.0])).sum(dim=-1)


def _evaluation_profile(actor, args):
    trained = getattr(actor, "observation_profile", ObservationProfile())
    detected = getattr(args, "detection_metres", None)
    geodesic = getattr(args, "geodesic_navigation", None)
    return ObservationProfile(trained.detection_metres if detected is None else detected,
                              trained.geodesic_navigation if geodesic is None else geodesic,
                              trained.cross_track_observation, trained.far_conflict_observation,
                              trained.selected_altitude_observation)


def _communication_interval(period, control_dt):
    if period is None:
        return 1
    if not isfinite(period) or not isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Communication and control periods must be finite and positive")
    ticks = round(period / control_dt)
    if ticks < 1 or not isclose(ticks * control_dt, period, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Communication period must be a whole number of control ticks")
    return ticks


@torch.inference_mode()
def _measure_steps(actor, directory, args, device, family, iterations):
    profile = _evaluation_profile(actor, args)
    project = protected_command
    batched_layer = getattr(args, "batched_layer_search", False)
    batched_layer_margin = getattr(args, "batched_layer_margin", 0.0)
    accelerated_layer = getattr(args, "accelerated_layer_forecast", False)
    preserve_imminent = getattr(args, "preserve_imminent_clearance", False)
    early_layer = getattr(args, "early_layer_search", False)
    exact_layer = getattr(args, "exact_layer_forecast", False)
    joint_layer = getattr(args, "joint_layer_search", False)
    joint_reserve = getattr(args, "joint_layer_reserve", 0.0)
    if not isfinite(batched_layer_margin) or batched_layer_margin < 0 or (batched_layer_margin and not batched_layer):
        raise ValueError("Vertical clearance margin must be finite and nonnegative and requires batched layer search")
    if (accelerated_layer or preserve_imminent or early_layer or exact_layer or joint_layer) and not batched_layer:
        raise ValueError("Layer forecast refinements require batched layer search")
    if not isfinite(joint_reserve) or joint_reserve < 0 or (joint_reserve and not joint_layer):
        raise ValueError("The reserve must be nonnegative and requires three-aircraft search")
    if iterations and getattr(args, "layer_fallback", False):
        if batched_layer:
            raise ValueError("Choose either batched or CPU discrete layer search")
        if device.type != "cpu":
            raise ValueError("Discrete layer evaluation requires CPU")
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
    loss_rate = getattr(args, "admm_packet_loss", None)
    communication_period = getattr(args, "admm_communication_period", None)
    delay = getattr(args, "admm_delay_steps", None)
    interval = _communication_interval(communication_period, args.control_dt)
    gap_policy = getattr(args, "admm_gap_policy", "hold")
    exchange = None
    if loss_rate is not None or communication_period is not None or delay is not None:
        exchange_args = (0.0 if loss_rate is None else loss_rate, args.seed, args.seeds, device)
        exchange = DelayedPairExchange(*exchange_args, delay) if delay else PairExchange(*exchange_args)
    if exchange is not None:
        projection_options["exchange"] = exchange
        projection_options["observe_residuals"] = exchange.observe_residuals
    if getattr(args, "admm_schedule", "jacobi") == "colored":
        projection_options["colored_updates"] = True
    cfg = WorkerConfig(1, args.aircraft, 0.05, args.control_dt, args.max_steps, args.seed, True, family,
                       not args.control_frame_diagnostics, detection_metres=profile.detection_metres,
                       geodesic_navigation=profile.geodesic_navigation,
                       cross_track_observation=profile.cross_track_observation,
                       far_conflict_observation=profile.far_conflict_observation,
                       selected_altitude_observation=profile.selected_altitude_observation,
                       navdata=getattr(args, "navdata", None),
                       include_potential=not getattr(args, "without_potential_edges", False))
    directory.mkdir()
    env = NativeWorkers(args.seeds, cfg, directory, device, seed_stride=1)
    try:
        graph = env.reset()
        live = torch.ones(args.seeds, device=device, dtype=torch.bool)
        results = torch.zeros(args.seeds, 6, device=device)
        gap_mpc = torch.zeros(args.seeds, 4, dtype=torch.long, device=device) if gap_policy == "mpc" else None
        statistics = [GraphStatistics() for _ in range(args.seeds)] if getattr(args, "graph_statistics", False) else None
        track_recovery = getattr(args, "track_yaw_recovery", False)
        recovery_tick = torch.zeros_like(graph[-1], dtype=torch.long) if track_recovery else None
        aligned_ticks = torch.zeros_like(recovery_tick) if track_recovery else None
        separation = {name: torch.full(graph[-1].shape, torch.inf, dtype=torch.float64, device=device)
                      for name in SEPARATION_FIELDS} if cfg.audit_frames else {}
        decision_ms = []
        cycle_ms = []
        start = perf_counter()
        for step in range(args.max_steps):
            if isinstance(exchange, DelayedPairExchange):
                exchange.begin_step(step, graph[-1], live)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decision_start = perf_counter()
            output = actor(*graph)
            action = encode_action(output)
            commands = action[..., 4:]
            scheduled = iterations and step % interval == 0
            if scheduled:
                if exchange is not None:
                    exchange.live = live
                    exchange.coordination_ticks.add_(live)
                tuning = None
                if not getattr(args, "cold_start_admm", False):
                    tuning = output.consensus_inputs
                commands, _ = project(action, env.physical, graph[2], graph[3], tuning, iterations,
                                      args.admm_fallback, by_component=args.component_projection,
                                      vertical_heading=args.vertical_heading,
                                      directional_heading=args.directional_heading,
                                      extra_rounds=args.dense_extra_rounds, **projection_options)
            elif iterations and gap_policy == "hold":
                # Zero increments leave the last advertised targets selected in BlueSky.
                commands = torch.zeros_like(commands)
            elif iterations and gap_mpc is not None:
                commands, before, after = plan_gap_maneuver(env.physical, graph[-1] & live.unsqueeze(-1), commands)
                adjusted = commands.abs().amax(dim=-1).gt(1e-5).sum(dim=-1)
                observations = torch.stack((live.long(), adjusted, before, after), dim=-1)
                gap_mpc.add_(observations)
            commands = commands.masked_fill(~graph[-1].unsqueeze(-1), 0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decision_ms.append((perf_counter() - decision_start) * 1000)
            changed = (commands - action[..., 4:]).square().sum(dim=-1) > 1e-5
            interventions = (changed & graph[-1]).sum(dim=-1) if scheduled else torch.zeros_like(live, dtype=torch.long)
            cost = _action_cost(commands)
            env.start_step(commands)
            if statistics is not None:
                for index, (adjacency, active, running) in enumerate(
                        zip(graph[2].cpu().numpy(), graph[-1].cpu().numpy(), live.tolist(), strict=True)):
                    if running:
                        statistics[index].add(adjacency, active)
            yield
            graph, reward, done, info = env.finish_step()
            if recovery_tick is not None:
                headings = torch.atan2(graph[0][..., 5], graph[0][..., 6]).abs()
                eligible = (live & ~done).unsqueeze(-1) & graph[-1]
                aligned = eligible & (headings <= 2.0 * torch.pi / 180)
                aligned_ticks = torch.where(aligned, aligned_ticks + 1, 0)
                first_recovery = (aligned_ticks >= 3) & (recovery_tick == 0)
                recovery_tick.masked_fill_(first_recovery, step + 1)
            if step == 0 and family in HELD_TOPOLOGIES:
                initial_routes = info["route_count"].tolist()
            for name, values in separation.items():
                separation[name] = torch.minimum(values, info[name].masked_fill(~live.unsqueeze(-1), torch.inf))
            results += torch.stack((info["arrivals"], info["collisions"], info["exits"],
                                    cost.masked_fill(~live.unsqueeze(-1), 0).sum(-1),
                                    torch.ones_like(reward[..., 0]), interventions), dim=-1) * live.unsqueeze(-1)
            live = live & ~done
            cycle_ms.append((perf_counter() - decision_start) * 1000)
            if not live.any():
                break
    finally:
        workers_stopped = env.close()
    archive = getattr(args, "navdata_archive", None)
    if workers_stopped and archive is not None:
        try:
            _archive_navdata(directory, archive)
        except OSError as error:
            warn(f"Could not share generated navigation caches: {error}", RuntimeWarning, stacklevel=2)
    rows = []
    recovery_times = recovery_tick.tolist() if recovery_tick is not None else None
    message_counts = exchange.report() if exchange is not None else None
    gap_rows = gap_mpc.tolist() if gap_mpc is not None else None
    distances = {name: values.tolist() for name, values in separation.items()}
    for seed_offset, (arrivals, collisions, exits, cost, decisions, interventions) in enumerate(results.tolist()):
        row = {"seed": args.seed + seed_offset, "aircraft": args.aircraft, "arrivals": int(arrivals),
               "collisions": int(collisions), "exits": int(exits),
               "unresolved": args.aircraft - int(arrivals) - int(collisions) - int(exits),
               "action_cost_per_aircraft": cost / args.aircraft, "control_steps": int(decisions),
               "admm_interventions": int(interventions)}
        if family in HELD_TOPOLOGIES:
            row["route_count"] = int(initial_routes[seed_offset])
        if recovery_times is not None:
            timings = [tick * args.control_dt if tick else None for tick in recovery_times[seed_offset]]
            row["yaw_recovery"] = dict(recovered_aircraft=sum(t is not None for t in timings),
                                       first_sustained_alignment_seconds=timings)
        for name, values in distances.items():
            finite = [value for value in values[seed_offset] if isfinite(value)]
            row[name] = min(finite) if finite else None
            row[f"mean_{name}"] = mean(finite) if finite else None
        if statistics is not None:
            row["conflict_graph"] = statistics[seed_offset].report()
        if message_counts is not None:
            row["communication"] = message_counts[seed_offset]
        if gap_rows is not None:
            keys = ("replans", "adjusted_aircraft", "hold_predicted_pair_contacts", "planned_predicted_pair_contacts")
            row["gap_mpc"] = dict(zip(keys, gap_rows[seed_offset], strict=True))
        rows.append(row)
    return {"seconds": perf_counter() - start, "admm_k": iterations,
            "admm_fallback": args.admm_fallback if iterations else 0,
            "mean_decision_ms_per_batch": mean(decision_ms),
            "p95_decision_ms_per_batch": sorted(decision_ms)[ceil(0.95 * len(decision_ms)) - 1],
            "mean_control_cycle_ms": mean(cycle_ms),
            "p95_control_cycle_ms": sorted(cycle_ms)[ceil(0.95 * len(cycle_ms)) - 1],
            "rows": rows}


def _measure_batch(jobs):
    """Overlap independent native workers while keeping actor inference on the calling device."""
    runs = [(label, _measure_steps(*arguments)) for label, arguments in jobs]
    pending = runs
    try:
        while pending:
            active = []
            for label, steps in pending:
                try:
                    next(steps)
                except StopIteration as completed:
                    yield label, completed.value
                else:
                    active.append((label, steps))
            pending = active
    finally:
        for _, steps in runs:
            steps.close()


def measure(actor, directory, args, device, family, iterations):
    (_, outcome), = _measure_batch([(None, (actor, directory, args, device, family, iterations))])
    return outcome


def _load(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(width=state["width"], equivariant_actor=state.get("equivariant_actor", False)).to(device)
    actor.load_state_dict(state["actor"])
    actor.observation_profile = ObservationProfile(**state.get("observation_profile", {}))
    return actor.eval()


def _checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_navdata(directory, archive):
    """Share byte-identical generated navigation caches after the workers have stopped."""
    archive.mkdir(exist_ok=True)
    for path in directory.glob("worker-*/cache/navdata.p"):
        canonical = archive / _checksum(path)
        try:
            canonical.hardlink_to(path)
        except FileExistsError:
            pass
        if canonical.samefile(path):
            continue
        temporary = path.with_suffix(".link")
        temporary.hardlink_to(canonical)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def compare(args):
    parallel_pairs = getattr(args, "parallel_pairs", 1)
    if parallel_pairs < 1:
        raise ValueError("Parallel native evaluations must be positive")
    if getattr(args, "track_yaw_recovery", False) and set(args.family) - {"yaw_plus15", "yaw_minus15"}:
        raise ValueError("Yaw recovery tracking requires the initial-yaw families")
    ablate_baseline = getattr(args, "ablate_baseline", False)
    if ablate_baseline and not (args.ablation and args.admm_k):
        raise ValueError("Baseline ablation requires paired projection and ablation")
    if args.device == "ppu" and importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    device = torch.device("cuda:0" if args.device == "ppu" else "cpu")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    policies = {"baseline": args.baseline, **{f"candidate_{i}": path for i, path in enumerate(args.candidates, 1)}}
    report = {
        "sim_dt": 0.05, "control_dt": args.control_dt, "first_seed": args.seed, "num_seeds": args.seeds,
        "event_sampling_seconds": args.control_dt if args.control_frame_diagnostics else 0.05,
        "component_projection": args.component_projection,
        "vertical_heading": args.vertical_heading,
        "directional_heading": args.directional_heading,
        "dense_extra_rounds": args.dense_extra_rounds,
        "without_potential_edges": getattr(args, "without_potential_edges", False),
        "diagnostic_projection": (getattr(args, "layer_fallback", False) or getattr(args, "batched_layer_search", False)
                                  or getattr(args, "cold_start_admm", False)
                                  or getattr(args, "without_potential_edges", False)
                                  or getattr(args, "admm_packet_loss", None) is not None
                                  or getattr(args, "admm_schedule", "jacobi") != "jacobi"
                                  or getattr(args, "admm_delay_steps", None) is not None
                                  or getattr(args, "admm_gap_policy", "hold") != "hold"
                                  or getattr(args, "admm_communication_period", None) is not None),
        "cold_start_admm": getattr(args, "cold_start_admm", False),
        "admm_packet_loss": getattr(args, "admm_packet_loss", None),
        "admm_delay_steps": getattr(args, "admm_delay_steps", None),
        "admm_schedule": getattr(args, "admm_schedule", "jacobi"),
        "admm_communication_period": getattr(args, "admm_communication_period", None),
        "admm_gap_policy": getattr(args, "admm_gap_policy", "hold"),
        "admm_communication_seed_offset": PACKET_SEED_OFFSET if getattr(args, "admm_packet_loss", None) is not None else None,
        "batched_layer_search": getattr(args, "batched_layer_search", False),
        "batched_layer_margin_m": getattr(args, "batched_layer_margin", 0.0),
        "accelerated_layer_forecast": getattr(args, "accelerated_layer_forecast", False),
        "preserve_imminent_clearance": getattr(args, "preserve_imminent_clearance", False),
        "early_layer_search": getattr(args, "early_layer_search", False),
        "exact_layer_forecast": getattr(args, "exact_layer_forecast", False),
        "joint_layer_search": getattr(args, "joint_layer_search", False),
        "joint_layer_reserve_m": getattr(args, "joint_layer_reserve", 0.0),
        "parallel_pairs": parallel_pairs,
        "graph_statistics": getattr(args, "graph_statistics", False),
        "yaw_recovery_protocol": ({"max_goal_heading_error_deg": 2, "consecutive_control_ticks": 3}
                                  if getattr(args, "track_yaw_recovery", False) else None),
        "sources": {name: {"path": str(path), "sha256": _checksum(path)}
                    for name, path in policies.items()}, "families": {},
    }
    actors = {name: _load(path, device) for name, path in policies.items()}
    profiles = {name: _evaluation_profile(actor, args) for name, actor in actors.items()}
    for name, profile in profiles.items():
        report["sources"][name]["observation_profile"] = asdict(profile)
        equivariant = getattr(args, "equivariant_actor", None)
        if equivariant is not None:
            actors[name].equivariant_actor = equivariant
        report["sources"][name]["equivariant_actor"] = actors[name].equivariant_actor
        if actors[name].equivariant_actor:
            report["sources"][name]["admm_warmstart_reflection"] = "even"
    ranges = {profile.detection_metres for profile in profiles.values()}
    report["detection_metres"] = next(iter(ranges)) if len(ranges) == 1 else None
    start = perf_counter()
    for family in args.family:
        results = {}
        modes = [(name, name, args.admm_k) for name in policies]
        if args.admm_k and args.ablation:
            modes.extend((f"{name}_without_projection", name, 0)
                         for name in policies if name != "baseline" or ablate_baseline)
        jobs = [(label, (actors[name], args.output / f"{family}-{label}", args, device, family, iterations))
                for label, name, iterations in modes]
        for offset in range(0, len(jobs), parallel_pairs):
            for label, outcome in _measure_batch(jobs[offset:offset + parallel_pairs]):
                results[label] = outcome
                report["families"][family] = {tag: results[tag] for tag, *_ in modes if tag in results}
                report["execution_wall_seconds"] = perf_counter() - start
                tmp = args.output / "comparison.tmp"
                tmp.write_text(json.dumps(report, indent=2) + "\n")
                tmp.replace(args.output / "comparison.json")
                print(json.dumps({"family": family, "policy": label, **outcome}), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=[], nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES + HELD_TOPOLOGIES, default=FAMILIES, nargs="+")
    parser.add_argument("--aircraft", type=int, default=12)
    parser.add_argument("--control-dt", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--parallel-pairs", type=int, default=1,
                        help="Overlap this many independent native policy evaluations on one device")
    parser.add_argument("--device", choices=("cpu", "ppu"), default="ppu")
    parser.add_argument("--admm-k", type=int, default=2)
    parser.add_argument("--admm-fallback", type=int, default=0)
    parser.add_argument("--cold-start-admm", action="store_true",
                        help="Use fixed penalties, uniform mobility, and zero initial consensus duals")
    parser.add_argument("--admm-packet-loss", type=float,
                        help="Drop this fraction of bilateral ADMM exchanges while retaining physical checks")
    parser.add_argument("--admm-delay-steps", type=int, choices=(0, 1, 2),
                        help="Delay bilateral exchange payloads by this many control ticks")
    parser.add_argument("--admm-schedule", choices=("jacobi", "colored"), default="jacobi",
                        help="Use synchronous updates or sequential independent color groups in each ADMM sweep")
    parser.add_argument("--admm-communication-period", type=float,
                        help="Run ADMM only on these periodic ticks while holding the control cadence fixed")
    parser.add_argument("--admm-gap-policy", choices=("hold", "actor", "mpc"), default="hold",
                        help="Hold targets, execute the actor, or replan from current observations on gap ticks")
    parser.add_argument("--component-projection", action="store_true")
    parser.add_argument("--vertical-heading", action="store_true")
    parser.add_argument("--directional-heading", action="store_true")
    parser.add_argument("--dense-extra-rounds", type=int, default=0)
    parser.add_argument("--layer-fallback", action="store_true",
                        help="Use CPU discrete altitude search for projection diagnostics")
    parser.add_argument("--batched-layer-search", action="store_true",
                        help="Use batched discrete altitude search for projection diagnostics")
    parser.add_argument("--batched-layer-margin", type=float, default=0.0,
                        help="Prefer this vertical clearance in metres beyond the nominal 300 ft separation")
    parser.add_argument("--accelerated-layer-forecast", action="store_true",
                        help="Check the acceleration and capture phases in the batched layer forecast")
    parser.add_argument("--preserve-imminent-clearance", action="store_true",
                        help="Preserve vertical clearance for existing unresolved pairs within 30 seconds")
    parser.add_argument("--early-layer-search", action="store_true")
    parser.add_argument("--exact-layer-forecast", action="store_true")
    parser.add_argument("--equivariant-actor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--joint-layer-search", action="store_true",
                        help="Try a joint three-aircraft repair after pair search")
    parser.add_argument("--joint-layer-reserve", type=float, default=0.0,
                        help="Retain this existing vertical buffer in metres during three-aircraft repair")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--navdata-archive", type=Path)
    parser.add_argument("--navdata", type=Path, help="Reuse a navigation cache from the same native checkout and runtime")
    parser.add_argument("--ablate-baseline", action="store_true",
                        help="Also evaluate the incoming policy without projection")
    parser.add_argument("--control-frame-diagnostics", action="store_true")
    parser.add_argument("--graph-statistics", action="store_true",
                        help="Record exact clique and component sizes of each live observed conflict graph")
    parser.add_argument("--without-potential-edges", action="store_true",
                        help="Ablate potential edges in the actor graph and graph-gated coordination")
    parser.add_argument("--track-yaw-recovery", action="store_true",
                        help="Record each aircraft's first sustained alignment with its goal bearing")
    return parser


if __name__ == "__main__":
    compare(build_parser().parse_args())
