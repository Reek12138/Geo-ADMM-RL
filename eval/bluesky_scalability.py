"""Evaluate frozen policy checkpoints at native BlueSky traffic scales."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from envs.bluesky_traffic import HELD_TOPOLOGIES
from eval.bluesky_pair import _checksum, build_parser, compare
from eval.stage_acceptance import mpc_communication_flags


def _aircraft_metrics(rows):
    count = sum(row["aircraft"] for row in rows)
    graphs = [row["conflict_graph"] for row in rows]
    activity = sum(graph["aircraft_decisions"] for graph in graphs)
    measures = {f"{name}_rate": sum(row[field] for row in rows) / count
                for name, field in (("arrival", "arrivals"), ("collision", "collisions"),
                                    ("exit", "exits"), ("unresolved", "unresolved"))}
    measures["mean_action_cost_per_aircraft"] = sum(
        row["action_cost_per_aircraft"] * row["aircraft"] for row in rows) / count
    measures["mean_degree"] = sum(graph["degree_sum"] for graph in graphs) / activity
    for name in ("peak_aircraft", "max_degree", "max_clique_size", "max_component_size"):
        measures[name] = max(graph[name] for graph in graphs)
    return measures


def _communication_metrics(rows):
    exchanges = [row["communication"] for row in rows]
    counts = ("attempted_pair_exchanges", "dropped_pair_exchanges", "delivered_pair_exchanges",
              "coordination_ticks", "projection_solves")
    measures = {key: sum(row[key] for row in exchanges) for key in counts}
    attempted = measures["attempted_pair_exchanges"]
    measures["observed_loss"] = measures["dropped_pair_exchanges"] / attempted if attempted else None
    solves = measures["projection_solves"]
    for field in ("speed_primal_residual_mean_mps", "altitude_primal_residual_mean_m"):
        total = sum(row[field] * row["projection_solves"] for row in exchanges if row["projection_solves"])
        measures[field] = total / solves if solves else None
    if all("used_delayed_pair_states" in row for row in exchanges):
        for key in ("in_flight_pair_exchanges", "used_delayed_pair_states"):
            measures[key] = sum(row[key] for row in exchanges)
        used = measures["used_delayed_pair_states"]
        age = sum(row["mean_used_message_age_steps"] * row["used_delayed_pair_states"]
                  for row in exchanges if row["used_delayed_pair_states"])
        measures["mean_used_message_age_steps"] = age / used if used else None
    if all("gap_mpc" in row for row in rows):
        for key in rows[0]["gap_mpc"]:
            measures["gap_mpc_" + key] = sum(row["gap_mpc"][key] for row in rows)
    return measures


def summarize_evaluation(report, names, training_seeds, size):
    if report["sim_dt"] != .05 or report["event_sampling_seconds"] != .05 or not report["graph_statistics"]:
        raise ValueError("Scale evaluation requires native frame sampling and conflict graph statistics")
    summaries = {}
    for family, policies in report["families"].items():
        per_policy = {}
        for label, values in policies.items():
            rows = values["rows"]
            if any(row["aircraft"] != size or row["conflict_graph"]["peak_aircraft"] != size for row in rows):
                raise ValueError(f"Incomplete concurrent fleet in {family}, {label}, size {size}")
            if [row["seed"] for row in rows] != list(range(report["first_seed"],
                                                          report["first_seed"] + report["num_seeds"])):
                raise ValueError("Scale evaluation used inconsistent traffic seeds")
            if family in HELD_TOPOLOGIES and any(row.get("route_count") != int(family.removeprefix("crossing")) for row in rows):
                raise ValueError(f"Route count differs from the held topology {family}")
            policy = label.removesuffix("_without_projection")
            index = names.index(policy)
            metrics = _aircraft_metrics(rows)
            if all("communication" in row for row in rows):
                metrics.update(_communication_metrics(rows))
            per_policy[label] = dict(training_seed=training_seeds[index],
                                     checkpoint_sha256=report["sources"][policy]["sha256"],
                                     mean_decision_ms_per_batch=values["mean_decision_ms_per_batch"],
                                     p95_decision_ms_per_batch=values["p95_decision_ms_per_batch"],
                                     per_traffic_seed=rows, metrics=metrics)
            for field in ("mean_control_cycle_ms", "p95_control_cycle_ms"):
                if field in values:
                    per_policy[label][field] = values[field]
            per_seed = [_aircraft_metrics([row]) for row in rows]
            traffic_summary = {}
            for field in per_seed[0]:
                samples = [row[field] for row in per_seed]
                traffic_summary[field] = dict(mean=mean(samples), sample_std=stdev(samples) if len(samples) > 1 else None)
            per_policy[label]["across_traffic_seeds"] = traffic_summary
        aggregate = {}
        for mode, suffix in (("projected", ""), ("without_projection", "_without_projection")):
            scores = [per_policy[name + suffix]["metrics"] for name in names if name + suffix in per_policy]
            if not scores:
                continue
            aggregate[mode] = {}
            for metric in scores[0]:
                values = [row[metric] for row in scores if row[metric] is not None]
                aggregate[mode][metric] = dict(mean=mean(values) if values else None,
                                                sample_std=stdev(values) if len(values) > 1 else None)
        summaries[family] = dict(per_checkpoint=per_policy, across_checkpoints=aggregate)
    return summaries


def run(options):
    paths = options.checkpoints
    if options.training_seeds and (len(options.training_seeds) != len(paths)
                                   or len(set(options.training_seeds)) != len(paths)):
        raise ValueError("Each checkpoint needs a distinct training seed when identifiers are provided")
    if len({_checksum(path) for path in paths}) != len(paths):
        raise ValueError("Provide distinct checkpoints for a multi-model evaluation")
    if options.seeds < 1 or options.max_steps < 1 or len(set(options.sizes)) != len(options.sizes):
        raise ValueError("Use positive evaluation limits and distinct fleet sizes")
    seeds = options.training_seeds or [None] * len(paths)
    options.output.mkdir(parents=True)
    result = dict(checkpoints=[str(path) for path in paths], training_seeds=seeds,
                  traffic_seeds=list(range(options.seed, options.seed + options.seeds)), fleet_sizes={})
    names = ["baseline", *(f"candidate_{i}" for i in range(1, len(paths)))]
    for size in options.sizes:
        folder = options.output / f"aircraft-{size}"
        arguments = [
            "--baseline", str(paths[0]), "--output", str(folder), "--device", options.device,
            "--family", *options.family, "--aircraft", str(size), "--max-steps", str(options.max_steps),
            "--seed", str(options.seed), "--seeds", str(options.seeds), "--parallel-pairs", str(options.parallel_pairs),
            "--admm-k", "2", "--dense-extra-rounds", "2", "--directional-heading", "--batched-layer-search",
            "--joint-layer-search", "--joint-layer-reserve", "10", "--graph-statistics",
            "--navdata-archive", str(options.output / "navdata"),
        ]
        if len(paths) > 1:
            arguments.extend(("--candidates", *map(str, paths[1:])))
        if options.ablation:
            arguments.extend(("--ablation", "--ablate-baseline"))
        if getattr(options, "mpc_gap", False):
            arguments.extend(mpc_communication_flags())
        if options.navdata:
            arguments.extend(("--navdata", str(options.navdata)))
        for key in ("early_layer_search", "exact_layer_forecast"):
            if getattr(options, key, False):
                arguments.append("--" + key.replace("_", "-"))
        if getattr(options, "batched_layer_margin", 0.):
            arguments.extend(("--batched-layer-margin", str(options.batched_layer_margin)))
        compare(build_parser().parse_args(arguments))
        path = folder / "comparison.json"
        report = json.loads(path.read_text())
        result["fleet_sizes"][size] = dict(report=str(path), families=summarize_evaluation(report, names, seeds, size))
        pending = options.output / "summary.tmp"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(options.output / "summary.json")


def build_scale_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", choices=(30, 50, 80), type=int, nargs="+", default=(30, 50, 80))
    parser.add_argument("--family", choices=("crossing3", "crossing4", "crossing6") + HELD_TOPOLOGIES,
                        nargs="+", default=("crossing6",))
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--parallel-pairs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--mpc-gap", action="store_true")
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--early-layer-search", action="store_true")
    parser.add_argument("--exact-layer-forecast", action="store_true")
    parser.add_argument("--batched-layer-margin", type=float, default=0.)
    return parser


if __name__ == "__main__":
    run(build_scale_parser().parse_args())
