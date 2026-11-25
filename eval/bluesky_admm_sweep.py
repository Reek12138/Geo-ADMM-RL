"""Compare fixed ADMM round budgets and cold starts on identical native traffic."""

import argparse
from itertools import product
import json
from pathlib import Path
from tempfile import mkdtemp

from eval.bluesky_pair import FAMILIES, _checksum, build_parser, compare
from eval.bluesky_scalability import summarize_evaluation


def run(options):
    if min(options.seeds, options.aircraft, options.max_steps) <= 0 or len(set(options.rounds)) != len(options.rounds):
        raise ValueError("Use positive evaluation limits and distinct round budgets")
    paths = options.checkpoints
    seeds = options.training_seeds or [None] * len(paths)
    if options.training_seeds and (len(seeds) != len(paths) or len(set(seeds)) != len(paths)):
        raise ValueError("Each checkpoint needs a distinct training-seed identifier")
    hashes = [_checksum(path) for path in paths]
    if len(set(hashes)) != len(paths):
        raise ValueError("Provide distinct checkpoints for a multi-model evaluation")
    losses = options.packet_loss_rates or ((0.,) if "colored" in options.schedules else (None,))
    periods = options.communication_periods or (None,)
    delays = options.delay_steps or (None,)
    if (len(set(losses)) != len(losses) or len(set(options.schedules)) != len(options.schedules)
            or len(set(periods)) != len(periods) or len(set(delays)) != len(delays)):
        raise ValueError("Packet loss rates, schedules, periods, and delays must be distinct")
    layer_profile = dict(early_layer_search=getattr(options, "early_layer_search", False),
                         exact_layer_forecast=getattr(options, "exact_layer_forecast", False),
                         batched_layer_margin_m=getattr(options, "batched_layer_margin", 0.))
    options.output.mkdir(parents=True, exist_ok=options.resume)
    protocol = dict(
        rounds_per_continuous_tier=list(options.rounds),
        traffic_seeds=list(range(options.seed, options.seed + options.seeds)),
        training_seeds=seeds,
        checkpoints=[str(path) for path in paths],
        checkpoint_sha256=hashes,
        families=list(options.family), aircraft=options.aircraft, max_steps=options.max_steps, device=options.device,
        parallel_pairs=options.parallel_pairs, navdata_sha256=_checksum(options.navdata) if options.navdata else None,
        packet_loss_rates=list(losses),
        schedules=list(options.schedules),
        communication_periods=list(periods),
        delay_steps=list(delays),
        gap_policy=options.gap_policy,
    )
    if any(layer_profile.values()):
        protocol["layer_safety_profile"] = layer_profile
    summary_path = options.output / "summary.json"
    if options.resume and summary_path.exists():
        result = json.loads(summary_path.read_text())
        if {key: value for key, value in result.items() if key != "results"} != protocol:
            raise ValueError("The requested continuation differs from the archived communication matrix")
    else:
        result = protocol | {"results": {}}
    names = ["baseline", *(f"candidate_{i}" for i in range(1, len(paths)))]
    trials = []
    for rounds in options.rounds:
        if rounds == 0:
            trials.append((rounds, "raw", None, "jacobi", None, None))
        else:
            for schedule in options.schedules:
                for mode in ("warm", "cold"):
                    trials.extend((rounds, mode, loss, schedule, period, delay)
                                  for loss, period, delay in product(losses, periods, delays))
    for rounds, mode, loss, schedule, period, delay in trials:
        label = f"k{rounds}-{mode}"
        if schedule == "colored":
            label += "-colored"
        if loss is not None:
            label += f"-loss-{loss:g}"
        if period is not None:
            label += f"-period-{period}"
        if delay is not None:
            label += f"-delay-{delay}"
        saved = result["results"].get(label)
        directory = Path(saved["report"]).parent if saved else options.output / label
        if saved is None and directory.exists():
            directory = Path(mkdtemp(prefix=f"{label}.", dir=options.output)) / "native"
        arguments = [
            "--baseline", str(paths[0]), "--output", str(directory), "--device", options.device,
            "--family", *options.family, "--aircraft", str(options.aircraft),
            "--max-steps", str(options.max_steps), "--seed", str(options.seed), "--seeds", str(options.seeds),
            "--parallel-pairs", str(options.parallel_pairs), "--admm-k", str(rounds),
            "--admm-schedule", schedule, "--graph-statistics", "--navdata-archive", str(options.output / "navdata"),
        ]
        if len(paths) > 1:
            arguments.extend(("--candidates", *map(str, paths[1:])))
        if rounds:
            arguments.extend(("--directional-heading", "--batched-layer-search",
                              "--joint-layer-search", "--joint-layer-reserve", "10"))
            for key in ("early_layer_search", "exact_layer_forecast"):
                if layer_profile[key]:
                    arguments.append("--" + key.replace("_", "-"))
            if layer_profile["batched_layer_margin_m"]:
                arguments.extend(("--batched-layer-margin", str(layer_profile["batched_layer_margin_m"])))
        if mode == "cold":
            arguments.append("--cold-start-admm")
        if loss is not None:
            arguments.extend(("--admm-packet-loss", str(loss)))
        if period is not None:
            arguments.extend(("--admm-communication-period", str(period), "--admm-gap-policy", options.gap_policy))
        if delay is not None:
            arguments.extend(("--admm-delay-steps", str(delay)))
        if options.navdata:
            arguments.extend(("--navdata", str(options.navdata)))
        if saved is None:
            compare(build_parser().parse_args(arguments))
        report = directory / "comparison.json"
        report_sha = _checksum(report)
        if saved is not None and report_sha != saved["report_sha256"]:
            raise ValueError(f"An archived trial changed since it was completed: {report}")
        data = json.loads(report.read_text())
        if (data["sim_dt"] != .05 or data["event_sampling_seconds"] != .05 or data["control_dt"] != 6
                or data["first_seed"] != options.seed or data["num_seeds"] != options.seeds
                or set(data["families"]) != set(options.family)
                or [data["sources"][name]["sha256"] for name in names] != hashes
                or any(set(family) != set(names) for family in data["families"].values())
                or any(data.get(key, False) != (value if rounds else False)
                       for key, value in layer_profile.items())
                or data["dense_extra_rounds"] != 0 or data["cold_start_admm"] != (mode == "cold")
                or data["admm_packet_loss"] != loss or data["admm_schedule"] != schedule
                or data["admm_communication_period"] != period
                or data["admm_gap_policy"] != (options.gap_policy if period is not None else "hold")
                or data["admm_delay_steps"] != delay
                or any(row["admm_k"] != rounds or row["admm_fallback"] != 0
                       for family in data["families"].values() for row in family.values())):
            raise RuntimeError("Evaluation applied a different continuous ADMM or communication budget")
        summary = summarize_evaluation(data, names, seeds, options.aircraft)
        if rounds == 0:
            for family in summary.values():
                aggregate = family["across_checkpoints"]
                aggregate["without_projection"] = aggregate.pop("projected")
        result["results"][label] = dict(report=str(report), report_sha256=report_sha,
                                        admm_k=rounds, cold_start=(mode == "cold"),
                                        deterministic_layer_repair=rounds > 0, packet_loss=loss, schedule=schedule,
                                        communication_period=period, delay_steps=delay, families=summary)
        pending = options.output / "summary.tmp"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(summary_path)


def build_sweep_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=(0, 1, 2, 3, 5, 10), nargs="+", default=(0, 1, 2, 3, 5))
    parser.add_argument("--packet-loss-rates", type=float, choices=(0., .2, .5), nargs="+")
    parser.add_argument("--schedules", choices=("jacobi", "colored"), nargs="+", default=("jacobi",))
    parser.add_argument("--communication-periods", choices=(6, 12, 18, 30), type=int, nargs="+")
    parser.add_argument("--delay-steps", choices=(0, 1, 2), type=int, nargs="+")
    parser.add_argument("--gap-policy", choices=("hold", "actor", "mpc"), default="hold")
    parser.add_argument("--family", choices=FAMILIES, nargs="+", default=("crossing3", "crossing4", "crossing6"))
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--aircraft", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--parallel-pairs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--early-layer-search", action="store_true")
    parser.add_argument("--exact-layer-forecast", action="store_true")
    parser.add_argument("--batched-layer-margin", type=float, default=0.)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_sweep_parser().parse_args())
