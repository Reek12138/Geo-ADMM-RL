"""Evaluate the admitted policy under native head-on, yaw, and traffic disturbances."""

import argparse
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

from eval.bluesky_pair import _checksum, build_parser, compare
from eval.stage_acceptance import (
    MPC_COMMUNICATION, mpc_communication_flags, paper_checkpoint_digest, paper_layer_flags, paper_uses_mpc,
)


SEEDS = tuple(range(12001, 12006))
DISTURBANCES = {
    "clear": (),
    "popup": ("popup",),
    "noncooperative": ("noncooperative",),
    "wind": ("wind-field",),
    "combined": ("popup", "noncooperative", "wind-field"),
}


def _five_seed_statistics(rows, *, outcome, cost):
    return {
        key: dict(mean=mean(values), sample_std=stdev(values))
        for key, values in {
            "arrival_rate": [outcome(r)["arrivals"] / r["aircraft"] for r in rows],
            "collision_rate": [outcome(r)["collisions"] / r["aircraft"] for r in rows],
            "action_cost_per_aircraft": [r[cost] for r in rows],
        }.items()
    }


def _compare(args, directory, aircraft, families):
    flags = [
        "--baseline", str(args.checkpoint), "--output", str(directory), "--device", args.device,
        "--seed", str(SEEDS[0]), "--seeds", str(len(SEEDS)), "--family", *families, "--aircraft", str(aircraft),
        "--admm-k", "2", "--dense-extra-rounds", "2", "--directional-heading", "--batched-layer-search",
        "--joint-layer-search", "--joint-layer-reserve", "10", "--ablation", "--ablate-baseline",
        "--navdata-archive", str(args.output / "navdata"),
    ]
    flags.extend(paper_layer_flags(args.acceptance))
    if paper_uses_mpc(args.acceptance):
        flags.extend(mpc_communication_flags())
    if args.navdata:
        flags.extend(("--navdata", str(args.navdata)))
    if families == ("yaw_plus15", "yaw_minus15"):
        flags.append("--track-yaw-recovery")
    compare(build_parser().parse_args(flags))
    report = directory / "comparison.json"
    data = json.loads(report.read_text())
    results = {}
    for family, arms in data["families"].items():
        results[family] = {}
        for arm, values in arms.items():
            rows = values["rows"]
            if tuple(row["seed"] for row in rows) != SEEDS:
                raise ValueError("The condition used different flow seeds")
            summary = _five_seed_statistics(rows, outcome=lambda row: row, cost="action_cost_per_aircraft")
            if "yaw_recovery" in rows[0]:
                rates = [r["yaw_recovery"]["recovered_aircraft"] / r["aircraft"] for r in rows]
                summary["sustained_alignment_rate"] = dict(mean=mean(rates), sample_std=stdev(rates))
            results[family][arm] = summary
    return dict(path=str(report), sha256=_checksum(report), statistics=results)


def run(args):
    digest = paper_checkpoint_digest(args.acceptance, args.checkpoint)
    mpc = paper_uses_mpc(args.acceptance)
    args.output.mkdir(parents=True)
    head_on = _compare(args, args.output / "head-on-pressure", 30, ("head_on",))
    yaw = _compare(args, args.output / "yaw", 12, ("yaw_plus15", "yaw_minus15"))
    disturbance_root = args.output / "disturbances"
    disturbance_root.mkdir()
    rows = {name: [] for name in DISTURBANCES}
    layouts = {}
    for seed, name in product(SEEDS, DISTURBANCES):
        switches = DISTURBANCES[name]
        run_directory = disturbance_root / f"{seed}-{name}"
        run_directory.mkdir()
        report = run_directory / "result.json"
        command = [
            sys.executable, "-B", "-m", "eval.bluesky_disturbance",
            "--checkpoint", str(args.checkpoint), "--workdir", str(run_directory / "world"), "--output", str(report),
            "--seed", str(seed), "--family", "crossing6", "--aircraft", "20",
        ]
        command.extend(paper_layer_flags(args.acceptance))
        if mpc:
            command.append("--mpc-gap")
        if args.navdata:
            command.extend(("--navdata", str(args.navdata)))
        if switches:
            command.extend(("--disturbances", *switches))
        with (run_directory / "run.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(report.read_text())
        layout = layouts.setdefault(seed, data["base_routes_sha256"])
        if (data["base_routes_sha256"] != layout or data["seed"] != seed or data["sim_dt"] != .05
                or data["checkpoint_sha256"] != digest or data["disturbances"] != list(switches)
                or sum(data["outcomes"].values()) != data["aircraft"] or not data["projection"]):
            raise ValueError(f"Disturbance did not preserve the paired layout or outcome accounting: {report}")
        if mpc and (not MPC_COMMUNICATION.items() <= data.get("communication", {}).items()
                    or data["communication"]["gap_ticks"] <= 0):
            raise ValueError(f"Missing MPC gap evidence: {report}")
        rows[name].append(data | {"report": str(report), "sha256": _checksum(report)})
    summary = dict(
        checkpoint_sha256=digest, acceptance_sha256=_checksum(args.acceptance), traffic_seeds=SEEDS,
        head_on_pressure=head_on, yaw=yaw, disturbance_rows=rows,
        disturbance_statistics={
            name: _five_seed_statistics(run, outcome=lambda row: row["outcomes"], cost="normalized_cost_per_aircraft")
            for name, run in rows.items()
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "checkpoint", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--navdata", type=Path)
    cli.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    run(cli.parse_args())
