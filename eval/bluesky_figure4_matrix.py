"""Run the three formal Figure 4 policy arms and summarize five paired traffic seeds."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from tempfile import mkdtemp

from eval.bluesky_pair import _checksum
from eval.stage_acceptance import (
    MPC_COMMUNICATION, paper_checkpoint_digest, paper_layer_flags, paper_layer_options, paper_uses_mpc,
)


ARMS = ("hold-kss", "sarl-kss", "geo")
SEEDS = tuple(range(12001, 12006))
CASES = (1, 2, 3)


def _write(path, report):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _mean_sd(values):
    return dict(mean=mean(values), std=stdev(values))


def summarize(rows, arms=ARMS):
    summary = {}
    for case in CASES:
        summary[case] = {}
        for arm in arms:
            samples = [rows[case, seed, arm] for seed in SEEDS]
            metrics = {
                "arrival_rate": [x["outcomes"]["arrivals"] / x["aircraft"] for x in samples],
                "collision_rate": [x["outcomes"]["collisions"] / x["aircraft"] for x in samples],
                "cost_per_aircraft": [x["action_cost_per_aircraft"] for x in samples],
            }
            summary[case][arm] = {key: _mean_sd(values) for key, values in metrics.items()}
    return summary


def run(args):
    geo_sha = paper_checkpoint_digest(args.acceptance, args.checkpoint)
    layer_flags = paper_layer_flags(args.acceptance)
    layer_options = paper_layer_options(args.acceptance)
    mpc = paper_uses_mpc(args.acceptance)
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = args.history / "source/scripts"
    source_hashes = {
        "plugins.case_step3_SARL": _checksum(args.sarl_checkout / "plugins/case_step3_SARL.py"),
        "paper_alignment": _checksum(scripts / "paper_alignment.py"),
        "paper_ctde": _checksum(scripts / "paper_ctde.py"),
    }
    rows = {}
    arms = ("geo", "geo-kss") if args.geo_kss_ablation else ARMS
    for case in CASES:
        for seed in SEEDS:
            manifest = args.manifests / f"case{case}-{seed}.json"
            data = json.loads(manifest.read_text())
            if (data["case"], data["seed"], data["reference_arm"], len(data["flights"])) != (case, seed, "hold_kss", 50):
                raise ValueError(f"Invalid formal release manifest {manifest}")
            expected = [dict(callsign=flight["callsign"], slot=slot, frame=flight["release_frame"])
                        for slot, flight in enumerate(data["flights"])]
            manifest_sha = _checksum(manifest)
            comparison = {}
            for arm in arms:
                case_dir = args.history / "evaluation" / f"case{case}"
                historical = "hold_kss" if arm == "hold-kss" else "actor_kss"
                settings = case_dir / historical / str(seed) / "config_resolved.json"
                actor_path = (args.checkpoint if arm.startswith("geo") else
                              Path(json.loads(settings.read_text())["actor_checkpoint"]) if arm == "sarl-kss" else None)
                expected_sha = geo_sha if arm.startswith("geo") else _checksum(actor_path) if actor_path else None
                label = f"case{case}-{seed}-{arm}"
                result = args.output / f"{label}.json"
                if not result.exists():
                    attempt = Path(mkdtemp(prefix=f"{label}.", dir=args.output))
                    argv = [sys.executable, "-B", "-m", "eval.bluesky_figure4_policies",
                            "--policy", arm, "--manifest", str(manifest), "--workdir", str(attempt / "world"),
                            "--output", str(result), "--navdata", str(args.navdata)]
                    if arm.startswith("geo"):
                        argv += ["--checkpoint", str(args.checkpoint), "--device", args.device]
                    if arm == "geo":
                        argv.extend(layer_flags)
                        if mpc:
                            argv.append("--mpc-gap")
                    if arm != "geo":
                        argv += ["--sarl-checkout", str(args.sarl_checkout),
                                 "--sarl-scripts", str(scripts),
                                 "--sarl-config", str(settings)]
                    with (attempt / "run.log").open("w") as log:
                        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads(result.read_text())
                communication = row.get("communication", {})
                if mpc and arm == "geo":
                    if not MPC_COMMUNICATION.items() <= communication.items() or communication["gap_ticks"] <= 0:
                        raise ValueError(f"Missing MPC gap control evidence in {result}")
                elif communication:
                    raise ValueError(f"Unexpected Figure 4 communication configuration in {result}")
                if (row["manifest_sha256"] != manifest_sha or row["checkpoint_sha256"] != expected_sha
                        or row["historical_config_sha256"] != (_checksum(settings) if arm != "geo" else None)
                        or row["historical_source_sha256"] != (source_hashes if arm != "geo" else None)
                        or any(row.get("layer_safety_options", {}).get(key, False) != value
                               for key, value in layer_options.items() if arm == "geo")
                        or row["policy"] != arm or row["departures"] != expected or row["aircraft"] != 50
                        or row["sim_dt"] != .05 or row["control_dt"] != 6
                        or sum(row["outcomes"].values()) != 50):
                    raise ValueError(f"Unpaired or incompatible output {result}")
                rows[case, seed, arm] = row
                comparison[arm] = [row[k] for k in ("collision_h_km", "collision_v_ft", "arrival_dist_km", "max_xtk_km")]
            if any(values != comparison["geo"] for values in comparison.values()):
                raise ValueError(f"Physical outcome thresholds differ in Case {case}, seed {seed}")
    _write(args.output / "summary.json", dict(
        stage2_acceptance_sha256=_checksum(args.acceptance), geo_checkpoint_sha256=geo_sha,
        sim_dt=.05, control_dt=6., seeds=SEEDS, action_cost_weights=[1, 2, 4], cases=summarize(rows, arms),
        rows={f"case{c}-{s}-{a}": row for (c, s, a), row in rows.items()},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "checkpoint", "manifests", "history", "sarl-checkout", "navdata", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    cli.add_argument("--geo-kss-ablation", action="store_true")
    run(cli.parse_args())
