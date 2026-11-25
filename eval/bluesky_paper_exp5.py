"""Run the confirmed Stage 2 policy through the native Exp 5 communication panels."""

import argparse
import json
from pathlib import Path

from eval.bluesky_admm_sweep import build_sweep_parser, run as sweep
from eval.bluesky_pair import FAMILIES, _checksum
from eval.stage_acceptance import paper_checkpoint_digest, paper_layer_flags


CADENCE = ("--rounds", "2", "--communication-periods", "6", "12", "18", "30")

PANELS = {
    "round-budget": ("--rounds", "0", "1", "2", "3", "5"),
    "cadence-hold": (*CADENCE, "--gap-policy", "hold"),
    "cadence-mpc": (*CADENCE, "--gap-policy", "mpc"),
    "loss-delay-coloring": ("--rounds", "2", "--packet-loss-rates", "0", ".2", ".5",
                           "--delay-steps", "0", "1", "2", "--schedules", "jacobi", "colored"),
}


def run(args):
    checkpoint_sha = paper_checkpoint_digest(args.acceptance, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["--checkpoints", str(args.checkpoint), "--device", args.device,
              "--family", *FAMILIES, "--seed", "12001", "--seeds", "5", "--resume"]
    common.extend(paper_layer_flags(args.acceptance))
    if args.navdata:
        common += ["--navdata", str(args.navdata)]
    outputs = {}
    for label, arguments in PANELS.items():
        destination = args.output / label
        sweep(build_sweep_parser().parse_args([*common, "--output", str(destination), *arguments]))
        summary = destination / "summary.json"
        rows = json.loads(summary.read_text())["results"]
        outputs[label] = dict(output=str(summary), sha256=_checksum(summary), arms=len(rows))
    if sum(panel["arms"] for panel in outputs.values()) != 61:
        raise RuntimeError("The Exp 5 matrix must contain 61 configurations")
    report = dict(checkpoint_sha256=checkpoint_sha, acceptance_sha256=_checksum(args.acceptance),
                  families=FAMILIES, seeds=list(range(12001, 12006)), panels=outputs)
    target = args.output / "summary.json"
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pending.replace(target)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "checkpoint", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--navdata", type=Path)
    cli.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    run(cli.parse_args())
