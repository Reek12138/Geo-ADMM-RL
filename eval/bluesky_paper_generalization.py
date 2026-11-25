"""Run Exp 2 topology transfer and Exp 3 fleet scaling on the admitted policy."""

import argparse
import json
from pathlib import Path

from envs.bluesky_traffic import HELD_TOPOLOGIES
from eval.bluesky_pair import _checksum
from eval.bluesky_scalability import build_scale_parser, run as evaluate
from eval.stage_acceptance import paper_checkpoint_digest, paper_layer_flags, paper_uses_mpc


def run(args):
    digest = paper_checkpoint_digest(args.acceptance, args.checkpoint)
    args.output.mkdir(parents=True)
    common = ["--checkpoints", str(args.checkpoint), "--device", args.device,
              "--seed", "12001", "--seeds", "5", "--ablation"]
    common.extend(paper_layer_flags(args.acceptance))
    if paper_uses_mpc(args.acceptance):
        common.append("--mpc-gap")
    if args.navdata:
        common += ["--navdata", str(args.navdata)]
    panels = {
        "exp2": ["--sizes", "30", "--family", "crossing6", *HELD_TOPOLOGIES],
        "exp3": ["--sizes", "30", "50", "80", "--family", "crossing6"],
    }
    reports = {}
    for name, options in panels.items():
        output = args.output / name
        evaluate(build_scale_parser().parse_args([*common, "--output", str(output), *options]))
        path = output / "summary.json"
        reports[name] = dict(path=str(path), sha256=_checksum(path))
    result = dict(checkpoint_sha256=digest, acceptance_sha256=_checksum(args.acceptance),
                  seeds=list(range(12001, 12006)), panels=reports)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "checkpoint", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--navdata", type=Path)
    cli.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    run(cli.parse_args())
