"""Recheck shortlisted Stage 1 weights alongside the final one on fresh development traffic."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from eval.bluesky_pair import _checksum
from eval.bluesky_stage_gate import run


def _score(report):
    families = report["families"]
    for row in families.values():
        if sum(row[name] for name in ("arrivals", "collisions", "exits", "timeouts")) != row["aircraft"]:
            raise ValueError("Resume selection requires complete outcome counts")
    return (
        report["arrival_rate"], -report["collision_rate"],
        families["head_on"]["arrivals"], families["yaw_plus15"]["arrivals"], families["yaw_minus15"]["arrivals"],
    )


def main(args):
    panel = json.loads(args.panel.read_text())
    if args.seed == panel["seed"]:
        raise ValueError("Resume confirmation requires an independent development seed")
    frontier = [panel["checkpoints"][name] for name in panel["pareto_frontier"]]
    frontier.sort(key=lambda row: (-row["arrival_rate"], row["collision_rate"], -row["head_on_arrivals"]))
    if frontier:
        safest = min(frontier, key=lambda row: (row["collision_rate"], -row["arrival_rate"], -row["head_on_arrivals"]))
        frontier.insert(1, safest)
    original = args.incumbent.resolve()
    candidates = [original]
    for row in frontier:
        path = Path(row["checkpoint"]).resolve()
        if path in candidates:
            continue
        if path.parent != original.parent or _checksum(path) != row["sha256"]:
            raise ValueError("Resume nominee must match its original Stage 1 checkpoint digest")
        candidates.append(path)
        if len(candidates) == 3:
            break

    args.output.mkdir()
    reports = {}
    for path in candidates:
        output = args.output / f"{path.stem}.json"
        run(SimpleNamespace(
            checkpoint=path, calibration=args.calibration, output=output, envs=args.envs,
            aircraft=12, seed=args.seed, sim_dt=.25, max_steps=args.max_steps, device=args.device,
            without_projection=False, detection_km=None, protection_detection_km=None,
        ))
        reports[path] = json.loads(output.read_text())

    incumbent = _score(reports[original])
    retained = [path for path, report in reports.items()
                if all(value >= initial for value, initial in zip(_score(report), incumbent, strict=True))]
    selected = max(retained, key=lambda path: _score(reports[path]))
    output = args.output / "selection.json"
    output.write_text(json.dumps({
        "incumbent": str(original),
        "selected": str(selected),
        "selected_sha256": reports[selected]["sha256"],
        "seed": args.seed,
        "score_fields": ["arrival_rate", "negative_collision_rate", "head_on_arrivals",
                         "positive_yaw_arrivals", "negative_yaw_arrivals"],
        "scores": {str(path): _score(report) for path, report in reports.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--panel", type=Path, required=True)
    cli.add_argument("--incumbent", type=Path, required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seed", type=int, default=45613)
    cli.add_argument("--envs", type=int, default=16)
    cli.add_argument("--max-steps", type=int, default=400)
    cli.add_argument("--device", choices=("cpu", "ppu", "cuda"), default="ppu")
    main(cli.parse_args())
