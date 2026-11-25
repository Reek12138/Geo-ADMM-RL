"""Evaluate native safety, graph, and return-horizon ablations for Exp 6."""

import argparse
import json
from pathlib import Path

import torch

from envs.bluesky_graph import ObservationProfile
from eval.bluesky_figure4_matrix import run as figure4
from eval.bluesky_pair import FAMILIES, _checksum, build_parser, compare
from eval.stage_acceptance import (
    mpc_communication_flags, paper_checkpoint_digest, paper_layer_flags, paper_uses_mpc,
)


def _verify_one_step_control(args):
    if args.one_step_checkpoint is None:
        return None
    full = json.loads(args.control_config.read_text())
    ablated = json.loads(args.one_step_config.read_text())
    for config in (full, ablated):
        config.setdefault("initialize_critic", False)
        config.setdefault("freeze_actor_encoder", False)
    if (full["return_window"] != 24 or ablated["return_window"] != 24
            or not full["multi_step_return"] or ablated["multi_step_return"]):
        raise ValueError("Only critic return length should change, with both arms retaining 24-decision replay maturity")
    digest = _checksum(args.one_step_checkpoint)
    if digest == _checksum(args.checkpoint):
        raise ValueError("One-step evaluation needs an independently trained control checkpoint")
    primary = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    control = torch.load(args.one_step_checkpoint, map_location="cpu", weights_only=True)
    if not primary.get("updates") or primary["updates"] != control.get("updates"):
        raise ValueError("The one-step control must use the same learner update count")
    if not primary.get("decisions") or primary["decisions"] != control.get("decisions"):
        raise ValueError("The one-step control must use the same simulator decision count")
    metadata = dict(training_plan=full["plan"], equivariant_actor=False, admm_warmstart_reflection=None)
    for key, default in metadata.items():
        if primary.get(key, default) != control.get(key, default):
            raise ValueError(f"The one-step control must preserve {key}")
    ignored = {"output", "multi_step_return"}
    stops = {"decisions", "target_episodes", "target_updates"}
    if (ablated["decisions"] == primary["decisions"] <= full["decisions"]
            and ablated["target_episodes"] == ablated["target_updates"] == 0):
        ignored.update(stops)
    if {k: v for k, v in full.items() if k not in ignored} != {k: v for k, v in ablated.items() if k not in ignored}:
        raise ValueError("The archived training configurations differ beyond the critic return and matched stop")
    profile = ObservationProfile(**primary.get("observation_profile", {}))
    if profile != ObservationProfile(**control.get("observation_profile", {})):
        raise ValueError("The one-step control must use the same inference observation profile")
    return dict(sha256=digest, control_config_sha256=_checksum(args.control_config),
                one_step_config_sha256=_checksum(args.one_step_config), updates=primary["updates"],
                decisions=primary["decisions"],
                detection_metres=profile.detection_metres)


def _native_panel(args, folder, *, drop_edges=False):
    flags = [
        "--baseline", str(args.checkpoint), "--output", str(folder), "--device", args.device,
        "--seed", "12001", "--seeds", "5", "--family", *FAMILIES,
        "--admm-k", "2", "--dense-extra-rounds", "2", "--directional-heading", "--batched-layer-search",
        "--joint-layer-search", "--joint-layer-reserve", "10", "--graph-statistics",
        "--navdata-archive", str(args.output / "navdata"), "--navdata", str(args.navdata),
    ]
    flags.extend(paper_layer_flags(args.acceptance))
    if paper_uses_mpc(args.acceptance):
        flags.extend(mpc_communication_flags())
    if drop_edges:
        flags.append("--without-potential-edges")
    else:
        flags.extend(("--ablation", "--ablate-baseline"))
        if args.one_step_checkpoint is not None:
            flags.extend(("--candidates", str(args.one_step_checkpoint)))
    compare(build_parser().parse_args(flags))
    path = folder / "comparison.json"
    report = json.loads(path.read_text())
    if report["sim_dt"] != .05 or report["event_sampling_seconds"] != .05 or report["without_potential_edges"] != drop_edges:
        raise ValueError(f"Ablation ran under a different native protocol: {path}")
    stats = {}
    for family, outcomes in report["families"].items():
        stats[family] = {}
        for arm, data in outcomes.items():
            rows = data["rows"]
            stats[family][arm] = {key: sum(row[key] for row in rows)
                                  for key in ("aircraft", "arrivals", "collisions", "exits", "unresolved")}
    return dict(path=str(path), sha256=_checksum(path), outcomes=stats)


def run(args):
    digest = paper_checkpoint_digest(args.acceptance, args.checkpoint)
    one_step = _verify_one_step_control(args)
    args.output.mkdir(parents=True)
    projected_and_raw = _native_panel(args, args.output / "projection")
    no_potential_edges = _native_panel(args, args.output / "existing-edges-only", drop_edges=True)
    kss = args.output / "kss-substitution"
    figure4(argparse.Namespace(
        acceptance=args.acceptance, checkpoint=args.checkpoint, manifests=args.manifests, history=args.history,
        sarl_checkout=args.sarl_checkout, navdata=args.navdata, output=kss, geo_kss_ablation=True,
        device="cuda:0" if args.device == "ppu" else "cpu",
    ))
    summary = kss / "summary.json"
    result = dict(
        checkpoint_sha256=digest, acceptance_sha256=_checksum(args.acceptance), one_step_control=one_step,
        projected_and_raw=projected_and_raw, no_potential_edges=no_potential_edges,
        kss_substitution=dict(path=str(summary), sha256=_checksum(summary)),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "checkpoint", "manifests", "history", "sarl-checkout", "navdata", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--device", choices=("cpu", "ppu"), default="ppu")
    for name in ("one-step-checkpoint", "control-config", "one-step-config"):
        cli.add_argument(f"--{name}", type=Path)
    options = cli.parse_args()
    one_step_arguments = (options.one_step_checkpoint, options.control_config, options.one_step_config)
    if any(one_step_arguments) and not all(one_step_arguments):
        cli.error("A one-step control requires its checkpoint and both matched training configurations")
    run(options)
