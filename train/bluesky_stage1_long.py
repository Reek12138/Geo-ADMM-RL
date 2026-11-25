"""Run guarded 144-second policy training and select on paired native traffic."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from envs.bluesky_traffic import MIXTURE_NAMES
from train.bluesky_stage2 import native_development_score, passes_native_development
from train.bluesky_td3 import build_parser, train


UPDATES = 20_000
CHECKPOINT_EVERY = 5_000
DEVELOPMENT_PANELS = ((70111, 3), (75121, 5))


def _run_command(arguments, log, environment=None):
    with log.open("w", encoding="utf-8") as output:
        subprocess.run([sys.executable, "-B", *map(str, arguments)], check=True,
                       env=environment, stdout=output, stderr=subprocess.STDOUT)


def _write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def run(options):
    saved = torch.load(options.incoming, map_location="cpu", weights_only=True)
    exact = options.exact_equivariant
    first_update, first_decision = (0, 0) if exact else (int(saved["updates"]), int(saved["decisions"]))
    width = int(saved["width"])
    del saved
    native_environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        map(str, (options.native_checkout, Path(__file__).resolve().parents[1]))))
    native_cache = options.output / "native-navcache"
    layer_flags = ("--early-layer-search", "--exact-layer-forecast", "--batched-layer-margin", "6") if exact else ()
    navdata_flags = ("--navdata", options.navdata) if options.navdata else ()
    development_panels = ((61231, 5), (75121, 5)) if exact else DEVELOPMENT_PANELS
    selections = []
    admitted = []
    feedback_pairs = None
    if options.feedback_audit is not None:
        feedback = json.loads(options.feedback_audit.read_text())
        if feedback["sim_dt"] != .05 or feedback["control_dt"] != 6:
            raise ValueError("The feedback diagnostic requires native 0.05 s physics and six-second decisions")
        feedback_pairs = {}
        for scenario, rows in feedback["cases"].items():
            original, changed = rows["baseline"]["outcome"], rows["first-only"]["outcome"]
            feedback_pairs[scenario + "-native"] = (original, changed)

    def audit(name, checkpoint):
        output = options.output / f"{name}-q.json"
        _run_command((options.q_audit, "--source", options.q_source, "--output", output,
                      "--crossing-four", options.q_crossing_four, "--crossing-six", options.q_crossing_six,
                      "--checkpoint", f"{name}={checkpoint}"), options.output / f"{name}-q.log")
        by_case = json.loads(output.read_text())[name]["cases"]
        cases = by_case.values()
        scores = {
            "native": sum(item["improvement_on_changed_aircraft"] > 0 for item in cases),
            "matching_plan_b": sum(item["used_training_observation_on_changed_aircraft"] > 0 for item in cases),
            "native_all_active": sum(item["improvement_all_active"] > 0 for item in cases),
            "matching_plan_b_all_active": sum(item["used_training_observation_all_active"] > 0 for item in cases),
        }
        if feedback_pairs is not None:
            if set(by_case) != set(feedback_pairs):
                raise ValueError("Feedback and first-action audit cases differ")
            scores["single_action_feedback"] = {
                scenario: dict(original=original, single_action=changed,
                               native_q_delta=by_case[scenario]["improvement_all_active"],
                               matching_plan_b_q_delta=by_case[scenario]["used_training_observation_all_active"])
                for scenario, (original, changed) in feedback_pairs.items()
            }
        return scores

    def evaluate(name, checkpoint, seed, seeds):
        directory = options.output / f"native-{name}-{seed}"
        _run_command(("-m", "eval.bluesky_pair", "--device", options.device, "--baseline", checkpoint,
                      "--output", directory, "--seed", seed, "--seeds", seeds, "--admm-k", 2,
                      "--dense-extra-rounds", 2, "--directional-heading", "--batched-layer-search",
                      "--joint-layer-search", "--joint-layer-reserve", 10, "--navdata-archive", native_cache,
                      *layer_flags, *navdata_flags),
                     directory.with_suffix(".log"), native_environment)
        report = directory / "comparison.json"
        if exact and name != "incoming":
            source = json.loads(report.read_text())["sources"]["baseline"]
            if not source.get("equivariant_actor") or source.get("admm_warmstart_reflection") != "even":
                raise ValueError("Plan B admission requires mirrored physical ADMM warm starts")
        return report, native_development_score(report, checkpoint, seed, seeds=seeds)

    incoming_q = audit("incoming", options.incoming)
    incoming_panels = {}
    next_milestone = CHECKPOINT_EVERY

    def monitor(learner, args, _decision, _totals, _final):
        nonlocal next_milestone
        updates = learner.updates - first_update
        if updates < next_milestone:
            return
        if updates != next_milestone:
            raise ValueError(f"Expected an exact {next_milestone}-update checkpoint, got {updates}")
        next_milestone += CHECKPOINT_EVERY
        name = f"update-{updates:05d}"
        checkpoint = args.output / f"checkpoint-{learner.updates:07d}.pt"
        q = audit(name, checkpoint)
        panels = []
        for seed, seeds in development_panels:
            if seed not in incoming_panels:
                incoming_panels[seed] = evaluate("incoming", options.incoming, seed, seeds)
            before_path, before = incoming_panels[seed]
            after_path, after = evaluate(name, checkpoint, seed, seeds)
            passes = passes_native_development(after, before)
            panels.append(dict(seed=seed, seeds=seeds, incoming=str(before_path), candidate=str(after_path),
                               incoming_score=before, candidate_score=after, passes=passes))
            if not passes:
                break
        entry = dict(updates=updates, checkpoint=str(checkpoint), q=q, panels=panels)
        selections.append(entry)
        if (len(panels) == len(development_panels) and all(panel["passes"] for panel in panels)
                and (not exact or q["native"] >= 3)):
            admitted.append(entry)
        _write(options.output / "milestone-evaluations.json", dict(incoming_q=incoming_q, checkpoints=selections))

    anchor_path = options.anchor or options.frozen_baseline
    anchor_scope = ["--anchor-families", *options.anchor_families] if options.anchor_families else []
    initialization = (["--initialize", str(options.incoming), "--initialize-critic", "--detection-km", "35",
                       "--equivariant-actor", *layer_flags] if exact else
                      ["--resume", str(options.incoming), "--anchor", str(anchor_path), "--anchor-weight", ".65",
                       *anchor_scope])
    training = build_parser().parse_args([
        "--plan", "b", "--device", options.device, "--output", str(options.output / "training"),
        *initialization,
        "--envs", "512", "--aircraft", "20", "--width", str(width), "--sim-dt", ".25", "--control-dt", "6",
        "--decisions", str(first_decision + (200 if exact else 23) + UPDATES // 10),
        "--target-updates", str(first_update + UPDATES),
        "--updates", "10", "--batch", "64", "--inference-batch", "256", "--replay", "8192",
        "--checkpoint-every", str(CHECKPOINT_EVERY), "--log-every", "50", "--warmup", "24",
        "--learning-rate", ".00012", "--cosine-updates", str(UPDATES), "--update-schedule-from-resume",
        "--seed", "89731", "--family", "full_mixture", "--physics64",
        "--performance-calibration", str(options.calibration), "--timed-departures", "--stratified-replay",
        "--admm-k", "2", "--dense-extra-rounds", "2", "--directional-heading", "--batched-layer-search",
        "--joint-layer-search", "--joint-layer-reserve", "10", "--shaped-reward", "--lookahead-weight", ".7",
        "--navigation-weight", "1.5", "--projection-weight", ".12", "--raw-risk-weight", ".75",
        "--crossing-exploration-steps", str(options.crossing_exploration_steps),
        "--return-window", "24",
        "--multi-step-return", "--executed-action-critic", "--projected-target-action",
        "--projected-actor-ste", "--mirror-replay",
    ])
    (options.output / "status.txt").write_text("stage1-training\n")
    train(training, monitor=monitor)
    if len(selections) != UPDATES // CHECKPOINT_EVERY:
        raise RuntimeError("Stage 1 completed without all milestone evaluations")
    if not admitted:
        (options.output / "status.txt").write_text("stage1-held-by-native-development\n")
        return

    def quality(entry):
        rows = entry["panels"][-1]["candidate_score"].values()
        return (sum(row["arrivals"] for row in rows), -sum(row["collisions"] for row in rows),
                -sum(row["cost"] for row in rows))

    selected = max(admitted, key=quality)
    gate = options.output / "stage2-native-admission.json"
    _write(gate, dict(protocol="native-guarded-development", incoming=str(options.incoming), **selected))
    if exact:
        (options.output / "status.txt").write_text("stage1-exact-equivariant-admitted\n")
        return
    (options.output / "status.txt").write_text("stage2-native-training\n")
    _run_command(("-m", "train.bluesky_stage2", "--device", options.device, "--gate", gate,
                  "--checkpoint", selected["checkpoint"], "--frozen-baseline", options.frozen_baseline,
                  "--raw-reference", options.incoming, "--output", options.output / "stage2", "--episodes", 2500,
                  "--executed-action-critic", "--return-window", 24, "--multi-step-return"),
                 options.output / "stage2.log", native_environment)
    (options.output / "status.txt").write_text("stage2-formal-completed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("incoming", "frozen-baseline", "calibration", "output", "native-checkout",
                 "q-audit", "q-source", "q-crossing-four", "q-crossing-six"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exact-equivariant", action="store_true",
                        help="Train 35 km mirror equivariance with the exact safety layer and stop at the Stage 2 gate")
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-families", choices=MIXTURE_NAMES, nargs="+", default=[])
    parser.add_argument("--crossing-exploration-steps", type=int, default=1)
    parser.add_argument("--feedback-audit", type=Path)
    parser.add_argument("--device", choices=("cpu", "ppu"), default="ppu")
    run(parser.parse_args())
