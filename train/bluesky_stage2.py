"""Admit a Stage 1 checkpoint, then choose Stage 2 checkpoints on development traffic."""

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from types import SimpleNamespace

import torch

from eval.bluesky_pair import FAMILIES, _checksum, _load, _measure_batch, compare
from eval.stage_acceptance import (
    EXACT_LAYER_OPTIONS, EXACT_NATIVE_PROTOCOL, MPC_COMMUNICATION, assess, assess_guarded,
    exact_refinement_acceptance,
)
from train.bluesky_td3 import _save, build_parser, train


MPC_DEVELOPMENT = "native-exact-mpc-development"
MPC_DEVELOPMENT_SEEDS = (61231, 75121)


def _admit(gate, checkpoint):
    if gate.get("protocol") == MPC_DEVELOPMENT:
        panels = gate["panels"]
        if tuple((panel["seed"], panel["seeds"]) for panel in panels) != tuple(
                (seed, 5) for seed in MPC_DEVELOPMENT_SEEDS):
            raise ValueError("MPC refinement requires both locked five-seed development panels")
        for panel in panels:
            seed = panel["seed"]
            candidate = native_development_score(Path(panel["candidate"]), checkpoint, seed, seeds=5, mpc=True)
            incoming = native_development_score(Path(panel["incoming"]), Path(gate["incoming"]), seed, seeds=5)
            if not passes_native_development(candidate, incoming):
                raise SystemExit(f"MPC refinement held on native development seed {seed}")
        return True
    if gate.get("protocol") == "native-guarded-development":
        panel = next((panel for panel in gate["panels"] if panel["seeds"] == 5), None)
        if panel is None:
            raise ValueError("Stage 2 needs a native five-seed development confirmation")
        candidate = native_development_score(Path(panel["candidate"]), checkpoint, panel["seed"],
                                             seeds=5, mode=panel.get("candidate_mode", "baseline"))
        incoming = native_development_score(Path(panel["incoming"]), Path(gate["incoming"]), panel["seed"], seeds=5)
        if not passes_native_development(candidate, incoming):
            raise SystemExit(f"Stage 2 held on native development seed {panel['seed']}")
        return True
    families = gate["families"]
    if (gate.get("diagnostic_observation_profile", gate.get("diagnostic_observation_range", False))
            or gate.get("diagnostic_projection", False) or gate.get("diagnostic_contact_tolerance", False)):
        raise ValueError("Stage 2 requires a standard development gate report")
    if (set(families) != set(FAMILIES) or gate["sha256"] != _checksum(checkpoint)
            or not gate.get("projection_enabled", True)):
        raise ValueError("Stage 1 gate requires six families and the exact checkpoint digest")
    for figures in families.values():
        total = sum(figures[k] for k in ("arrivals", "collisions", "exits", "timeouts"))
        if total != figures["aircraft"] or total < 60:
            raise ValueError("Incomplete Stage 1 family evaluation")
    n = sum(f["aircraft"] for f in families.values())
    arrival = sum(f["arrivals"] for f in families.values()) / n
    collision = sum(f["collisions"] for f in families.values()) / n
    if not (arrival > .90 and collision < .03):
        raise SystemExit(f"Stage 2 held at Stage 1 development rates {arrival:.4f} arrival, {collision:.4f} collision")
    return False


def native_development_score(path, checkpoint, seed, *, seeds=3, mode="baseline", mpc=False):
    report = json.loads(path.read_text())
    communication = MPC_COMMUNICATION if mpc else dict(admm_communication_period=None, admm_gap_policy="hold")
    if (report["sim_dt"] != 0.05 or report["event_sampling_seconds"] != 0.05
            or report["control_dt"] != 6 or seed in range(11997, 12006)
            or not report["batched_layer_search"]
            or not report["joint_layer_search"] or report["joint_layer_reserve_m"] != 10.0
            or report.get("without_potential_edges", False)
            or not report["directional_heading"] or report.get("cold_start_admm")
            or report.get("admm_packet_loss") is not None or report.get("admm_schedule", "jacobi") != "jacobi"
            or report.get("admm_delay_steps") is not None
            or (report.get("admm_communication_period"), report.get("admm_gap_policy", "hold"))
            != (communication["admm_communication_period"], communication["admm_gap_policy"])
            or report["sources"][mode]["sha256"] != _checksum(checkpoint)
            or set(report["families"]) != set(FAMILIES)):
        raise ValueError(f"Native development protocol or checkpoint differs in {path}")
    if mpc:
        protocol = EXACT_NATIVE_PROTOCOL | MPC_COMMUNICATION | dict(first_seed=seed, num_seeds=seeds)
        source = report["sources"][mode]
        if (not protocol.items() <= report.items() or report.get("detection_metres") != 35000.
                or not source.get("equivariant_actor") or source.get("admm_warmstart_reflection") != "even"):
            raise ValueError(f"MPC development must use the exact 35 km equivariant policy in {path}")
    score = {}
    for family, modes in report["families"].items():
        result = modes[mode]
        rows = result["rows"]
        if (result["admm_k"] != 2 or (mpc and result["admm_fallback"] != 0)
                or [row["seed"] for row in rows] != list(range(seed, seed + seeds))):
            raise ValueError(f"Unpaired development seeds in {path}")
        for row in rows:
            counts = sum(row[k] for k in ("arrivals", "collisions", "exits", "unresolved"))
            if counts != 12 or row["aircraft"] != 12:
                raise ValueError(f"Incomplete development outcomes in {path}")
        score[family] = dict(aircraft=12 * seeds, arrivals=sum(r["arrivals"] for r in rows),
                             collisions=sum(r["collisions"] for r in rows),
                             cost=sum(r["action_cost_per_aircraft"] for r in rows))
    return score


def passes_native_development(candidate, incoming):
    arrivals = sum(r["arrivals"] for r in candidate.values())
    baseline = sum(r["arrivals"] for r in incoming.values())
    aircraft = sum(r["aircraft"] for r in candidate.values())
    head_on = candidate["head_on"]
    return ((100 * arrivals >= 95 * aircraft or arrivals - baseline >= 4)
            and 100 * head_on["arrivals"] >= 95 * head_on["aircraft"]
            and all(100 * r["arrivals"] >= 88 * r["aircraft"] for r in candidate.values()))


def _protocol(**overrides):
    defaults = dict(aircraft=12, control_dt=6.0, max_steps=400, seed=59201, seeds=5, device="ppu",
                    admm_k=2, admm_fallback=0, control_frame_diagnostics=False, component_projection=False,
                    vertical_heading=False, directional_heading=True, dense_extra_rounds=2, ablation=False,
                    family=FAMILIES)
    return SimpleNamespace(**(defaults | overrides))


def _evaluate(actor, output, device, settings):
    output.mkdir()
    results = {}
    raw = {}
    for family in FAMILIES:
        paired = dict(_measure_batch([
            ("projected", (actor, output / family, settings, device, family, settings.admm_k)),
            ("raw", (actor, output / f"{family}-without-projection", settings, device, family, 0)),
        ]))
        results[family] = paired["projected"]
        raw[family] = paired["raw"]
    rows = [row for values in results.values() for row in values["rows"]]
    raw_rows = [row for values in raw.values() for row in values["rows"]]
    score = {
        "arrivals": sum(x["arrivals"] for x in rows),
        "collisions": sum(x["collisions"] for x in rows),
        "cost": sum(x["action_cost_per_aircraft"] for x in rows) / len(rows),
        "head_on": sum(x["arrivals"] for x in results["head_on"]["rows"]),
        "yaw_plus15": sum(x["arrivals"] for x in results["yaw_plus15"]["rows"]),
        "yaw_minus15": sum(x["arrivals"] for x in results["yaw_minus15"]["rows"]),
        "raw_arrivals": sum(x["arrivals"] for x in raw_rows),
        "raw_collisions": sum(x["collisions"] for x in raw_rows),
        "raw_head_on": sum(x["arrivals"] for x in raw["head_on"]["rows"]),
        "aircraft_by_family": {name: sum(row["aircraft"] for row in group["rows"])
                               for name, group in results.items()},
        "arrivals_by_family": {name: sum(row["arrivals"] for row in group["rows"])
                               for name, group in results.items()},
    }
    report = {"score": score, "families": results, "without_projection": raw}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return score


def _quality(score, confirmation=()):
    panels = (score, *confirmation)
    return (sum(panel["arrivals"] for panel in panels),
            -sum(panel["collisions"] for panel in panels),
            -sum(panel["cost"] for panel in panels))


def _meets_development_floors(score):
    n = score["aircraft_by_family"]
    return (100 * score["head_on"] >= 95 * n["head_on"]
            and all(100 * count >= 88 * n[name] for name, count in score["arrivals_by_family"].items()))


def run(options):
    short = options.short_refine
    equivariant_refinement = options.equivariant_refinement
    gate = json.loads(options.gate.read_text())
    mpc = gate.get("protocol") == MPC_DEVELOPMENT
    if mpc and not short:
        raise ValueError("MPC admission requires short native refinement")
    if equivariant_refinement and not short:
        raise ValueError("The zero-training equivariant admission is only for short refinement")
    if short:
        if not 500 <= options.episodes <= 1000 or options.every_episodes != 200 or not all(
                (options.q_audit, options.q_source, options.q_crossing_four, options.q_crossing_six)):
            raise ValueError("Short native refinement uses 500–1000 episodes, 200-episode checks and a frozen Q audit")
    elif not 2000 <= options.episodes <= 5000 or options.every_episodes <= 0:
        raise ValueError("Stage 2 needs 2000–5000 episodes and a positive validation interval")
    guarded = _admit(gate, options.checkpoint)
    if equivariant_refinement:
        panel = next(panel for panel in gate["panels"] if panel["seeds"] == 5)
        source = json.loads(Path(panel["candidate"]).read_text())["sources"][panel.get("candidate_mode", "baseline")]
        if not source.get("equivariant_actor"):
            raise ValueError("The admission run did not test the equivariant Actor")
    if options.device == "ppu" and importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    device = torch.device("cuda:0" if options.device == "ppu" else "cpu")
    torch.set_num_threads(1)
    options.output.mkdir(parents=True)
    safety = dict(batched_layer_search=True, joint_layer_search=True, joint_layer_reserve=10.0) if guarded else {}
    if short:
        if not guarded:
            raise ValueError("Short refinement requires the guarded native development admission")
        safety.update(EXACT_LAYER_OPTIONS)
    communication = MPC_COMMUNICATION if mpc else {}
    settings = _protocol(device=options.device, navdata_archive=options.output / "cache", **safety, **communication)
    if short:
        settings.seed = 61231
        settings.navdata = options.navdata
    incoming = options.checkpoint
    if equivariant_refinement:
        source = torch.load(incoming, map_location="cpu", weights_only=True)
        source["equivariant_actor"] = True
        incoming = options.output / "incoming-equivariant.pt"
        torch.save(source, incoming)
        del source
    baseline_actor = _load(incoming, device)
    if options.detection_km is not None:
        profile = replace(baseline_actor.observation_profile, detection_metres=1000 * options.detection_km)
        if profile != baseline_actor.observation_profile:
            baseline_actor.observation_profile = profile
            state = torch.load(incoming, map_location="cpu", weights_only=True)
            state["observation_profile"] = asdict(profile)
            incoming = options.output / "incoming-observation-profile.pt"
            torch.save(state, incoming)
            del state
    def audit(checkpoint, output):
        report_path = output / "frozen-q.json"
        with (output / "frozen-q.log").open("w") as log:
            subprocess.run([sys.executable, "-B", str(options.q_audit), "--source", str(options.q_source),
                            "--crossing-four", str(options.q_crossing_four), "--crossing-six", str(options.q_crossing_six),
                            "--checkpoint", f"refine={checkpoint}", "--output", str(report_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(report_path.read_text())["refine"]
        if result["checkpoint_same_digest"] != _checksum(checkpoint) or len(result["cases"]) != 4:
            raise ValueError("Invalid four-scenario frozen Q report")
        return sum(case["improvement_on_changed_aircraft"] > 0 for case in result["cases"].values())

    def evaluate_development(actor, output):
        if not mpc:
            return [_evaluate(actor, output, device, settings)]
        output.mkdir()
        scores = []
        for seed in MPC_DEVELOPMENT_SEEDS:
            panel_settings = SimpleNamespace(**(vars(settings) | {"seed": seed}))
            scores.append(_evaluate(actor, output / str(seed), device, panel_settings))
        return scores

    starting_dir = options.output / "development-incoming"
    starting, *confirmation = evaluate_development(baseline_actor, starting_dir)
    starting_q = audit(incoming, starting_dir) if short else None
    if short and starting_q < 3:
        raise ValueError("Short refinement requires at least 3/4 incoming frozen Q rankings")
    del baseline_actor
    best = _quality(starting, confirmation)
    selected = options.output / "selected.pt"
    if incoming == options.checkpoint:
        copyfile(incoming, selected)
    else:
        selected.hardlink_to(incoming)
    evaluations = [{"episodes": 0, "selected": True, "score": starting,
                    "checkpoint": str(incoming), "checkpoint_sha256": _checksum(selected)}]
    if mpc:
        evaluations[0]["confirmation_score"], = confirmation
    if incoming != options.checkpoint:
        evaluations[0]["source_checkpoint"] = str(options.checkpoint)
        evaluations[0]["source_checkpoint_sha256"] = _checksum(options.checkpoint)
    if short:
        evaluations[0]["frozen_q"] = starting_q
    next_validation = options.every_episodes
    regressions = 0

    def monitor(learner, run_args, decision, totals, final):
        nonlocal best, next_validation, regressions
        episodes = int(totals[0])
        if episodes < next_validation and not final:
            return
        validation = options.output / f"development-{decision:06d}"
        score, *confirmation = evaluate_development(learner.actor.eval(), validation)
        preserves = all(_meets_development_floors(panel) for panel in (score, *confirmation))
        quality = _quality(score, confirmation)
        checkpoint_args = SimpleNamespace(output=validation, width=run_args.width, plan=run_args.plan,
                                          detection_km=run_args.detection_km,
                                          geodesic_navigation=run_args.geodesic_navigation,
                                          cross_track_observation=run_args.cross_track_observation,
                                          far_conflict_observation=run_args.far_conflict_observation,
                                          selected_altitude_observation=run_args.selected_altitude_observation)
        milestone = validation / "checkpoint.pt"
        _save(learner, checkpoint_args, decision, totals, milestone.name)
        digest = _checksum(milestone)
        q = audit(milestone, validation) if short else None
        if short:
            regressions = regressions + 1 if quality < best else 0
        replace = preserves and quality > best and (not short or q >= 3)
        if replace:
            pending_selection = selected.with_suffix(".link")
            try:
                pending_selection.hardlink_to(milestone)
                pending_selection.replace(selected)
            finally:
                pending_selection.unlink(missing_ok=True)
            best = quality
        evaluations.append({"episodes": episodes, "updates": learner.updates, "selected": replace,
                            "preserves_development_score": preserves, "score": score,
                            "checkpoint": str(milestone.relative_to(options.output)), "checkpoint_sha256": digest})
        if mpc:
            evaluations[-1]["confirmation_score"], = confirmation
        stop = short and (q < 3 or regressions >= 2)
        if short:
            evaluations[-1].update(frozen_q=q, stopped_early=stop)
        destination = options.output / "selection.json"
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(evaluations, indent=2) + "\n")
        temp.replace(destination)
        next_validation = (episodes // options.every_episodes + 1) * options.every_episodes
        return stop

    learned_options = []
    raw_risk_weight = "0.75"
    if options.stage1_arm in {"combined", "temporal"}:
        learned_options = ["--dense-risk", "--safe-navigation-weight", "0.35"]
        raw_risk_weight = "0.15"
    if options.stage1_arm == "temporal":
        learned_options.append("--overlap-risk")
    if options.teacher_data:
        learned_options += ["--teacher-data", *map(str, options.teacher_data),
                            "--teacher-weight", str(options.teacher_weight)]
    if options.correlated_yaw:
        learned_options.append("--correlated-yaw")
    if options.batched_layer_search:
        learned_options.append("--batched-layer-search")
    if guarded:
        learned_options.extend(("--batched-layer-search", "--joint-layer-search", "--joint-layer-reserve", "10",
                                "--projected-target-action", "--projected-actor-ste", "--mirror-replay",
                                "--training-yaw-degrees", "15"))
    if options.executed_action_critic or guarded:
        learned_options.append("--executed-action-critic")
    if options.unprotected_rollout_fraction:
        learned_options.extend(("--unprotected-rollout-fraction", str(options.unprotected_rollout_fraction)))
    if options.collision_critic_weight:
        learned_options.extend(("--collision-critic-weight", str(options.collision_critic_weight)))
    if options.collision_label_horizon:
        learned_options.extend(("--collision-label-horizon", str(options.collision_label_horizon),
                                "--collision-critic-target", options.collision_critic_target))
    if options.return_window != 1 or options.multi_step_return:
        learned_options.extend(("--return-window", str(options.return_window)))
    if options.multi_step_return:
        learned_options.append("--multi-step-return")
    if options.coordination_weight:
        learned_options.extend(("--coordination-weight", str(options.coordination_weight)))
    if options.topology_replay:
        learned_options.append("--topology-replay")
    if options.expected_mode_risk:
        learned_options.append("--expected-mode-risk")
    if options.initialize_critic or short:
        learned_options.append("--initialize-critic")
    if options.freeze_actor_encoder:
        learned_options.append("--freeze-actor-encoder")
    if getattr(options, "freeze_actor_router", False):
        learned_options.append("--freeze-actor-router")
    if options.detection_km is not None:
        learned_options.extend(("--detection-km", str(options.detection_km)))
    if short:
        learned_options.extend(("--early-layer-search", "--exact-layer-forecast", "--batched-layer-margin", "6"))
    rate = "0.000024" if short else "0.00004"
    cosine = options.episodes * 12 if short else 28000
    anchor = [] if short else ["--anchor", str(options.checkpoint), "--anchor-weight", "1.25"]
    training = build_parser().parse_args([
        "--plan", "a", "--device", options.device, "--output", str(options.output / "training"),
        "--initialize", str(incoming) if short else str(options.checkpoint), *anchor,
        "--workers", "9", "--envs", "8", "--aircraft", "20", "--sim-dt", "0.05", "--control-dt", "6",
        "--target-episodes", str(options.episodes), "--decisions", "30000", "--updates", "4", "--batch", "64",
        "--inference-batch", "72", "--replay", "4096", "--checkpoint-every", "10000",
        "--learning-rate", rate, "--cosine-updates", str(cosine), "--lookahead-weight", "0.7",
        "--rollout-router-temperature", str(options.rollout_router_temperature),
        "--cross-track-weight", str(options.cross_track_weight),
        "--raw-risk-weight", raw_risk_weight, "--navigation-weight", "1.5", "--projection-weight", "0.12",
        "--admm-k", "2", "--dense-extra-rounds", "2", "--directional-heading", "--family", "full_mixture",
        "--overlap-native", "--audit-frames", "--timed-departures", "--stratified-replay",
        "--shaped-reward", "--seed", str(options.seed), *learned_options,
    ])
    if short:
        training.monitor_every_episodes = 200
        training.navdata = options.navdata
    train(training, monitor=monitor)
    if options.development_only:
        return
    if short:
        improved = _checksum(selected) != _checksum(incoming)
        if not improved and equivariant_refinement:
            return
        formal = options.output / "formal-short-refinement"
        candidates = [selected, options.raw_reference]
        if improved:
            candidates.append(incoming)
        compare(_protocol(device=options.device, seed=12001, ablation=True, ablate_baseline=True,
                          baseline=options.frozen_baseline, **safety, **communication,
                          candidates=candidates, output=formal,
                          navdata=options.navdata, navdata_archive=settings.navdata_archive, parallel_pairs=2))
        result = exact_refinement_acceptance(formal / "comparison.json", mpc=mpc)
        (options.output / "stage2-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    formal = options.output / "formal-stage2"
    compare(_protocol(device=options.device, seed=12001, ablation=True, ablate_baseline=guarded,
                      baseline=options.frozen_baseline, **safety,
                      candidates=[selected, options.raw_reference], output=formal,
                      navdata_archive=settings.navdata_archive))
    evaluator = assess_guarded if guarded else assess
    result = evaluator(json.loads((formal / "comparison.json").read_text()))
    (options.output / "stage2-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-baseline", type=Path, required=True)
    parser.add_argument("--raw-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--initialize-critic", action="store_true")
    parser.add_argument("--freeze-actor-encoder", action="store_true")
    parser.add_argument("--freeze-actor-router", action="store_true")
    parser.add_argument("--detection-km", type=float)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--every-episodes", type=int)
    parser.add_argument("--short-refine", action="store_true")
    parser.add_argument("--equivariant-refinement", action="store_true",
                        help="Initialize from the equivariant inference mode verified by the admission report")
    for argument in ("q-audit", "q-source", "q-crossing-four", "q-crossing-six", "navdata"):
        parser.add_argument(f"--{argument}", type=Path)
    parser.add_argument("--seed", type=int, default=83721)
    parser.add_argument("--stage1-arm", choices=("incoming", "control", "combined", "temporal"), default="incoming")
    parser.add_argument("--teacher-data", type=Path, nargs="+", default=[])
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--correlated-yaw", action="store_true")
    parser.add_argument("--rollout-router-temperature", type=float, default=1.0)
    parser.add_argument("--cross-track-weight", type=float, default=0.0)
    parser.add_argument("--batched-layer-search", action="store_true")
    parser.add_argument("--executed-action-critic", action="store_true")
    parser.add_argument("--unprotected-rollout-fraction", type=float, default=0.0)
    parser.add_argument("--collision-critic-weight", type=float, default=0.0)
    parser.add_argument("--collision-label-horizon", type=int, default=0)
    parser.add_argument("--collision-critic-target", choices=("td", "observed"), default="td")
    parser.add_argument("--return-window", type=int, default=1)
    parser.add_argument("--multi-step-return", action="store_true")
    parser.add_argument("--coordination-weight", type=float, default=0.0)
    parser.add_argument("--topology-replay", action="store_true")
    parser.add_argument("--expected-mode-risk", action="store_true")
    parser.add_argument("--device", choices=("cpu", "ppu"), default="ppu")
    options = parser.parse_args()
    if options.every_episodes is None:
        options.every_episodes = 200 if options.short_refine else 500
    run(options)
