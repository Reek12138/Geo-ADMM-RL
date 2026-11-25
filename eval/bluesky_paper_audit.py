"""Verify the completed panels of a frozen native paper suite."""

import argparse
from errno import ESTALE
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from time import sleep

from envs.bluesky_traffic import HELD_TOPOLOGIES
from eval.admm_communication import PACKET_SEED_OFFSET
from eval.bluesky_figure4_matrix import ARMS, CASES, SEEDS
from eval.bluesky_pair import FAMILIES, _checksum
from eval.bluesky_paper_exp4 import DISTURBANCES
from eval.bluesky_paper_exp6 import _verify_one_step_control
from eval.stage_acceptance import MPC_COMMUNICATION, paper_checkpoint_digest, paper_layer_options, paper_uses_mpc


OUTCOMES = ("arrivals", "collisions", "exits", "unresolved")
RESIDUAL_MEANS = ("speed_primal_residual_mean_mps", "altitude_primal_residual_mean_m")
PROJECTED_AND_RAW = ("baseline", "baseline_without_projection")
EXP5_CONFIGURATIONS = {
    "round-budget": {(0, False, None, None, None, "jacobi")} | {
        (k, cold, None, None, None, "jacobi") for k, cold in product((1, 2, 3, 5), (False, True))
    },
    **{
        f"cadence-{policy}": {
            (2, cold, period, None, None, "jacobi") for cold, period in product((False, True), (6, 12, 18, 30))
        } for policy in ("hold", "mpc")
    },
    "loss-delay-coloring": {
        (2, cold, None, loss, delay, schedule)
        for cold, loss, delay, schedule in product((False, True), (0., .2, .5), (0, 1, 2), ("jacobi", "colored"))
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _figure4(data, arms):
    pairs = set(product(CASES, SEEDS, arms))
    rows = data["rows"]
    keys = {f"case{case}-{seed}-{arm}" for case, seed, arm in pairs}
    _require(set(rows) == keys, "Figure 4 has an incomplete paired replay matrix")
    for case in CASES:
        for seed in SEEDS:
            reference = rows[f"case{case}-{seed}-{arms[0]}"]
            for arm in arms:
                row = rows[f"case{case}-{seed}-{arm}"]
                _require((row["case"], row["seed"], row["policy"]) == (case, seed, arm)
                         and row["aircraft"] == sum(row["outcomes"].values()) == 50
                         and row["sim_dt"] == .05 and row["control_dt"] == 6.
                         and row["departures"] == reference["departures"]
                         and row["manifest_sha256"] == reference["manifest_sha256"],
                         f"Unpaired Figure 4 replay for case {case}, seed {seed}, {arm}")
                thresholds = ("collision_h_km", "collision_v_ft", "arrival_dist_km", "max_xtk_km")
                _require(all(row[key] == reference[key] for key in thresholds),
                         f"Changed Figure 4 outcome thresholds for case {case}, seed {seed}, {arm}")
    totals = {}
    for case in CASES:
        totals[case] = {}
        for arm in arms:
            group = [rows[f"case{case}-{seed}-{arm}"] for seed in SEEDS]
            totals[case][arm] = {
                metric: sum(row["outcomes"][metric] for row in group)
                for metric in ("arrivals", "collisions", "exits", "timeouts")
            }
            totals[case][arm]["mean_action_cost_per_aircraft"] = mean(
                row["action_cost_per_aircraft"] for row in group)
    return totals


def _transport_counts(report, rows):
    period = report["admm_communication_period"]
    loss = report["admm_packet_loss"]
    delay = report["admm_delay_steps"]
    mpc = report["admm_gap_policy"] == "mpc"
    if period is None and loss is None and delay is None:
        return {}
    interval = int(period // report["control_dt"]) if period is not None else 1
    counts = dict(control_steps=0, coordination_ticks=0, projection_solves=0, attempted_pair_exchanges=0,
                  dropped_pair_exchanges=0, delivered_pair_exchanges=0, in_flight_pair_exchanges=0)
    if mpc:
        counts.update(mpc_replans=0, mpc_adjusted_aircraft=0,
                      mpc_hold_predicted_pair_contacts=0, mpc_planned_predicted_pair_contacts=0)
    if delay:
        counts["used_delayed_pair_states"] = 0
    age_weighted_sum = 0.
    min_ages = []
    max_ages = []
    residual_sums = dict.fromkeys(RESIDUAL_MEANS, 0.)
    for row in rows:
        communication = row["communication"]
        steps = row["control_steps"]
        expected_ticks = (steps + interval - 1) // interval
        _require(communication["coordination_ticks"] == communication["projection_solves"] == expected_ticks,
                 f"Unexpected communication cadence for seed {row['seed']}")
        if expected_ticks:
            for key in RESIDUAL_MEANS:
                value = communication[key]
                _require(value is not None and isfinite(value) and value >= 0,
                         f"Invalid {key} for seed {row['seed']}")
                residual_sums[key] += value * expected_ticks
        attempted = communication["attempted_pair_exchanges"]
        dropped = communication["dropped_pair_exchanges"]
        delivered = communication["delivered_pair_exchanges"]
        in_flight = communication.get("in_flight_pair_exchanges", 0)
        _require(min(attempted, dropped, delivered, in_flight) >= 0 and attempted == dropped + delivered + in_flight
                 and (loss not in (None, 0) or dropped == 0),
                 f"Unaccounted communication exchanges for seed {row['seed']}")
        if delay:
            used = communication["used_delayed_pair_states"]
            min_age = communication["min_used_message_age_steps"]
            mean_age = communication["mean_used_message_age_steps"]
            max_age = communication["max_used_message_age_steps"]
            _require(communication["delay_steps"] == delay and used >= 0 and
                     ((used == 0 and (min_age, mean_age, max_age) == (None, None, None)) or
                      (used > 0 and delay <= min_age <= mean_age <= max_age < steps)),
                     f"Invalid delivered-message ages for seed {row['seed']}")
            if used:
                age_weighted_sum += used * mean_age
                min_ages.append(min_age)
                max_ages.append(max_age)
            counts["used_delayed_pair_states"] += used
        if mpc:
            gap = row["gap_mpc"]
            _require(gap["replans"] == steps - expected_ticks, f"Missing MPC gap replan for seed {row['seed']}")
            _require(0 <= gap["planned_predicted_pair_contacts"] <= gap["hold_predicted_pair_contacts"],
                     f"MPC increased predicted contacts for seed {row['seed']}")
            for key in ("replans", "adjusted_aircraft", "hold_predicted_pair_contacts", "planned_predicted_pair_contacts"):
                counts[f"mpc_{key}"] += gap[key]
        counts["control_steps"] += steps
        for key in ("coordination_ticks", "projection_solves", "attempted_pair_exchanges",
                    "dropped_pair_exchanges", "delivered_pair_exchanges"):
            counts[key] += communication[key]
        counts["in_flight_pair_exchanges"] += in_flight
    if counts["attempted_pair_exchanges"]:
        counts["observed_loss"] = counts["dropped_pair_exchanges"] / counts["attempted_pair_exchanges"]
    if counts["coordination_ticks"]:
        counts["attempted_pair_exchanges_per_tick"] = counts["attempted_pair_exchanges"] / counts["coordination_ticks"]
        counts.update({key: total / counts["projection_solves"] for key, total in residual_sums.items()})
    if delay and counts["used_delayed_pair_states"]:
        counts.update(min_used_message_age_steps=min(min_ages),
                      mean_used_message_age_steps=age_weighted_sum / counts["used_delayed_pair_states"],
                      max_used_message_age_steps=max(max_ages))
    return counts


def _communication_metrics(report):
    runs = [group["baseline"] for group in report["families"].values()]
    rows = [row for run in runs for row in run["rows"]]
    totals = {key: sum(row[key] for row in rows) for key in ("aircraft", *OUTCOMES)}
    seed_totals = {}
    for index, seed in enumerate(SEEDS):
        seed_totals[seed] = {key: sum(run["rows"][index][key] for run in runs) for key in OUTCOMES}
    return {
        **totals,
        "mean_action_cost_per_aircraft": mean(row["action_cost_per_aircraft"] for row in rows),
        "mean_decision_ms_per_batch_across_families": mean(run["mean_decision_ms_per_batch"] for run in runs),
        "wall_seconds": report["execution_wall_seconds"],
        "per_seed": seed_totals,
        "transport": _transport_counts(report, rows),
    }



def _paired_outcomes(reference, intervention):
    return {
        seed: {"arrival_gain": counts["arrivals"] - reference["per_seed"][seed]["arrivals"],
               "collision_reduction": reference["per_seed"][seed]["collisions"] - counts["collisions"]}
        for seed, counts in intervention["per_seed"].items()
    }


def _network_pairs(arms, fields):
    references = {
        tuple(arm["configuration"][field] for field in fields): label
        for label, arm in arms.items()
    }
    pairs = {}
    for factor, baseline in (("packet_loss", 0.), ("delay_steps", 0), ("schedule", "jacobi")):
        comparisons = {}
        for label, arm in arms.items():
            config = arm["configuration"]
            if config[factor] == baseline:
                continue
            reference_config = config | {factor: baseline}
            reference_label = references.get(tuple(reference_config[field] for field in fields))
            if reference_label is None:
                continue
            reference = arms[reference_label]
            exchange_rate = "attempted_pair_exchanges_per_tick"
            comparisons[label] = {
                "reference_arm": reference_label,
                "per_seed": _paired_outcomes(reference, arm),
                "total_exit_delta": arm["exits"] - reference["exits"],
                "mean_action_cost_delta": arm["mean_action_cost_per_aircraft"] - reference["mean_action_cost_per_aircraft"],
                "mean_batch_decision_ms_delta": (arm["mean_decision_ms_per_batch_across_families"]
                                                 - reference["mean_decision_ms_per_batch_across_families"]),
                "attempted_pair_exchanges_per_tick_delta": arm["transport"][exchange_rate] - reference["transport"][exchange_rate],
            }
        pairs[factor] = comparisons
    return pairs


def _disturbance_effects(row):
    switches = row["disturbances"]
    seed = row["seed"]
    popup = row["popup"]
    actual = popup["actual_seconds"]
    if "popup" in switches:
        _require(actual is not None and abs(actual - popup["scheduled_seconds"]) <= 1e-7,
                 f"Missing scheduled pop-up insertion for seed {seed}")
    else:
        _require(actual is None, f"Unexpected pop-up insertion for seed {seed}")
    slots = row["noncooperative_slots"]
    targets = row["noncooperative_targets"]
    _require(len(slots) == len(set(slots)) == (2 if "noncooperative" in switches else 0)
             and set(targets) == set(map(str, slots))
             and all(0 <= slot < 20 for slot in slots)
             and all(len(value) == 3 and all(isfinite(v) for v in value) for value in targets.values()),
             f"Missing verified noncooperative targets for seed {seed}")
    wind = row["wind"]
    if "wind-field" in switches:
        _require(len(wind["profiles"]) == 2 and wind["aircraft_samples"] > 0 and wind["max_mps"] > 0
                 and 0 <= wind["max_ground_velocity_error_mps"] <= 1e-7, f"Missing measured wind for seed {seed}")
    else:
        _require(not wind["profiles"] and wind["max_mps"] == 0., f"Unexpected wind for seed {seed}")


def audit(trial, *, inherited_trial=None):
    trial = trial.resolve()
    base = inherited_trial.resolve() if inherited_trial is not None else trial
    acceptance = base / "acceptance.json"
    inputs = (base / "inputs.txt").read_text().splitlines()
    checkpoint = Path(inputs[0])
    digest = paper_checkpoint_digest(acceptance, checkpoint)
    acceptance_digest = _checksum(acceptance)
    release = json.loads(acceptance.read_text())
    observation = release["sources"]["candidate_1"]["observation_profile"]
    deployed_mpc = paper_uses_mpc(acceptance)
    safety = paper_layer_options(acceptance)
    native_safety = {
        "early_layer_search": safety.get("early_layer_search", False),
        "exact_layer_forecast": safety.get("exact_layer_forecast", False),
        "batched_layer_margin_m": safety.get("batched_layer_margin", 0.),
    }
    evidence = {}

    def read(path, expected_sha=None):
        path = Path(path).resolve()
        root = trial if path.is_relative_to(trial) else base
        key = str(path.relative_to(root))
        for attempt in range(4):
            try:
                raw = path.read_bytes()
                break
            except OSError as error:
                if error.errno != ESTALE or attempt == 3:
                    raise
                sleep(.2 * (attempt + 1))
        content = json.loads(raw)
        digest = sha256(raw).hexdigest()
        _require(expected_sha is None or digest == expected_sha, f"Changed paper result {path}")
        evidence[key] = digest
        return content

    def native(path, families, arms, size, *, sha=None, policy_digest=digest, deployment=False):
        report = read(path, sha)
        _require(report["sim_dt"] == report["event_sampling_seconds"] == .05 and report["control_dt"] == 6
                 and (report["first_seed"], report["num_seeds"]) == (SEEDS[0], len(SEEDS))
                 and report["sources"]["baseline"]["sha256"] == policy_digest
                 and set(report["families"]) == set(families), f"Unpaired native report {path}")
        if deployment and deployed_mpc:
            _require(MPC_COMMUNICATION.items() <= report.items(), f"Wrong deployed MPC protocol in {path}")
        for groups in report["families"].values():
            _require(set(groups) == set(arms), f"Unexpected policy arms in {path}")
            for label, arm in groups.items():
                _require(arm["admm_fallback"] == 0 and
                         (not label.endswith("_without_projection") or arm["admm_k"] == 0),
                         f"Unexpected projection budget in {path}")
                if arm["admm_k"]:
                    _require(all(report.get(key, False) == value for key, value in native_safety.items()),
                             f"Unexpected safety profile in {path}")
                rows = arm["rows"]
                _require(tuple(row["seed"] for row in rows) == SEEDS
                         and all(row["aircraft"] == size and sum(row[k] for k in OUTCOMES) == size for row in rows),
                         f"Incomplete native outcomes in {path}")
        return report

    results = {}
    figures = base / "exp1/summary.json"
    if figures.is_file():
        figure = read(figures)
        _require(figure["geo_checkpoint_sha256"] == digest and figure["stage2_acceptance_sha256"] == acceptance_digest
                 and figure["sim_dt"] == .05 and tuple(figure["seeds"]) == SEEDS, "Unexpected Figure 4 provenance")
        results["exp1"] = _figure4(figure, ARMS)
        for label, row in figure["rows"].items():
            if row["policy"] == "geo":
                _require(row["checkpoint_sha256"] == digest and
                         all(row["layer_safety_options"].get(key) == value for key, value in safety.items()),
                         f"Wrong Geo policy or safety profile in {label}")
            if row["policy"] == "geo" and deployed_mpc:
                communication = row.get("communication", {})
                _require(MPC_COMMUNICATION.items() <= communication.items() and communication.get("gap_ticks", 0) > 0,
                         f"Wrong Geo MPC profile in {label}")
            _require(read(base / "exp1" / f"{label}.json") == row, f"Figure 4 replay differs from its summary: {label}")

    generalization = base / "exp2-3/summary.json"
    if generalization.is_file():
        combined = read(generalization)
        _require((combined["checkpoint_sha256"], combined["acceptance_sha256"]) == (digest, acceptance_digest),
                 "Unexpected Exp 2/3 provenance")
        for key, sizes, families in (("exp2", {"30"}, ("crossing6", *HELD_TOPOLOGIES)),
                                     ("exp3", {"30", "50", "80"}, ("crossing6",))):
            ref = combined["panels"][key]
            variants = read(ref["path"], ref["sha256"])["fleet_sizes"]
            _require(set(variants) == sizes, f"Unexpected fleet sizes for {key}")
            for size, scenario in variants.items():
                report = native(scenario["report"], families, PROJECTED_AND_RAW, int(size), deployment=True)
                for family, arms in report["families"].items():
                    for arm, data in arms.items():
                        _require("mean_decision_ms_per_batch" in data and
                                 all({"max_degree", "max_clique_size", "peak_aircraft"} <= row["conflict_graph"].keys()
                                     and row["conflict_graph"]["peak_aircraft"] == int(size) for row in data["rows"]),
                                 f"Missing concurrent graph evidence for {key}/{family}/{arm}")
                        if family in HELD_TOPOLOGIES:
                            routes = int(family.removeprefix("crossing"))
                            _require(all(row.get("route_count") == routes for row in data["rows"]),
                                     f"Wrong held topology for {family}")
            results[key] = combined["panels"][key]

    disturbances = base / "exp4/summary.json"
    if disturbances.is_file():
        report = read(disturbances)
        _require((report["checkpoint_sha256"], report["acceptance_sha256"]) == (digest, acceptance_digest),
                 "Unexpected Exp 4 provenance")
        for field, families, size in (("head_on_pressure", ("head_on",), 30), ("yaw", ("yaw_plus15", "yaw_minus15"), 12)):
            ref = report[field]
            comparison = native(ref["path"], families, PROJECTED_AND_RAW, size, sha=ref["sha256"], deployment=True)
            if field == "yaw":
                protocol = dict(max_goal_heading_error_deg=2, consecutive_control_ticks=3)
                _require(comparison["yaw_recovery_protocol"] == protocol, "Unexpected yaw recovery protocol")
                for arms in comparison["families"].values():
                    for mode in arms.values():
                        for row in mode["rows"]:
                            recovery = row["yaw_recovery"]
                            times = recovery["first_sustained_alignment_seconds"]
                            _require(len(times) == size
                                     and sum(t is not None for t in times) == recovery["recovered_aircraft"],
                                     f"Missing sustained yaw alignment outcomes for seed {row['seed']}")
        layouts = {}
        condition_outcomes = {}
        _require(set(report["disturbance_rows"]) == set(DISTURBANCES), "Incomplete disturbance conditions")
        for condition, runs in report["disturbance_rows"].items():
            _require(tuple(item["seed"] for item in runs) == SEEDS, f"Unpaired {condition} disturbance seeds")
            for item in runs:
                row = read(item["report"], item["sha256"])
                _require(item == (row | {"report": item["report"], "sha256": item["sha256"]})
                         and row["sim_dt"] == .05 and row["control_dt"] == 6
                         and row["family"] == "crossing6" and row["checkpoint_sha256"] == digest
                         and row["disturbances"] == list(DISTURBANCES[condition]) and row["projection"]
                         and all(row["layer_safety_options"].get(key) == value for key, value in safety.items())
                         and sum(row["outcomes"].values()) == row["aircraft"] == 20 + ("popup" in row["disturbances"])
                         and row["base_routes_sha256"] == layouts.setdefault(item["seed"], row["base_routes_sha256"]),
                         f"Unpaired disturbance result {item['report']}")
                if deployed_mpc:
                    traffic = row.get("communication", {})
                    _require(MPC_COMMUNICATION.items() <= traffic.items() and traffic.get("gap_ticks", 0) > 0,
                             f"Wrong disturbance MPC protocol for seed {row['seed']}")
                _disturbance_effects(row)
            condition_outcomes[condition] = {
                key: sum(row["outcomes"][key] for row in runs)
                for key in ("arrivals", "collisions", "exits", "timeouts")
            }
        results["exp4"] = {"disturbance_replays": sum(len(x) for x in report["disturbance_rows"].values()),
                           "disturbance_outcomes": condition_outcomes}

    communication = base / "exp5/summary.json"
    manifest = read(communication) if communication.is_file() else None
    if manifest:
        _require((manifest["checkpoint_sha256"], manifest["acceptance_sha256"]) == (digest, acceptance_digest)
                 and set(manifest["panels"]) == set(EXP5_CONFIGURATIONS), "Unexpected Exp 5 provenance")
    communication_progress = {}
    communication_metrics = {}
    fields = ("admm_k", "cold_start", "communication_period", "packet_loss", "delay_steps", "schedule")
    for panel, conditions in EXP5_CONFIGURATIONS.items():
        ref = manifest["panels"][panel] if manifest else None
        path = Path(ref["output"]) if ref else base / "exp5" / panel / "summary.json"
        communication_progress[panel] = 0
        if not path.is_file() and ref is None:
            continue
        rows = read(path, ref["sha256"] if ref else None)["results"]
        actual = {tuple(row[key] for key in fields) for row in rows.values()}
        _require(len(rows) == len(actual) and actual <= conditions, f"Unexpected configurations in {panel}")
        if ref:
            _require(len(rows) == ref["arms"] == len(conditions) and actual == conditions,
                     f"Incomplete configurations in {panel}")
        communication_metrics[panel] = {}
        for label, row in rows.items():
            report = native(row["report"], FAMILIES, ("baseline",), 12, sha=row["report_sha256"])
            projected = row["admm_k"] > 0
            run_conditions = {
                "cold_start_admm": row["cold_start"], "admm_communication_period": row["communication_period"],
                "admm_packet_loss": row["packet_loss"], "admm_delay_steps": row["delay_steps"],
                "admm_schedule": row["schedule"], "dense_extra_rounds": 0,
                "admm_gap_policy": "mpc" if panel == "cadence-mpc" else "hold",
                "admm_communication_seed_offset": PACKET_SEED_OFFSET if row["packet_loss"] is not None else None,
                "detection_metres": observation["detection_metres"], "graph_statistics": True,
                "diagnostic_projection": projected, "directional_heading": projected,
                "batched_layer_search": projected, "joint_layer_search": projected,
                "joint_layer_reserve_m": 10. if projected else 0., "without_potential_edges": False,
                "vertical_heading": False, "component_projection": False,
                "accelerated_layer_forecast": False, "preserve_imminent_clearance": False,
            }
            _require(run_conditions.items() <= report.items()
                     and report["sources"]["baseline"]["observation_profile"] == observation
                     and all(family["baseline"]["admm_k"] == row["admm_k"] for family in report["families"].values()),
                     f"ADMM configuration mismatch in {row['report']}")
            communication_metrics[panel][label] = {
                **_communication_metrics(report), "configuration": {key: row[key] for key in fields},
            }
        communication_progress[panel] = len(rows)
    raw_reference = communication_metrics.get("round-budget", {}).get("k0-raw")
    warm_cold_pairs = {}
    for panel, arms in communication_metrics.items():
        comparisons = {}
        cold_labels = {
            tuple(arm["configuration"][key] for key in fields): label
            for label, arm in arms.items() if arm["configuration"]["cold_start"]
        }
        for label, arm in arms.items():
            if raw_reference:
                arm["paired_delta_from_k0"] = _paired_outcomes(raw_reference, arm)
            config = arm["configuration"]
            if config["admm_k"] == 0 or config["cold_start"]:
                continue
            cold_configuration = config | {"cold_start": True}
            cold_label = cold_labels.get(tuple(cold_configuration[key] for key in fields))
            if cold_label is None:
                continue
            cold = arms[cold_label]
            comparisons[label] = {
                "cold_arm": cold_label,
                "per_seed_warm_minus_cold": _paired_outcomes(cold, arm),
                "mean_action_cost_delta": arm["mean_action_cost_per_aircraft"] - cold["mean_action_cost_per_aircraft"],
                "mean_batch_decision_ms_delta": (arm["mean_decision_ms_per_batch_across_families"]
                                                 - cold["mean_decision_ms_per_batch_across_families"]),
            }
        warm_cold_pairs[panel] = comparisons
    gap_pairs = {}
    hold_arms = communication_metrics.get("cadence-hold", {})
    for label, mpc in communication_metrics.get("cadence-mpc", {}).items():
        hold = hold_arms.get(label)
        if hold is None:
            continue
        _require(hold["configuration"] == mpc["configuration"], f"Unmatched MPC and Hold arms for {label}")
        gap_pairs[label] = {
            "configuration": mpc["configuration"],
            "per_seed_mpc_minus_hold": _paired_outcomes(hold, mpc),
            "mean_action_cost_delta": mpc["mean_action_cost_per_aircraft"] - hold["mean_action_cost_per_aircraft"],
            "mean_batch_decision_ms_delta": (mpc["mean_decision_ms_per_batch_across_families"]
                                             - hold["mean_decision_ms_per_batch_across_families"]),
        }
    if manifest:
        results["exp5"] = communication_progress

    ablations = trial / "exp6/summary.json"
    if ablations.is_file():
        report = read(ablations)
        _require((report["checkpoint_sha256"], report["acceptance_sha256"]) == (digest, acceptance_digest),
                 "Unexpected Exp 6 provenance")
        one_step = base / "one-step"
        control_config = one_step / "reference-config.json" if deployed_mpc else Path(inputs[2])
        matched = _verify_one_step_control(argparse.Namespace(
            checkpoint=checkpoint, control_config=control_config,
            one_step_checkpoint=one_step / "selected-one-step.pt",
            one_step_config=one_step / "training/config.json",
        ))
        training = read(one_step / "summary.json")
        _require(matched == report["one_step_control"] and training["checkpoint_sha256"] == matched["sha256"]
                 and training["primary_checkpoint_sha256"] == digest
                 and training["deployment_observation_profile"] == observation
                 and all(training[key] == matched[key] for key in
                         ("control_config_sha256", "one_step_config_sha256", "updates", "decisions")),
                 "Missing matched one-step-return training evidence")
        full = report["projected_and_raw"]
        comparison = native(full["path"], FAMILIES, (*PROJECTED_AND_RAW, "candidate_1", "candidate_1_without_projection"),
                            12, sha=full["sha256"], deployment=True)
        _require(comparison["sources"]["candidate_1"]["sha256"] == matched["sha256"]
                 and comparison["sources"]["candidate_1"]["observation_profile"] == observation
                 and comparison["without_potential_edges"] is False,
                 "One-step-return control was not used in the paired ablation")
        edges = report["no_potential_edges"]
        no_edges = native(edges["path"], FAMILIES, ("baseline",), 12, sha=edges["sha256"], deployment=True)
        nonprotocol_fields = {"families", "sources", "execution_wall_seconds", "without_potential_edges"}
        _require(no_edges["without_potential_edges"] is True
                 and no_edges["sources"]["baseline"] == comparison["sources"]["baseline"]
                 and all(no_edges[key] == comparison[key] for key in no_edges.keys() - nonprotocol_fields)
                 and all(group["baseline"]["admm_k"] == 2 for group in no_edges["families"].values())
                 and all(group[arm]["admm_k"] == 2 for group in comparison["families"].values()
                         for arm in ("baseline", "candidate_1")),
                 "Potential-edge ablation changed more than adjacency")
        substitution = report["kss_substitution"]
        figure = read(substitution["path"], substitution["sha256"])
        _require(figure["geo_checkpoint_sha256"] == digest and tuple(figure["seeds"]) == SEEDS
                 and figure["stage2_acceptance_sha256"] == acceptance_digest,
                 "Unexpected KSS substitution provenance")
        kss = _figure4(figure, ("geo", "geo-kss"))
        original = read(figures)["rows"]
        for label, row in figure["rows"].items():
            pair = f"case{row['case']}-{row['seed']}"
            reference = original[f"{pair}-geo"]
            _require(row["checkpoint_sha256"] == digest and row["manifest_sha256"] == reference["manifest_sha256"]
                     and row["departures"] == reference["departures"]
                     and read(Path(substitution["path"]).parent / f"{label}.json") == row,
                     f"Missing original Figure 4 replay conditions for {label}")
            if row["policy"] == "geo-kss":
                kss_reference = original[f"{pair}-sarl-kss"]
                _require(all(row[key] == kss_reference[key] for key in
                             ("historical_config_sha256", "historical_source_sha256")),
                         f"Changed Figure 4 KSS controller configuration for {label}")
            elif deployed_mpc:
                _require(row == reference, f"KSS control must reuse the released Geo trajectory in {label}")
        results["exp6"] = {"matched_one_step": matched, "kss_substitution": kss}

    missing = [f"exp{i}" for i in range(1, 7) if f"exp{i}" not in results]
    return dict(checkpoint_sha256=digest, acceptance_sha256=acceptance_digest, complete=not missing,
                pending=missing, completed=results, verified_exp5_arms=communication_progress,
                exp5_outcomes=communication_metrics, exp5_warm_cold_pairs=warm_cold_pairs,
                exp5_gap_policy_pairs=gap_pairs,
                exp5_network_pairs=_network_pairs(communication_metrics.get("loss-delay-coloring", {}), fields),
                evidence_sha256=evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--inherited-trial", type=Path, help="Read the frozen Exp 1–5 and one-step artifacts from this run")
    args = parser.parse_args()
    print(json.dumps(audit(args.trial, inherited_trial=args.inherited_trial), indent=2, allow_nan=False))
