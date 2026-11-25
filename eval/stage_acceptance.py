"""Check a frozen native BlueSky stage report against the operational gates."""

import argparse
from itertools import product
import json
from pathlib import Path
from statistics import mean

from eval.bluesky_pair import FAMILIES, _checksum


EXACT_BASELINE = "native-exact-margin6-baseline"
EXACT_REFINEMENT = "native-exact-margin6-refinement"
EXACT_MPC_REFINEMENT = "native-exact-margin6-mpc-refinement"
EXACT_LAYER_OPTIONS = dict(early_layer_search=True, exact_layer_forecast=True, batched_layer_margin=6.)
MPC_COMMUNICATION = dict(admm_communication_period=12., admm_gap_policy="mpc")
EXACT_NATIVE_PROTOCOL = dict(
    control_dt=6., first_seed=12001, num_seeds=5, sim_dt=.05, event_sampling_seconds=.05,
    directional_heading=True, dense_extra_rounds=2, batched_layer_search=True,
    joint_layer_search=True, joint_layer_reserve_m=10., batched_layer_margin_m=6.,
    early_layer_search=True, exact_layer_forecast=True, component_projection=False,
    vertical_heading=False, without_potential_edges=False, cold_start_admm=False,
    accelerated_layer_forecast=False, preserve_imminent_clearance=False,
    admm_packet_loss=None, admm_delay_steps=None, admm_schedule="jacobi",
    admm_communication_period=None, admm_gap_policy="hold",
)


def exact_safety_baseline(comparison):
    comparison = comparison.resolve()
    report = json.loads(comparison.read_text())
    expected = EXACT_NATIVE_PROTOCOL | {"detection_metres": 35000.}
    if not expected.items() <= report.items():
        raise ValueError("The comparison does not use the frozen exact safety profile")
    for family in report["families"].values():
        if family["candidate_1"]["admm_k"] != 2 or family["candidate_1"]["admm_fallback"] != 0:
            raise ValueError("The comparison uses a different ADMM budget")
    families = report["families"]
    if report["sim_dt"] != .05 or report["event_sampling_seconds"] != .05 or set(families) != set(FAMILIES):
        raise ValueError("Expected all six families at native 0.05 s sampling")
    outcomes = ("arrivals", "collisions", "exits", "unresolved")
    trial = dict.fromkeys(("aircraft", *outcomes), 0)
    for family in families.values():
        rows = family["candidate_1"]["rows"]
        if tuple(row["seed"] for row in rows) != tuple(range(12001, 12006)):
            raise ValueError("The candidate does not have all five frozen seeds")
        for row in rows:
            if row["aircraft"] != 12 or sum(row[key] for key in outcomes) != 12:
                raise ValueError("Incomplete native outcome labels")
            for key in trial:
                trial[key] += row[key]
    return dict(release_protocol=EXACT_BASELINE, formal_report=str(comparison),
                formal_report_sha256=_checksum(comparison), sources=report["sources"],
                totals={"candidate_1": trial},
                paper_evaluation_ready=trial["aircraft"] == 360 and 100 * trial["arrivals"] >= 94 * 360)


def exact_refinement_acceptance(comparison, *, mpc=False):
    comparison = comparison.resolve()
    report = json.loads(comparison.read_text())
    protocol = EXACT_NATIVE_PROTOCOL | (MPC_COMMUNICATION if mpc else {})
    if not protocol.items() <= report.items():
        raise ValueError("Refinement acceptance requires the frozen exact safety profile")
    source = report["sources"]["candidate_1"]
    if (source["observation_profile"]["detection_metres"] != 35000.
            or source.get("equivariant_actor") is not True
            or source.get("admm_warmstart_reflection") != "even"):
        raise ValueError("Refinement acceptance requires the 35 km equivariant policy")
    for family in report["families"].values():
        budget = (("baseline", 2), ("candidate_1", 2), ("candidate_2", 2),
                  ("baseline_without_projection", 0), ("candidate_1_without_projection", 0),
                  ("candidate_2_without_projection", 0))
        for arm, rounds in budget:
            result = family[arm]
            if result["admm_k"] != rounds or result["admm_fallback"] != 0:
                raise ValueError(f"Unexpected ADMM budget for {arm}")
    result = assess(report, guarded=True, native_protocol=protocol)
    release = EXACT_MPC_REFINEMENT if mpc else EXACT_REFINEMENT
    return result | {"release_protocol": release, "formal_report": str(comparison),
                     "formal_report_sha256": _checksum(comparison)}


def paper_layer_options(acceptance):
    if json.loads(acceptance.read_text()).get("release_protocol") in (
            EXACT_BASELINE, EXACT_REFINEMENT, EXACT_MPC_REFINEMENT):
        return EXACT_LAYER_OPTIONS.copy()
    return {}


def paper_layer_flags(acceptance):
    options = paper_layer_options(acceptance)
    return ["--early-layer-search", "--exact-layer-forecast", "--batched-layer-margin",
            str(options["batched_layer_margin"])] if options else []


def paper_uses_mpc(acceptance):
    return json.loads(acceptance.read_text()).get("release_protocol") == EXACT_MPC_REFINEMENT


def mpc_communication_flags():
    return ["--admm-communication-period", str(MPC_COMMUNICATION["admm_communication_period"]),
            "--admm-gap-policy", MPC_COMMUNICATION["admm_gap_policy"]]


def paper_checkpoint_digest(acceptance, checkpoint):
    result = json.loads(acceptance.read_text())
    protocol = result.get("release_protocol")
    if protocol in (EXACT_BASELINE, EXACT_REFINEMENT, EXACT_MPC_REFINEMENT):
        report = Path(result["formal_report"])
        if protocol == EXACT_BASELINE:
            verified = exact_safety_baseline(report)
        else:
            verified = exact_refinement_acceptance(report, mpc=protocol == EXACT_MPC_REFINEMENT)
        digest = verified["sources"]["candidate_1"]["sha256"]
        if verified != result or not verified["paper_evaluation_ready"] or digest != _checksum(checkpoint):
            raise ValueError("The frozen native exact-safety checkpoint did not pass its release gate")
        return digest
    if not result.get("paper_evaluation_ready"):
        raise ValueError("The checkpoint has not passed the Stage 2 paper-evaluation gate")
    source = result["sources"]["candidate_1"]
    totals = result["totals"]["candidate_1"]
    if (totals["aircraft"] != 360
            or 100 * totals["arrivals"] <= 95 * totals["aircraft"]
            or 100 * totals["collisions"] > 3 * totals["aircraft"]
            or source["sha256"] != _checksum(checkpoint)):
        raise ValueError("The checkpoint has not passed the Stage 2 paper-evaluation gate")
    return source["sha256"]


def _paired_direction(values):
    observed = sum(values)
    count = sum(sum(delta * sign for delta, sign in zip(values, signs, strict=True)) >= observed
                for signs in product((-1, 1), repeat=len(values)))
    return count / (2 ** len(values))


def assess(report, *, guarded=False, native_protocol=None):
    if native_protocol is not None and (not guarded or not native_protocol.items() <= report.items()):
        raise ValueError("Refinement acceptance requires the frozen exact safety profile")
    if guarded and native_protocol is None:
        if (not report.get("batched_layer_search") or not report.get("joint_layer_search")
                or report.get("without_potential_edges", False)
                or report.get("joint_layer_reserve_m") != 10.0 or not report.get("directional_heading")
                or report.get("batched_layer_margin_m") or report.get("preserve_imminent_clearance")
                or report.get("early_layer_search") or report.get("exact_layer_forecast")
                or report.get("accelerated_layer_forecast") or report.get("cold_start_admm")
                or report.get("admm_packet_loss") is not None or report.get("admm_schedule", "jacobi") != "jacobi"
                or report.get("admm_communication_period") is not None or report.get("admm_delay_steps") is not None
                or report.get("admm_gap_policy", "hold") != "hold"):
            raise ValueError("Guarded acceptance requires the validated three-aircraft safety profile")
    elif not guarded and report.get("diagnostic_projection", False):
        raise ValueError("Stage acceptance requires the standard projection protocol")
    families = report["families"]
    seeds = tuple(range(12001, 12006))
    required = ("baseline", "candidate_1", "candidate_1_without_projection")
    if guarded:
        required += ("baseline_without_projection", "candidate_2", "candidate_2_without_projection")
    if report["sim_dt"] != .05 or report["event_sampling_seconds"] != .05 or set(families) != set(FAMILIES):
        raise ValueError("Stage acceptance needs all six families at 0.05 s native sampling")
    for groups in families.values():
        for name in required:
            rows = groups[name]["rows"]
            if tuple(r["seed"] for r in rows) != seeds or any(r["aircraft"] != 12 for r in rows):
                raise ValueError("Expected the five frozen flow seeds and twelve aircraft per flow")
            for row in rows:
                if sum(row[k] for k in ("arrivals", "collisions", "exits", "unresolved")) != row["aircraft"]:
                    raise ValueError("Incomplete outcome labels")

    def count(group, metric, scenarios=FAMILIES):
        return sum(row[metric] for family in scenarios for row in families[family][group]["rows"])

    totals = {g: {k: count(g, k) for k in ("aircraft", "arrivals", "collisions", "exits", "unresolved")}
              for g in required}
    trial = totals["candidate_1"]
    total = trial["aircraft"]
    yaw_families = ("yaw_plus15", "yaw_minus15")
    yaw = {f: count("candidate_1", "arrivals", (f,)) for f in yaw_families}
    head_on = count("candidate_1", "arrivals", ("head_on",))
    gates = {
        "arrival_over_90_percent": trial["arrivals"] / total > .90,
        "collision_under_3_percent": trial["collisions"] / total < .03,
        "both_yaw_over_95_percent": all(x / 60 > .95 for x in yaw.values()),
        "head_on_at_least_58_of_60": head_on >= 58,
        "projection_reduces_collisions": trial["collisions"] < totals["candidate_1_without_projection"]["collisions"],
        "projection_keeps_each_yaw_arrival": all(yaw[f] >= count("candidate_1_without_projection", "arrivals", (f,))
                                                for f in yaw_families),
    }
    comparison = None
    comparator = "candidate_2_without_projection"
    if all(comparator in results for results in families.values()):
        differences = []
        raw = "candidate_1_without_projection"
        for index, seed in enumerate(seeds):
            gain = 0
            for groups in families.values():
                previous = groups[comparator]["rows"][index]
                if previous["seed"] != seed or previous["aircraft"] != 12:
                    raise ValueError("Unpaired ablation comparator")
                gain += previous["collisions"] - groups[raw]["rows"][index]["collisions"]
            differences.append(gain)
        comparison = {"collision_reduction_per_seed": differences, "one_sided_p": _paired_direction(differences),
                      "arrival_gain": count(raw, "arrivals") - count(comparator, "arrivals")}
        gates["raw_improves_on_previous_policy"] = comparison["one_sided_p"] < 0.05 and comparison["arrival_gain"] >= 0
    else:
        gates["raw_improves_on_previous_policy"] = None
    if guarded:
        gates.pop("arrival_over_90_percent")
        gates.pop("collision_under_3_percent")
        gates.pop("both_yaw_over_95_percent")
        gates.pop("head_on_at_least_58_of_60")
        gates.update(arrival_at_least_95_percent=trial["arrivals"] / total >= .95,
                     collision_at_most_3_percent=trial["collisions"] / total <= .03,
                     both_yaw_at_least_95_percent=all(x / 60 >= .95 for x in yaw.values()),
                     head_on_at_least_98_percent=head_on / 60 >= .98)
        paired = {}
        for name, trial_group, baseline_group in (
                ("projected", "candidate_1", "baseline"),
                ("raw", "candidate_1_without_projection", "baseline_without_projection")):
            arrival_gain, collision_gain = [], []
            for index in range(len(seeds)):
                rows = [(groups[trial_group]["rows"][index], groups[baseline_group]["rows"][index])
                        for groups in families.values()]
                arrival_gain.append(sum(after["arrivals"] - before["arrivals"] for after, before in rows))
                collision_gain.append(sum(before["collisions"] - after["collisions"] for after, before in rows))
            paired[name] = dict(arrival_gain=arrival_gain, collision_reduction=collision_gain,
                                arrival_p=_paired_direction(arrival_gain), collision_p=_paired_direction(collision_gain))
            gates[f"{name}_improves_on_frozen"] = (
                sum(arrival_gain) >= 0 and sum(collision_gain) >= 0
                and min(paired[name]["arrival_p"], paired[name]["collision_p"]) < .05)
        comparison = {"against_incoming_raw": comparison, "against_frozen": paired}
    sprint = trial["arrivals"] / total
    result = {"gates": gates, "passed": all(gates.values()),
            "sprint_over_97_percent": sprint >= .97 if guarded else sprint > .97,
            "totals": totals, "yaw_arrivals": yaw, "head_on_arrivals": head_on, "raw_comparison": comparison}
    if guarded:
        result["paper_evaluation_ready"] = (100 * trial["arrivals"] > 95 * total
                                            and 100 * trial["collisions"] <= 3 * total)
        result["sprint_target_met"] = (100 * trial["arrivals"] >= 97 * total
                                       and 100 * trial["collisions"] <= 2 * total
                                       and min(yaw.values()) >= 57)
        result["sources"] = report["sources"]
        result["families"] = {
            family: {
                arm: {
                    **{key: count(arm, key, (family,)) for key in totals[arm]},
                    "mean_action_cost_per_aircraft": mean(row["action_cost_per_aircraft"]
                                                         for row in groups[arm]["rows"]),
                } for arm in required
            } for family, groups in families.items()
        }
    return result


def assess_guarded(report):
    return assess(report, guarded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--exact-safety-baseline", action="store_true")
    protocol.add_argument("--exact-refinement", action="store_true")
    protocol.add_argument("--exact-mpc-refinement", action="store_true")
    protocol.add_argument("--guarded", action="store_true",
                        help="Apply the Stage 2 three-aircraft safety and paper-evaluation gates")
    args = parser.parse_args()
    if args.exact_safety_baseline:
        result = exact_safety_baseline(args.comparison)
    elif args.exact_refinement or args.exact_mpc_refinement:
        result = exact_refinement_acceptance(args.comparison, mpc=args.exact_mpc_refinement)
    else:
        result = assess(json.loads(args.comparison.read_text()), guarded=args.guarded)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
