"""Attribute actor updates on a set of replayable native avoidance interventions."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from envs.bluesky_graph import traffic_graph
from envs.bluesky_safety import lookahead_risk, protected_command
from eval.bluesky_pair import _checksum, _load
from models.graphformer import Critic
from train.td3 import _active_mean, _anchor_loss, _navigation_loss, _projection_loss, encode_action


def _state(trace, decision, profile, *, matching_plan_b):
    state = trace["records"][decision]["before"]
    route_altitude = trace["records"][0]["before"]["alt"]
    planned = state["sel_alt"] if matching_plan_b and not profile.selected_altitude_observation else route_altitude
    return traffic_graph(
        latitude=state["lat"], longitude=state["lon"], altitude=state["alt"], heading=state["hdg"],
        east=state["gseast"], north=state["gsnorth"], vertical=state["vs"], goal_latitude=state["goal_lat"],
        goal_longitude=state["goal_lon"], planned_altitude=planned, lnav=state["swlnav"], active=state["active"],
        center_latitude=trace["centre"][0], center_longitude=trace["centre"][1], airspace_half_km=trace["area_km"],
        cas=state["cas"], tas=state["tas"], selected_altitude=state["sel_alt"],
        geodesic_navigation=profile.geodesic_navigation,
        route_origin=state["origins"].unbind(-1) if profile.cross_track_observation else None,
        far_conflict_observation=profile.far_conflict_observation,
        selected_altitude_observation=profile.selected_altitude_observation,
    )


def _gradient(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(grad if grad is not None else torch.zeros_like(param)).flatten()
                      for grad, param in zip(grads, parameters, strict=True)])


def _measure(actor, critic, anchor, graph, physical, desired, changed, config, draws):
    active = graph[-1]
    parameters = tuple(actor.parameters())
    with torch.no_grad():
        reference = anchor(*graph)
    observations = []
    for draw in range(draws):
        torch.manual_seed(19261 + draw)
        policy = actor(*graph, temperature=.35)
        action = encode_action(policy)
        components = {
            "critic": -_active_mean(critic(*graph, action), active),
            "lookahead_risk": config["raw_risk_weight"] * _active_mean(
                lookahead_risk(physical, active, action[..., 4:], dense=config["dense_risk"],
                               overlap=config["overlap_risk"]), active),
            "navigation": config["navigation_weight"] * _navigation_loss(
                policy, graph, physical, config["safe_navigation_weight"]),
            "anchor": config["anchor_weight"] * _anchor_loss(policy, reference, graph),
        }
        if config["projection_weight"]:
            with torch.no_grad():
                consensus = policy.consensus_inputs
                delivered, feasible = protected_command(
                    action, physical, graph[2], active, consensus, config["admm_k"], config["admm_fallback"],
                    by_component=config["component_projection"], vertical_heading=config["vertical_heading"],
                    directional_heading=config["directional_heading"], extra_rounds=config["dense_extra_rounds"],
                    batched_layer_search=config["batched_layer_search"],
                    joint_layer_search=config["joint_layer_search"], joint_layer_reserve=config["joint_layer_reserve"],
                )
                taught = active & feasible & ((delivered - action[..., 4:]).square().sum(dim=-1) > 1e-4)
            components["projection"] = config["projection_weight"] * _projection_loss(policy, delivered, taught)
        components["combined"] = sum(components.values())
        teacher = _gradient(_projection_loss(policy, desired, changed), parameters)
        teacher_norm = teacher.norm()
        if teacher_norm <= 0 or not torch.isfinite(teacher_norm):
            raise ValueError("Expected a finite nonzero gradient toward the native avoidance command")
        descent = {}
        for name, loss in components.items():
            grad = _gradient(loss, parameters)
            norm = grad.norm()
            projection = torch.dot(grad, teacher) / teacher_norm
            if not torch.isfinite(norm) or not torch.isfinite(projection):
                raise ValueError(f"Nonfinite evaluation gradient for {name}")
            descent[name] = dict(norm=float(norm), teacher_aligned_component=float(projection),
                                 cosine=float(projection / norm) if norm > 0 else None)
        observations.append(descent)
    summary = {}
    for name in observations[0]:
        values = [draw[name] for draw in observations]
        aligned = [draw["teacher_aligned_component"] for draw in values]
        summary[name] = dict(median_norm=median(draw["norm"] for draw in values),
                             median_teacher_aligned_component=median(aligned),
                             teacher_descent_draws=sum(component > 0 for component in aligned),
                             draws=values)
    return summary


def run(args):
    torch.set_num_threads(1)
    settings = json.loads(args.config.read_text())
    unsupported = ("collision_critic_weight", "head_on_anchor", "unprotected_rollout_fraction", "teacher_weight",
                   "expected_mode_risk", "coordination_weight")
    if any(settings.get(key) for key in unsupported):
        raise ValueError("The gradient audit requires the guarded Stage 1 loss configuration")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = _load(args.checkpoint, torch.device("cpu")).train()
    anchor_path = Path(settings["anchor"])
    anchor = _load(anchor_path, torch.device("cpu")).requires_grad_(False)
    critic = Critic(width=state["width"], action_features=7).eval().requires_grad_(False)
    critic.load_state_dict({key.removeprefix("first."): weight for key, weight in state["critic"].items()
                            if key.startswith("first.")})
    del state
    paths = {"crossing4": args.crossing_four, "crossing6": args.crossing_six}
    traces = {}
    for family, plan in (("crossing4", "crossing4-120-sequence.json"), ("crossing6", "crossing6-60-plan.json")):
        expected = json.loads((args.source / plan).read_text())["capture_sha256"]
        if _checksum(paths[family]) != expected:
            raise ValueError(f"{family} trajectory differs from the verified native replay")
        traces[family] = torch.load(paths[family], map_location="cpu", weights_only=True)
    cases = {}
    for name, actions in json.loads((args.source / "first-action-risk.json").read_text()).items():
        outcome = json.loads((args.source / f"{name}.json").read_text())
        trace = traces[outcome["family"]]
        decision = outcome["exact_prefix_decisions"]
        nominal = trace["records"][decision]["commands"]
        slots = [int(row["slot"]) for row in outcome["plan"]]
        torch.testing.assert_close(nominal[0, slots], torch.tensor(actions["nominal_controls"]))
        desired = nominal.clone()
        desired[0, slots] = torch.tensor(actions["nominal_first_intervention"])
        changed = (desired - nominal).abs().amax(dim=-1) > 1e-5
        cases[name] = dict(decision=decision + 1, original_collisions=outcome["original"]["collisions"],
                           intervention_collisions=outcome["intervention"]["collisions"])
        for label, matching in (("native", False), ("training_observation", True)):
            graph, physical = _state(trace, decision, actor.observation_profile, matching_plan_b=matching)
            cases[name][label] = _measure(actor, critic, anchor, graph, physical, desired,
                                          changed & graph[-1], settings, args.draws)
    args.output.write_text(json.dumps(dict(
        checkpoint_sha256=_checksum(args.checkpoint), anchor_sha256=_checksum(anchor_path),
        config_sha256=_checksum(args.config), draws=args.draws, cases=cases,
    ), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "source", "crossing-four", "crossing-six", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=8)
    options = parser.parse_args()
    if options.draws < 1:
        parser.error("--draws must be positive")
    run(options)
