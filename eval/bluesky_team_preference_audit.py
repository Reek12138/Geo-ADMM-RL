"""Audit the transfer of native paired preferences across traffic seeds."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from envs.observations import LAYER
from eval.bluesky_actor_gradient_audit import _state
from eval.bluesky_pair import _checksum, _load
from models.graphformer import TwinCritic
from train.td3 import encode_action, encode_executed_action


OUTCOMES = ("arrivals", "collisions", "exits", "timeouts")


@dataclass
class Pair:
    name: str
    family: str
    seed: int
    native: tuple
    training: tuple
    commands: tuple
    original: dict
    alternative: dict
    sources: tuple
    decision: int
    proposal: torch.Tensor | None = None
    control: bool = False

    @property
    def group(self):
        return f"{self.family}/{self.seed}"


def read(path):
    return json.loads(path.read_text())


def read_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate(pair):
    original, alternative = pair.original, pair.alternative
    if sum(original.values()) != sum(alternative.values()):
        raise ValueError(f"{pair.name} does not cover the same number of aircraft")
    if pair.control:
        if original != alternative:
            raise ValueError(f"{pair.name} changed aggregate outcomes")
    elif not (alternative["arrivals"] > original["arrivals"]
              and all(alternative[key] <= original[key] for key in OUTCOMES[1:])):
        raise ValueError(f"{pair.name} lacks a trajectory-level improvement")
    changed = (pair.commands[1] - pair.commands[0]).abs().amax(dim=-1) > 1e-5
    if not changed.any() or (changed & ~pair.native[-1]).any():
        raise ValueError(f"{pair.name} has invalid changes to active aircraft")


def cross_pairs(source, crossing_four, crossing_six, profile):
    paths = {"crossing4": crossing_four, "crossing6": crossing_six}
    plans = {"crossing4": "crossing4-120-sequence.json", "crossing6": "crossing6-60-plan.json"}
    traces = {}
    for family, path in paths.items():
        if _checksum(path) != read(source / plans[family])["capture_sha256"]:
            raise ValueError(f"{family} capture differs from the native experiment")
        traces[family] = read_tensor(path)
    command_table = source / "first-action-risk.json"
    for name, record in read(command_table).items():
        outcome_file = source / f"{name}.json"
        outcome = read(outcome_file)
        family, seed = outcome["family"], outcome["seed"]
        trace = traces[family]
        decision = outcome["exact_prefix_decisions"]
        if outcome["step_seconds"] != .05 or seed != trace["seed"] or decision < 1:
            raise ValueError(f"{name} has invalid replay provenance")
        row = trace["records"][decision]
        original = row["commands"]
        slots = [int(move["slot"]) for move in outcome["plan"]]
        torch.testing.assert_close(original[0, slots], torch.tensor(record["nominal_controls"]))
        alternative = original.clone()
        alternative[0, slots] = torch.tensor(record["nominal_first_intervention"])
        native, _ = _state(trace, decision, profile, matching_plan_b=False)
        training, _ = _state(trace, decision, profile, matching_plan_b=True)
        yield Pair(name, family, seed, native, training, (original, alternative), outcome["original"],
                   outcome["intervention"], (outcome_file, command_table, paths[family]), decision + 1)


def joint_layer_pairs(root, captures, profile):
    manifest = root / "summary.json"
    references = read(manifest)
    for path in captures:
        reference = references[path.stem]
        if _checksum(path) != reference["source_sha256"]:
            raise ValueError(f"{path.name} differs from the native replay reference")
        trace = read_tensor(path)
        family, seed = trace["family"], trace["seed"]
        outcome_path = root / f"native-{family}-{seed}-joint.json"
        output = read(outcome_path)
        decision = output["original_trajectory_verified_until"]
        if (output["audited_dt"] != .05 or output["checkpoint_sha256"] != trace["checkpoint_sha256"]
                or (output["family"], output["seed"]) != (family, seed)
                or decision is None or output["repairs"][0]["decision"] != decision + 1):
            raise ValueError(f"{path.name} has an unverified native prefix")
        repaired = output["repairs"][0]
        expected = reference["revised"][0]
        if (expected["decision"], expected["initial_pairs"], expected["remaining_pairs"]) != (
            repaired["decision"], repaired["before"], repaired["after"]
        ):
            raise ValueError(f"{path.name} rebuilt repair differs from the native intervention")
        altitude_path = root / f"{path.stem}-altitude.pt"
        original = trace["records"][decision]["commands"]
        alternative = original.clone()
        alternative[..., 1] = read_tensor(altitude_path)[decision]
        slots = (alternative[..., 1] - original[..., 1]).abs().gt(1e-5).nonzero()[:, 1].tolist()
        if slots != repaired["slots"] or slots != expected["aircraft_slots"]:
            raise ValueError(f"{path.name} rebuilt commands change different aircraft")
        native, _ = _state(trace, decision, profile, matching_plan_b=False)
        training, _ = _state(trace, decision, profile, matching_plan_b=True)
        outcomes = {key: sum(row["counts"][key] for row in trace["records"]) for key in OUTCOMES}
        yield Pair(f"{family}-{seed}-joint-layer", family, seed, native, training, (original, alternative),
                   outcomes, output["outcome"], (manifest, path, outcome_path, altitude_path), decision + 1)


def snapshot_training_graph(graph, physical, profile):
    if profile.selected_altitude_observation:
        return graph
    node = graph[0].clone()
    node[..., 3] = ((physical[0][..., 2] - physical[-1]) / LAYER).masked_fill(~graph[-1], 0)
    return (node, *graph[1:])


def yaw_pairs(root, profile):
    outcomes = dict(arrivals="arrival", collisions="collision", exits="exit", timeouts="timeout")
    summary_file = root / "return-audit-summary.json"
    for name, result in read(summary_file).items():
        snapshot = root / f"{name}-snapshot" / "first-guidance-state.pt"
        control = root / f"{name}-shadow" / "trace.json"
        guided = root / f"{name}-guided" / "trace.json"
        nominal_trace, guided_trace = read(control), read(guided)
        state = read_tensor(snapshot)
        branch_identity = ("family", "seed", "sim_dt", "checkpoint_sha256")
        if (any(nominal_trace[key] != guided_trace[key] for key in branch_identity)
                or nominal_trace["sim_dt"] != .05
                or state["decision"] != result["first_guidance_decision"]):
            raise ValueError(f"{name} has invalid native guidance provenance")
        orig, alt = ({key: result["outcomes"][branch].get(kind, 0) for key, kind in outcomes.items()}
                     for branch in ("shadow", "guided"))
        native = state["graph"]
        training = snapshot_training_graph(native, state["physical"], profile)
        yield Pair(name, nominal_trace["family"], nominal_trace["seed"], native, training,
                   (state["projected"], state["guided"]), orig, alt, (summary_file, snapshot, control, guided),
                   state["decision"])


def transfer_pairs(root, profile):
    summary = root / "counterfactual-summary.json"
    for name, result in read(summary).items():
        snapshot = root / f"{name}-qstate" / "first-override-state.pt"
        plan_path = root / f"{name}-plan.json"
        state, plan = read_tensor(snapshot), read(plan_path)
        if state["decision"] < result["same_prefix_control_steps"]:
            raise ValueError(f"{name} changed commands during the matching native prefix")
        native = state["graph"]
        training = snapshot_training_graph(native, state["physical"], profile)
        yield Pair(name, plan["family"], plan["seed"], native, training,
                   (state["original_commands"], state["committed_commands"]),
                   result["branches"]["control"]["outcome"], result["branches"]["committed"]["outcome"],
                   (summary, snapshot, plan_path), state["decision"], proposal=state["proposal"], control=True)


def measure(actor, critic, pairs, view, *, step_norm, margin):
    parameters = tuple(param for model in (critic.first, critic.second) for param in model.value.parameters())
    measures, gradients = {}, {}
    positives = defaultdict(list)
    for pair in pairs:
        graph = getattr(pair, view)
        active = graph[-1]
        changed = active & ((pair.commands[1] - pair.commands[0]).abs().amax(dim=-1) > 1e-5)
        with torch.no_grad():
            proposal = pair.proposal if pair.proposal is not None else encode_action(actor(*graph))
            actions = [encode_executed_action(proposal, command) for command in pair.commands]
        reference, alternative = [critic(*graph, action) for action in actions]
        deltas = [(alt - ref)[active].mean() for ref, alt in zip(reference, alternative, strict=True)]
        min_delta = torch.minimum(*alternative) - torch.minimum(*reference)
        objective = torch.stack(deltas).mean()
        grad = torch.autograd.grad(objective, parameters)
        gradients[pair.name] = torch.cat([value.detach().flatten() for value in grad])
        measures[pair.name] = dict(critic_mean_deltas=[float(delta.detach()) for delta in deltas],
                                   combined_mean_delta=float(objective.detach()),
                                   clipped_mean_delta=float(min_delta[active].mean().detach()),
                                   clipped_changed_delta=float(min_delta[changed].sum().detach()),
                                   changed_aircraft=int(changed.sum()), active_aircraft=int(active.sum()))
        if not pair.control:
            positives[pair.group].append(pair.name)
    if len(positives) < 2:
        raise ValueError("At least two independent traffic groups are required")

    def direction(groups):
        # Equal traffic-group weight handles alternative interventions on the same flight history.
        blocks = []
        for group in groups:
            contributions = []
            for name in positives[group]:
                residual = margin - measures[name]["combined_mean_delta"]
                contributions.append(torch.sigmoid(torch.tensor(residual)) * gradients[name])
            blocks.append(torch.stack(contributions).mean(dim=0))
        combined = torch.stack(blocks).mean(dim=0)
        return step_norm * combined / combined.norm().clamp_min(1e-12)

    for pair in pairs:
        groups = [group for group in positives if group != pair.group]
        change = float(torch.dot(gradients[pair.name], direction(groups)))
        row = measures[pair.name]
        row.update(first_order_delta=change, first_order_margin=row["combined_mean_delta"] + change,
                   fit_groups=groups)
    positive_rows = [measures[pair.name] for pair in pairs if not pair.control]
    return dict(positive_pairs=len(positive_rows), independent_traffic_groups=len(positives),
                before_ranked=sum(row["combined_mean_delta"] > 0 for row in positive_rows),
                before_ranked_clipped=sum(row["clipped_mean_delta"] > 0 for row in positive_rows),
                positive_transfer=sum(row["first_order_delta"] > 0 for row in positive_rows),
                after_ranked_linearized=sum(row["first_order_margin"] > 0 for row in positive_rows),
                cases=measures)


def run(args):
    torch.set_num_threads(1)
    outputs = {}
    for checkpoint in args.checkpoints:
        actor = _load(checkpoint, torch.device("cpu")).requires_grad_(False)
        profile = actor.observation_profile
        pairs = [*cross_pairs(args.lead, args.crossing_four, args.crossing_six, profile),
                 *joint_layer_pairs(args.joint_layer, args.joint_captures, profile),
                 *yaw_pairs(args.yaw, profile), *transfer_pairs(args.transfer, profile)]
        for pair in pairs:
            validate(pair)
        state = read_tensor(checkpoint)
        critic = TwinCritic(width=state["width"], action_features=7).eval().requires_grad_(False)
        critic.load_state_dict(state["critic"])
        del state
        critic.first.value.requires_grad_(True)
        critic.second.value.requires_grad_(True)
        outputs[str(checkpoint)] = dict(
            sha256=_checksum(checkpoint),
            views={view: measure(actor, critic, pairs, view, step_norm=args.step_norm, margin=args.margin)
                   for view in ("native", "training")},
        )
    provenance = {pair.name: dict(group=pair.group, decision=pair.decision, original=pair.original,
                                  alternative=pair.alternative, no_terminal_gain=pair.control,
                                  sources=[str(path) for path in pair.sources]) for pair in pairs}
    args.output.write_text(json.dumps(dict(step_norm=args.step_norm, margin=args.margin,
                                          pairs=provenance, checkpoints=outputs), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lead", "crossing-four", "crossing-six", "joint-layer", "yaw", "transfer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--joint-captures", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--step-norm", type=float, default=.01)
    args = parser.parse_args()
    if args.margin <= 0 or args.step_norm <= 0:
        parser.error("Margin and step norm must be positive")
    run(args)
