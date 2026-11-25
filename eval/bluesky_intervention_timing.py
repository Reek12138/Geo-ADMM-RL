"""Isolate the first command of a verified native avoidance sequence."""

import argparse
from collections import Counter
import json
from pathlib import Path

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
from bluesky.tools.aero import ft, kts
import numpy as np
import torch

from envs.bluesky_plan_a import GeoNativeBlueSky
from envs.bluesky_safety import lookahead_risk, protected_command
from eval.bluesky_pair import _checksum, _load
from train.bluesky_td3 import _shaping
from train.td3 import encode_action


OUTCOMES = ("arrivals", "collisions", "exits", "timeouts")


def planned_commands(nominal, moves, slots, config):
    controls = nominal.copy()
    present = {int(slot): i for i, slot in enumerate(slots)}
    for move in moves:
        slot = move["slot"]
        if slot not in present:
            continue
        i = present[slot]
        if move["speed_delta_kts"]:
            speed = (move["target_cas_kts"] - bs.traf.cas[i] / kts) / config.max_speed_delta_kts
            controls[0, slot, 0] = np.clip(speed, -1, 1)
        if move["layer_delta"]:
            height = (move["target_alt_ft"] - bs.traf.alt[i] / ft) / config.max_alt_delta_ft
            controls[0, slot, 1] = np.clip(height, -1, 1)
            if move["speed_delta_kts"] and abs(controls[0, slot, 1]) > .01:
                controls[0, slot, 0] = 0
    return controls


def feedback_commands(actor, graph, physical, capture, *, joint_layer):
    policy = actor(*graph)
    proposal = encode_action(policy)
    tuning = policy.consensus_inputs
    commands, _ = protected_command(
        proposal, physical, graph[2], graph[-1], tuning, capture.get("rounds", 2), 0,
        directional_heading=True, extra_rounds=2, batched_layer_search=True,
        batched_layer_margin=capture.get("margin", 0.0), joint_layer_search=joint_layer,
        joint_layer_reserve=10.0 if joint_layer else 0.0,
    )
    return proposal, commands.masked_fill(~graph[-1].unsqueeze(-1), 0)


def feedback_reward(settings, graph, future, physical, proposal, controls, reward, done):
    active = graph[-1]
    terminal = ~future[-1] | torch.from_numpy(done).unsqueeze(-1)
    correction = (controls - proposal[..., 4:]).square().sum(dim=-1).masked_fill(~active, 0)
    reward = torch.from_numpy(reward).float() - .05 * correction
    if settings["shaped_reward"]:
        reward = reward + _shaping(graph, future, proposal, terminal)
    if settings["lookahead_weight"]:
        risk = lookahead_risk(physical, future[-1], dense=settings["dense_risk"],
                              overlap=settings["overlap_risk"]).masked_fill(terminal, 0)
        reward = reward - settings["lookahead_weight"] * risk
    return {
        "active": active[0].tolist(), "terminal": terminal[0].tolist(),
        "reward": reward.masked_fill(~active, 0)[0].tolist(),
    }


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    reference = json.loads(args.reference.read_text())
    plan = json.loads(args.plan.read_text())
    digest = _checksum(args.capture)
    state = torch.load(args.capture, map_location="cpu", weights_only=True)
    prefix = plan["intervention_decision"]
    if (not plan["feasible"] or plan["capture_sha256"] != digest
            or reference["step_seconds"] != .05 or reference["exact_prefix_decisions"] != prefix
            or reference["plan"] != plan["changed"]
            or any(reference[key] != plan[key] or plan[key] != state[key] for key in ("family", "seed"))):
        raise ValueError("The plan and replay do not describe the same verified native intervention")
    feedback = args.feedback_checkpoint is not None
    actor = None
    graph_options = {}
    feedback_sha256 = None
    settings = None
    if feedback:
        feedback_sha256 = _checksum(args.feedback_checkpoint)
        if feedback_sha256 != state["checkpoint_sha256"]:
            raise ValueError("Closed-loop control needs the source actor used in the trajectory")
        actor = _load(args.feedback_checkpoint, torch.device("cpu"))
        graph_options = vars(actor.observation_profile)
        settings = json.loads(args.reward_config.read_text())
        if state["family"] not in ("crossing3", "crossing4", "crossing6"):
            raise ValueError("Closed-loop reward auditing only covers crossing traffic")
        if settings["control_dt"] != 6 or settings["unprotected_rollout_fraction"]:
            raise ValueError("Expected the protected six-second training configuration")
    args.workdir.mkdir()
    if args.navdata:
        cache = args.workdir / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(args.navdata)
    records = state["records"]
    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=records[0]["commands"].shape[1],
                                 sim_dt=.05, control_dt=6, hdg_deadband_deg=.25, seed=state["seed"])
    env = GeoNativeBlueSky(config, workdir=args.workdir, family=state["family"], **graph_options)
    graph = env.reset()
    enable_rl_fast_mode(False, simdt=.05)
    start = prefix + (args.mode == "skip-first")
    window = 1 if args.mode == "first-only" else args.intervention_steps
    stop = min(start + window, config.max_steps) if window is not None else config.max_steps
    totals = Counter(dict.fromkeys(OUTCOMES, 0))
    contacts = []
    modified = []
    rewards = []
    empty = np.zeros_like(records[0]["commands"].numpy())
    for decision in range(config.max_steps):
        nominal = records[decision]["commands"].numpy() if decision < len(records) else empty
        if decision == start:
            archived = records[decision]["before"]
            np.testing.assert_array_equal(env.ac_slot, archived["active"][0].nonzero().flatten().numpy())
            for field in ("lat", "lon", "alt", "cas", "hdg"):
                np.testing.assert_allclose(getattr(bs.traf, field), archived[field][0, env.ac_slot].numpy(),
                                           rtol=0, atol=1e-9)
        ordinary = nominal
        if feedback:
            joint = args.feedback_joint_layer and decision >= prefix
            proposal, policy_commands = feedback_commands(actor, graph, env.physical, state, joint_layer=joint)
            ordinary = policy_commands.numpy()
            if decision < prefix or (args.mode == "baseline" and not joint):
                np.testing.assert_allclose(ordinary, nominal, atol=1e-6, rtol=1e-6)
        apply_plan = args.mode != "baseline" and start <= decision < stop
        controls = planned_commands(ordinary, plan["changed"], env.ac_slot, config) if apply_plan else ordinary
        changed = np.flatnonzero(np.any(np.abs(controls - ordinary)[0] > 1e-5, axis=-1))
        if changed.size:
            modified.append(dict(decision=decision + 1, slots=changed.tolist()))
        future, reward, done, info = env.step_native_audited(controls, track_separation=False, track_contacts=True)
        counts = {key: int(info[key][0]) for key in OUTCOMES}
        if decision < start and counts != records[decision]["counts"]:
            raise ValueError(f"Native outcome diverged during the shared prefix at step {decision + 1}")
        totals.update(counts)
        contacts.extend(dict(decision=decision + 1, **contact) for contact in info["contacts"])
        if feedback and decision >= prefix:
            rewards.append(dict(decision=decision + 1, **feedback_reward(
                settings, graph, future, env.physical, proposal, torch.from_numpy(controls), reward, done
            )))
        graph = future
        if done[0]:
            break
    if (args.mode != "baseline" and not modified) or (args.mode == "first-only" and len(modified) != 1):
        raise ValueError("No verified intervention was applied")
    if not feedback and args.mode == "full" and dict(totals) != reference["intervention"]:
        raise ValueError("The full native intervention no longer reproduces its recorded outcome")
    if args.mode == "baseline" and not args.feedback_joint_layer and dict(totals) != reference["original"]:
        raise ValueError("Feedback without intervention must reproduce the original native outcomes")
    output = dict(mode=args.mode, intervention_steps=window, family=state["family"], seed=state["seed"],
                  sim_dt=.05, control_dt=6,
                  prefix_decisions=start, command_changes=modified, capture_sha256=digest, plan_sha256=_checksum(args.plan),
                  original=reference["original"], full_intervention=reference["intervention"], outcome=totals,
                  contacts=contacts)
    if feedback:
        output.update(feedback_checkpoint_sha256=feedback_sha256, feedback_joint_layer=args.feedback_joint_layer,
                      reward_config_sha256=_checksum(args.reward_config), rewards=rewards)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(args.mode, plan["requested_lead_s"], dict(totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "plan", "reference", "output", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--mode", choices=("first-only", "skip-first", "full", "bounded", "baseline"), required=True)
    parser.add_argument("--intervention-steps", type=int)
    parser.add_argument("--feedback-checkpoint", type=Path)
    parser.add_argument("--feedback-joint-layer", action="store_true")
    parser.add_argument("--reward-config", type=Path)
    options = parser.parse_args()
    if options.mode == "bounded":
        if options.intervention_steps is None or options.intervention_steps < 2:
            parser.error("A bounded intervention needs at least two steps")
    elif options.intervention_steps is not None:
        parser.error("Intervention steps are only used in bounded mode")
    if bool(options.feedback_checkpoint) != bool(options.reward_config):
        parser.error("Feedback and reward configuration must be provided together")
    if (options.mode == "baseline" or options.feedback_joint_layer) and not options.feedback_checkpoint:
        parser.error("This option requires a feedback checkpoint")
    run(options)
