"""Trace per-aircraft return-to-route decisions through a native yaw scenario."""

import argparse
import json
from math import isfinite
from pathlib import Path

import torch

from envs.bluesky_safety import CAS_DELTA, PhysicalState, _directional_heading, protected_command
from eval.bluesky_pair import _checksum, _load
from train.td3 import encode_action


def _linked_return(graph, physical, commands, course_error, deadband, enabled):
    """Check training labels and optionally add only guarded turns in this replay."""
    active = graph[-1]
    adjacency = graph[2]
    linked = adjacency.any(dim=-1)
    needs_turn = (graph[0][..., 8] < .5) & (course_error.abs() > deadband)
    target = (course_error / 5.0).clamp(-1, 1).masked_fill(~(active & linked & needs_turn), 0.0)
    state = PhysicalState(*physical)
    labels = _directional_heading(state, state.ground_speed, target, torch.zeros_like(target), adjacency, active)
    label_available = labels.abs() > .05
    current = commands[..., 2]
    eligible = enabled & label_available & (current.abs() < deadband / 5.0)
    proposal = torch.where(eligible, labels, current)
    ground_delta = CAS_DELTA * (state.tas / state.cas.clamp_min(1)).clamp(.8, 2.0)
    speed = state.ground_speed + ground_delta * commands[..., 0]
    checked = _directional_heading(state, speed, proposal, commands[..., 1], adjacency, active)
    changed = eligible & (checked.abs() >= deadband / 5.0)
    additions = torch.where(changed, checked, current)
    final = _directional_heading(state, speed, additions, commands[..., 1], adjacency, active)
    preserves = (final == additions).all(dim=-1, keepdim=True)
    return label_available, torch.where(preserves, additions, current)


@torch.inference_mode()
def run(args):
    import bluesky as bs
    from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
    from envs.bluesky_graph import cross_track_km
    from envs.bluesky_plan_a import GeoNativeBlueSky

    if args.without_projection and args.apply_linked_return:
        raise ValueError("The counterfactual requires the guarded projection")
    if args.navigation_reference and args.apply_linked_return:
        raise ValueError("Run the learned reference and geometry guidance as separate counterfactuals")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    actor = _load(args.checkpoint, torch.device("cpu"))
    profile = actor.observation_profile
    reference = None
    if args.navigation_reference:
        reference = _load(args.navigation_reference, torch.device("cpu"))
        if reference.observation_profile != profile:
            raise ValueError("The reference must use the same observation profile")
    return_audit = None
    if args.return_audit_config:
        from eval.bluesky_yaw_return_audit import YawReturnAudit

        return_audit = YawReturnAudit(args.checkpoint, args.return_audit_config)
    first_guidance = None
    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=12, sim_dt=.05, control_dt=6,
                                  hdg_deadband_deg=.25, seed=args.seed)
    if not isfinite(args.minimum_cross_track_km) or not 0 <= args.minimum_cross_track_km < config.max_xtk_km:
        raise ValueError("The recovery threshold must be inside the native cross-track boundary")
    env = GeoNativeBlueSky(config, workdir=args.output, family=args.family,
                           detection_metres=profile.detection_metres, geodesic_navigation=profile.geodesic_navigation,
                           cross_track_observation=profile.cross_track_observation,
                           far_conflict_observation=profile.far_conflict_observation,
                           selected_altitude_observation=profile.selected_altitude_observation)
    graph = env.reset()
    enable_rl_fast_mode(False, simdt=.05)
    aircraft = [{"slot": slot, "outcome": "unresolved", "decisions": []} for slot in range(config.aircraft_per_env)]
    for decision in range(1, config.max_steps + 1):
        previous = graph
        target_values = return_audit.values(graph, env.physical, decision) if return_audit is not None else None
        policy = actor(*graph)
        raw = encode_action(policy)
        proposal = raw
        observation = graph[0][0]
        course_error = torch.atan2(observation[:, 5], observation[:, 6]).rad2deg()
        if reference is not None:
            learned_turn = encode_action(reference(*graph))
            deadband = config.hdg_deadband_deg / config.max_hdg_delta_deg
            missing = raw[..., 6].abs() < deadband
            recovery = ((learned_turn[..., 6].abs() >= deadband) & (learned_turn[..., 6] * course_error > 0)
                        & (course_error.abs() > 1.0) & (observation[:, 8] < .5) & graph[-1] & missing)
            proposal = torch.where(recovery.unsqueeze(-1), learned_turn, raw)
        commands = proposal[..., 4:]
        if not args.without_projection:
            tuning = policy.consensus_inputs
            commands, _ = protected_command(proposal, env.physical, graph[2], graph[-1], tuning, 2, 0,
                                             directional_heading=True, extra_rounds=2, batched_layer_search=True,
                                             joint_layer_search=True, joint_layer_reserve=10.0)
        projected_commands = commands
        active = graph[-1][0]
        xtk = cross_track_km(*(torch.from_numpy(value) for value in (
            bs.traf.lat, bs.traf.lon, env.ac_goal_lat, env.ac_goal_lon, env.ac_orig_lat, env.ac_orig_lon)))
        if args.audit_linked_return or args.apply_linked_return:
            enabled = torch.zeros_like(graph[-1])
            enabled[0, env.ac_slot] = xtk.abs() >= args.minimum_cross_track_km
            labels, guided = _linked_return(graph, env.physical, commands, course_error.unsqueeze(0),
                                            config.hdg_deadband_deg, enabled)
            if args.apply_linked_return:
                commands = torch.cat((commands[..., :2], guided.unsqueeze(-1)), dim=-1)
                if return_audit is not None and first_guidance is None and (commands != projected_commands).any():
                    first_guidance = return_audit.first_action_comparison(
                        graph, env.physical, proposal, projected_commands, commands, decision,
                    )
                    torch.save(dict(decision=decision, graph=graph, physical=env.physical,
                                    projected=projected_commands, guided=commands),
                               args.output / "first-guidance-state.pt")
        for index, slot in enumerate(env.ac_slot.tolist()):
            row = {
                "decision": decision, "error_deg": float(course_error[slot]), "cross_track_km": float(xtk[index]),
                "degree": int(graph[2][0, slot].sum()), "mode": int(policy.mode[0, slot].argmax()),
                "raw_turn_deg": float(raw[0, slot, 6]) * config.max_hdg_delta_deg,
                "delivered_turn_deg": float(commands[0, slot, 2]) * config.max_hdg_delta_deg,
            }
            if reference is not None:
                row["reference_turn_used"] = bool(recovery[0, slot])
            if args.audit_linked_return or args.apply_linked_return:
                row.update(safe_navigation_teacher=bool(labels[0, slot]),
                           guarded_return_deg=float(guided[0, slot]) * config.max_hdg_delta_deg,
                           guidance_applied=bool((commands[0, slot] != projected_commands[0, slot]).any()))
            aircraft[slot]["decisions"].append(row)
        graph, reward, done, info = env.step_native_audited(commands.numpy(), track_separation=False)
        if return_audit is not None:
            rewards = return_audit.rewards(previous, graph, env.physical, proposal, commands, reward, done, info)
            for slot in active.nonzero().flatten().tolist():
                row = aircraft[slot]["decisions"][-1]
                row["reward"] = {key: float(value[slot]) for key, value in rewards.items()}
                row["target_value"] = {key: float(value[slot]) for key, value in target_values.items()}
        reward = torch.as_tensor(reward)[0]
        exits = active & (reward == -50) & ~torch.as_tensor(info["collision_aircraft"])[0]
        arrivals = active & (reward == 200)
        for outcome, mask in (("arrival", arrivals), ("exit", exits),
                              ("collision", torch.as_tensor(info["collision_aircraft"])[0])):
            if int(mask.sum()) != int(info[{"arrival": "arrivals", "exit": "exits",
                                           "collision": "collisions"}[outcome]][0]):
                raise ValueError(f"Aircraft identities differ from the native {outcome} counter")
            for slot in mask.nonzero().flatten().tolist():
                aircraft[slot]["outcome"] = outcome
        if done[0]:
            break
    summary = dict(checkpoint_sha256=_checksum(args.checkpoint), seed=args.seed, family=args.family,
                   sim_dt=config.sim_dt, control_dt=config.control_dt, projection=not args.without_projection,
                   apply_linked_return=args.apply_linked_return, max_cross_track_km=config.max_xtk_km,
                   minimum_cross_track_km=args.minimum_cross_track_km,
                   navigation_reference_sha256=_checksum(args.navigation_reference) if reference is not None else None,
                   aircraft=aircraft)
    if return_audit is not None:
        summary["return_audit"] = return_audit.metadata
        summary["first_guidance_q"] = first_guidance
    (args.output / "trace.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("yaw_plus15", "yaw_minus15"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--without-projection", action="store_true")
    parser.add_argument("--audit-linked-return", action="store_true")
    parser.add_argument("--apply-linked-return", action="store_true",
                        help="Apply guarded linked-aircraft turns in a counterfactual simulation")
    parser.add_argument("--minimum-cross-track-km", type=float, default=0.0)
    parser.add_argument("--navigation-reference", type=Path)
    parser.add_argument("--return-audit-config", type=Path)
    run(parser.parse_args())
