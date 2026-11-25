"""Run device replay and Geo-ADMM policy updates from either BlueSky sampler."""

import argparse
from dataclasses import asdict
from errno import EDQUOT, ENOSPC
from functools import partial
import importlib.util
import json
from math import cos, isfinite, pi
from pathlib import Path
import sys
from time import perf_counter, sleep

import torch

from envs.bluesky_graph import (
    FAR_CONFLICT_FEATURES, SELECTED_ALTITUDE_EDGE_FEATURES, SELECTED_ALTITUDE_NODE_FEATURE, ObservationProfile,
)
from envs.bluesky_safety import lookahead_risk, protected_command
from envs.bluesky_traffic import FAMILIES, MIXTURE_NAMES
from train.td3 import (
    REPLAY_EVENTS, CollisionHorizon, CrossingExploration, DeviceReplay, MultiStepReturns, Rollout, TD3, encode_executed_action,
    load_teacher_data, mirror_replay_batch,
)


def _environment(args, device):
    if args.plan in ("a", "native"):
        from envs.bluesky_workers import NativeWorkers, WorkerConfig

        config = WorkerConfig(args.envs, args.aircraft, args.sim_dt, args.control_dt, args.max_steps,
                              args.seed, args.plan == "native", args.family,
                              args.audit_frames or args.timed_departures or bool(args.collision_critic_weight),
                              args.timed_departures,
                              detection_metres=1000 * args.detection_km,
                              geodesic_navigation=args.geodesic_navigation, correlated_yaw=args.correlated_yaw,
                              training_yaw_degrees=args.training_yaw_degrees,
                              cross_track_observation=args.cross_track_observation,
                              far_conflict_observation=args.far_conflict_observation,
                              selected_altitude_observation=args.selected_altitude_observation,
                              navdata=getattr(args, "navdata", None))
        return NativeWorkers(args.workers, config, args.output, device)

    from bluesky.rl_gpu import B737PerfConfig, TorchVecEnvConfig
    from envs.bluesky_plan_b import GeoTorchBlueSky

    config = TorchVecEnvConfig(num_envs=args.envs, max_aircraft=args.aircraft, sim_dt=args.sim_dt,
                               control_dt=args.control_dt, max_steps=args.max_steps, hdg_deadband_deg=0.25,
                               device=str(device), seed=args.seed, dtype=torch.float64 if args.physics64 else torch.float32)
    perf = None
    if args.performance_calibration:
        measured = json.loads(args.performance_calibration.read_text())
        names = ("axmax_mps2", "bank_def_rad", "vs_def_mps")
        perf = B737PerfConfig(**{name: measured[name] for name in names})
    return GeoTorchBlueSky(config, family=args.family, perf=perf, timed_departures=args.timed_departures,
                          audit_physics=bool(args.collision_critic_weight and args.family != "full_mixture"
                                             and args.event_checks_per_decision == 1),
                          detection_metres=1000 * args.detection_km, geodesic_navigation=args.geodesic_navigation,
                          correlated_yaw=args.correlated_yaw, cross_track_observation=args.cross_track_observation,
                          native_contact_tolerance=args.native_contact_tolerance,
                          event_checks_per_decision=args.event_checks_per_decision,
                          far_conflict_observation=args.far_conflict_observation,
                          selected_altitude_observation=args.selected_altitude_observation)


def _shaping(graph, future, action, terminal):
    active = graph[-1]
    progress = (graph[0][..., 7] - future[0][..., 7]).clamp(-0.02, 0.02) * 25
    turn_improvement = (future[0][..., 6] - graph[0][..., 6]).clamp(-0.2, 0.2)
    hold = action[..., 0] * (~graph[2].any(dim=-1))
    shaped = (progress + 0.1 * turn_improvement).masked_fill(terminal, 0.0)
    return (shaped + 0.015 * hold).masked_fill(~active, 0.0)


def _cross_track_penalty(graph, future, terminal, category):
    free = ~graph[2].any(dim=-1) & ~future[2].any(dim=-1)
    return_to_route = (category == 2).unsqueeze(-1) & graph[-1] & (graph[0][..., 8] < 0.5)
    deviation = (future[0][..., 9].abs() - 0.15).clamp_min(0).square()
    return deviation.masked_fill(~(free & return_to_route) | terminal, 0.0)


def _write_with_storage_retry(write, path):
    delays = (2, 5, 10, 20, 40)
    for attempt in range(len(delays) + 1):
        try:
            write()
            return
        except OSError as error:
            if error.errno not in (ENOSPC, EDQUOT) or attempt == len(delays):
                raise
            print(f"{path}: {error.strerror}. Retrying in {delays[attempt]} seconds.",
                  file=sys.stderr, flush=True)
            sleep(delays[attempt])


def _append_metrics(path, record):
    data = (record + "\n").encode()
    offset = None

    def append():
        nonlocal offset
        with path.open("ab", buffering=0) as log:
            end = log.tell()
            if offset is None:
                offset = end
            elif end > offset:
                # Roll back a partial write before retrying this same record.
                log.truncate(offset)
            if log.write(data) != len(data):
                raise OSError(ENOSPC, "Incomplete metrics append")

    _write_with_storage_retry(append, path)


def _save(learner, args, decision, totals, filename):
    checkpoint = args.output / filename
    temporary = checkpoint.with_suffix(".tmp")
    state = {
        "actor": learner.actor.state_dict(), "critic": learner.critic.state_dict(), "width": args.width,
        "target_actor": learner.target_actor.state_dict(), "target_critic": learner.target_critic.state_dict(),
        "actor_optimizer": learner.actor_optimizer.state_dict(), "critic_optimizer": learner.critic_optimizer.state_dict(),
        "updates": learner.updates, "decisions": decision, "episodes": int(totals[0]),
        "training_plan": args.plan,
        "equivariant_actor": getattr(learner.actor, "equivariant_actor", False),
        "outcome_counts": totals.tolist(),
        "observation_profile": asdict(ObservationProfile(1000 * args.detection_km, args.geodesic_navigation,
                                                       args.cross_track_observation,
                                                       args.far_conflict_observation,
                                                       args.selected_altitude_observation)),
    }
    if state["equivariant_actor"]:
        state["admm_warmstart_reflection"] = "even"
    if learner.collision_heads is not None:
        state["collision_critic"] = learner.collision_state()
    def save():
        with temporary.open("wb") as handle:
            torch.save(state, handle)
        temporary.replace(checkpoint)

    _write_with_storage_retry(save, checkpoint)


@torch.no_grad()
def _reset_columns(weight, columns, optimizer=None):
    weight[:, columns] = 0
    if optimizer is not None:
        for value in optimizer.state.get(weight, {}).values():
            if torch.is_tensor(value) and value.shape == weight.shape:
                value[:, columns] = 0


def _reset_far_conflict_weights(network, optimizer=None):
    _reset_columns(network.encoder.embed_nodes.weight, FAR_CONFLICT_FEATURES, optimizer)


@torch.no_grad()
def _initialize_selected_altitude_weights(network, optimizer=None, *, from_plan_b=False):
    nodes = network.encoder.embed_nodes.weight
    selected = SELECTED_ALTITUDE_NODE_FEATURE
    if from_plan_b:
        # The old Plan B's field 3 was altitude minus selected altitude.
        former = -nodes[:, 3].clone()
        _reset_columns(nodes, (3, selected), optimizer)
        nodes[:, selected].copy_(former)
    else:
        _reset_columns(nodes, selected, optimizer)
    _reset_columns(network.encoder.embed_edges.weight, SELECTED_ALTITUDE_EDGE_FEATURES, optimizer)


def _training_plan(checkpoint, path):
    if "training_plan" in checkpoint:
        return checkpoint["training_plan"]
    config = path.with_name("config.json")
    return json.loads(config.read_text()).get("plan") if config.is_file() else None


def _unprotected_worlds(previous, reset, fraction, rng):
    draws = torch.rand(previous.shape, device=previous.device, generator=rng)
    return torch.where(reset, draws < fraction, previous)


def _run(args, device, learner, environment, monitor=None, resume=None):
    graph = environment.reset()
    exploration = (CrossingExploration(graph[-1], args.crossing_exploration_steps, args.seed + 29)
                   if args.crossing_exploration_steps > 1 else None)
    horizon = CollisionHorizon(args.collision_label_horizon) if args.collision_label_horizon else None
    collision_returns = bool(args.multi_step_return and args.collision_critic_weight and args.collision_critic_target == "td")
    return_window = (MultiStepReturns(args.return_window, learner.discount, aggregate=args.multi_step_return,
                                      collision_returns=collision_returns) if args.return_window > 1 else None)
    replay = DeviceReplay(args.replay, device, args.seed + 1, stratified=args.stratified_replay,
                          terminal_replay=args.terminal_replay, collision_labels=bool(args.collision_critic_weight),
                          observed_collision_labels=horizon is not None, route_counts=args.stratified_replay,
                          topology_replay=args.topology_replay, collision_returns=collision_returns,
                          protection_labels=bool(args.unprotected_rollout_fraction),
                          successor_geometry=args.projected_target_action)
    project = partial(protected_command, iterations=args.admm_k, fallback=args.admm_fallback,
                      by_component=args.component_projection, vertical_heading=args.vertical_heading,
                      directional_heading=args.directional_heading, extra_rounds=args.dense_extra_rounds,
                      batched_layer_search=args.batched_layer_search, joint_layer_search=args.joint_layer_search,
                      batched_layer_margin=getattr(args, "batched_layer_margin", 0.0),
                      early_layer_search=getattr(args, "early_layer_search", False),
                      exact_layer_forecast=getattr(args, "exact_layer_forecast", False),
                      preserve_imminent_clearance=getattr(args, "preserve_imminent_clearance", False),
                      joint_layer_reserve=args.joint_layer_reserve)
    target_projector = project if args.projected_target_action else None
    actor_projector = project if args.projected_actor_ste else None
    sampled_worlds = args.batch // 2 if args.mirror_replay else args.batch
    start_decision = int(resume["decisions"]) if resume else 0
    initial_counts = resume.get("outcome_counts", [int(resume["episodes"]), 0, 0, 0, 0]) if resume else [0] * 5
    totals = torch.tensor(initial_counts, dtype=torch.int64, device=device)
    start_updates = learner.updates
    learner.anchor_start_update = start_updates
    if args.decisions <= start_decision:
        raise ValueError("Decision limit must exceed the saved decision count")
    losses = torch.zeros(2, device=device)
    coordination_losses = torch.zeros((), device=device) if args.coordination_weight else None
    collision_losses = torch.zeros((), device=device) if args.collision_critic_weight else None
    collision_labels_seen = torch.zeros((), dtype=torch.long, device=device) if args.collision_critic_weight else None
    observed_collision_labels_seen = torch.zeros((), dtype=torch.long, device=device) if horizon else None
    interventions = torch.zeros((), dtype=torch.int64, device=device)
    accumulated_risk = torch.zeros((), device=device)
    trained_mixture = torch.zeros(3, dtype=torch.long, device=device)
    critic_events = torch.zeros(len(REPLAY_EVENTS), dtype=torch.long, device=device)
    actor_events = torch.zeros_like(critic_events)
    sampled_topology = torch.zeros((len(MIXTURE_NAMES), 9), dtype=torch.long, device=device)
    mixed_traffic = args.family == "full_mixture"
    outcomes_by_traffic = torch.zeros((len(MIXTURE_NAMES), len(initial_counts)), dtype=torch.long, device=device)
    unprotected = None
    if args.unprotected_rollout_fraction:
        projection_rng = torch.Generator(device=device).manual_seed(args.seed + 19)
        first = torch.ones(graph[-1].shape[0], dtype=torch.bool, device=device)
        unprotected = _unprotected_worlds(~first, first, args.unprotected_rollout_fraction, projection_rng)
        unprotected_steps = torch.zeros((), dtype=torch.long, device=device)
        unprotected_outcomes = torch.zeros_like(totals)
        protection_events = torch.zeros((2, len(REPLAY_EVENTS)), dtype=torch.long, device=device)
    explored = 0
    schedule_start = start_updates if args.update_schedule_from_resume else 0
    monitor_interval = getattr(args, "monitor_every_episodes", 0)
    next_episode_check = initial_counts[0] + monitor_interval
    stopped_by_monitor = False
    next_checkpoint = (schedule_start + ((start_updates - schedule_start) // args.checkpoint_every + 1)
                       * args.checkpoint_every if args.checkpoint_every else 0)
    begin = perf_counter()
    last = begin

    def learn(decision):
        if replay.size >= args.batch and decision - start_decision >= args.warmup:
            for _ in range(args.updates):
                if args.cosine_updates:
                    fraction = min((learner.updates - schedule_start) / args.cosine_updates, 1.0)
                    rate = args.learning_rate * (0.1 + 0.9 * (1 + cos(pi * fraction)) / 2)
                    for optimizer in (learner.actor_optimizer, learner.critic_optimizer):
                        optimizer.param_groups[0]["lr"] = rate
                batch, actor_batch = replay.sample_for_learning(sampled_worlds, update_actor=learner.updates % 2 == 1)
                if args.mirror_replay:
                    batch = mirror_replay_batch(batch)
                    if actor_batch is not None:
                        actor_batch = mirror_replay_batch(actor_batch)
                if batch.unprotected is not None:
                    indices = batch.unprotected.long() * len(REPLAY_EVENTS) + batch.events
                    counts = torch.nn.functional.one_hot(indices, num_classes=protection_events.numel()).sum(dim=0)
                    protection_events.add_(counts.reshape_as(protection_events))
                if args.stratified_replay:
                    trained_mixture.add_(torch.nn.functional.one_hot(batch.category, num_classes=3).sum(dim=0))
                    critic_events.add_(torch.nn.functional.one_hot(batch.events, num_classes=len(REPLAY_EVENTS)).sum(dim=0))
                    indices = batch.category * sampled_topology.shape[-1] + batch.route_count
                    counts = torch.nn.functional.one_hot(indices, num_classes=sampled_topology.numel()).sum(dim=0)
                    sampled_topology.add_(counts.reshape_as(sampled_topology))
                critic, actor = learner.update(batch, actor_batch=actor_batch, target_projector=target_projector,
                                              actor_projector=actor_projector)
                losses[0] += critic
                if collision_losses is not None:
                    collision_losses.add_(learner.last_collision_loss)
                    collision_labels_seen.add_(batch.collision_aircraft.sum())
                if observed_collision_labels_seen is not None:
                    observed_collision_labels_seen.add_(batch.observed_collision.sum())
                if actor is not None:
                    losses[1] += actor
                    if coordination_losses is not None:
                        coordination_losses.add_(learner.last_coordination_loss)
                    if args.stratified_replay:
                        actor_sample = actor_batch if actor_batch is not None else batch
                        actor_events.add_(torch.nn.functional.one_hot(actor_sample.events,
                                                                      num_classes=len(REPLAY_EVENTS)).sum(dim=0))

    for decision in range(start_decision + 1, args.decisions + 1):
        chunks = zip(*(tensor.split(args.inference_batch) for tensor in graph), strict=True)
        if exploration is None:
            predictions = [learner.proposal_and_consensus(chunk, explore=True) for chunk in chunks]
        else:
            noise = exploration.draw(graph[-1], environment.category)
            noise_chunks = zip(*(tensor.split(args.inference_batch) for tensor in noise), strict=True)
            predictions = [learner.proposal_and_consensus(chunk, explore=True, crossing_exploration=bias)
                           for chunk, bias in zip(chunks, noise_chunks, strict=True)]
        action, *consensus = [torch.cat(tensors) for tensors in zip(*predictions, strict=True)]
        physical = environment.physical
        executed = action[..., 4:]
        feasible = torch.zeros_like(graph[-1])
        if args.admm_k:
            executed, feasible = project(action, physical, graph[2], graph[3], consensus)
        if unprotected is not None:
            executed = torch.where(unprotected[:, None, None], action[..., 4:], executed)
            feasible = feasible & ~unprotected.unsqueeze(-1)
        if args.overlap_native:
            environment.start_step(executed)
            learn(decision)
            future, reward, done, info = environment.finish_step()
        else:
            future, reward, done, info = environment.step(executed)
        correction = (executed - action[..., 4:]).square().sum(dim=-1)
        interventions += ((correction > 1e-5) & graph[-1]).sum()
        reward = reward.float() - 0.05 * correction * graph[-1]
        terminal = ~future[-1] | done.unsqueeze(-1)
        if exploration is not None:
            exploration.end(terminal)
        if args.shaped_reward:
            reward = reward + _shaping(graph, future, action, terminal)
        if args.lookahead_weight:
            risk = lookahead_risk(environment.physical, future[-1],
                                  dense=args.dense_risk, overlap=args.overlap_risk).masked_fill(terminal, 0.0)
            reward = reward - args.lookahead_weight * risk
            accumulated_risk += risk.sum()
        learned_projection = graph[-1] & feasible & (correction > 1e-4)
        category = info.get("category")
        if category is None:
            category = torch.zeros_like(done, dtype=torch.long)
        if args.cross_track_weight:
            reward = reward - args.cross_track_weight * _cross_track_penalty(graph, future, terminal, category)
        events = torch.where(info["collisions"] > 0, 2, (info["arrivals"] > 0).long())
        replay_action = encode_executed_action(action, executed) if args.executed_action_critic else action
        transition = Rollout(graph, replay_action, reward, future, terminal, executed, learned_projection,
                             physical, category, events,
                             info["collision_aircraft"] if args.collision_critic_weight else None,
                             route_count=info["route_count"] if args.stratified_replay else None,
                             unprotected=unprotected,
                             next_physical=environment.physical if args.projected_target_action else None)
        if horizon is not None:
            transition = horizon.append(transition)
        if transition is not None and return_window is not None:
            transition = return_window.append(transition)
        if transition is not None:
            replay.append(transition)
        graph = future
        explored += action.shape[0]
        zero = torch.zeros_like(done, dtype=torch.long)
        outcomes = torch.stack((done, info["arrivals"], info["collisions"],
                                info.get("exits", zero), info.get("timeouts", zero)), dim=-1)
        totals += outcomes.sum(dim=0)
        if unprotected is not None:
            unprotected_steps += unprotected.sum()
            unprotected_outcomes += (outcomes * unprotected.unsqueeze(-1)).sum(dim=0)
            unprotected = _unprotected_worlds(unprotected, done, args.unprotected_rollout_fraction, projection_rng)
        if mixed_traffic:
            membership = torch.nn.functional.one_hot(category, num_classes=len(MIXTURE_NAMES))
            outcomes_by_traffic += (membership.unsqueeze(-1) * outcomes.unsqueeze(-2)).sum(dim=0)
        if not args.overlap_native:
            learn(decision)
        checkpoint_due = next_checkpoint and learner.updates >= next_checkpoint
        episode_boundary = next_episode_check
        if monitor_interval and args.target_episodes and learner.updates >= args.target_updates:
            episode_boundary = min(episode_boundary, args.target_episodes)
        episode_check_due = monitor_interval and int(totals[0]) >= episode_boundary
        if decision % args.log_every == 0 or decision == args.decisions or checkpoint_due or episode_check_due:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - begin
            episodes, arrivals, collisions, exits, timeouts = totals.tolist()
            target_met = bool(args.target_episodes or args.target_updates) and (
                episodes >= args.target_episodes and learner.updates >= args.target_updates
            )
            session_updates = learner.updates - start_updates
            result = {
                "plan": args.plan, "decisions": decision, "seconds": elapsed,
                "episodes": episodes, "arrivals": arrivals, "collisions": collisions,
                "exits": exits, "timeouts": timeouts, "target_met": target_met,
                "admm_interventions": int(interventions),
                "episodes_per_hour": 3600 * (episodes - initial_counts[0]) / elapsed, "updates": learner.updates,
                "updates_per_second": session_updates / elapsed, "env_steps_per_second": explored / elapsed,
                "replay_samples_per_second": args.batch * session_updates / elapsed,
                "peak_torch_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                "critic_loss": float(losses[0]) / max(session_updates, 1),
                "actor_loss": float(losses[1]) / max(learner.updates // 2 - start_updates // 2, 1),
                "interval_seconds": perf_counter() - last,
                "learning_rate": learner.actor_optimizer.param_groups[0]["lr"],
                "update_schedule_start": schedule_start,
                "lookahead_risk_per_airspace_step": float(accumulated_risk) / explored,
            }
            if resume:
                result["resume_decision"] = start_decision
                result["resume_updates"] = start_updates
                result["previous_outcomes_available"] = "outcome_counts" in resume
            if coordination_losses is not None:
                actor_updates = learner.updates // 2 - start_updates // 2
                result["coordination_loss"] = float(coordination_losses) / max(actor_updates, 1)
            if args.anchor:
                result["anchor_weight"] = learner.current_anchor_weight
            if collision_losses is not None:
                result["collision_critic_loss"] = float(collision_losses) / max(session_updates, 1)
                result["collision_critic_updates"] = learner.collision_updates
                result["collision_aircraft_sampled"] = int(collision_labels_seen)
            if observed_collision_labels_seen is not None:
                result["observed_collision_aircraft_sampled"] = int(observed_collision_labels_seen)
            if return_window is not None:
                result["critic_return_decisions"] = args.return_window if args.multi_step_return else 1
                result["replay_maturity_decisions"] = args.return_window + max(args.collision_label_horizon - 1, 0)
            if exploration is not None:
                result["crossing_exploration_steps"] = args.crossing_exploration_steps
            if unprotected is not None:
                result["unprotected_airspace_steps"] = int(unprotected_steps)
                result["unprotected_outcomes"] = dict(zip(("episodes", "arrivals", "collisions", "exits", "timeouts"),
                                                           unprotected_outcomes.tolist(), strict=True))
                result["critic_replay_events_by_protection"] = {
                    name: dict(zip(REPLAY_EVENTS, counts, strict=True))
                    for name, counts in zip(("protected", "unprotected"), protection_events.tolist(), strict=True)
                }
            if mixed_traffic and hasattr(environment, "traffic_draws"):
                result["traffic_draws"] = dict(zip(MIXTURE_NAMES, environment.traffic_draws.tolist(), strict=True))
                fields = ("episodes", "arrivals", "collisions", "exits", "timeouts")
                result["session_outcomes_by_traffic"] = {
                    name: dict(zip(fields, counts, strict=True))
                    for name, counts in zip(MIXTURE_NAMES, outcomes_by_traffic.tolist(), strict=True)
                }
                if args.stratified_replay:
                    result["training_sample_mixture"] = dict(zip(MIXTURE_NAMES, trained_mixture.tolist(), strict=True))
                    result["critic_replay_events"] = dict(zip(REPLAY_EVENTS, critic_events.tolist(), strict=True))
                    result["actor_replay_events"] = dict(zip(REPLAY_EVENTS, actor_events.tolist(), strict=True))
                    result["training_sample_topology"] = {
                        name: {str(lanes): counts[lanes] for lanes in (2, 3, 4, 6, 8)}
                        for name, counts in zip(MIXTURE_NAMES, sampled_topology.tolist(), strict=True)
                    }
                if args.timed_departures:
                    result["launched_aircraft"] = int(environment.launched_aircraft)
            record = json.dumps(result, allow_nan=False)
            _append_metrics(args.output / "metrics.jsonl", record)
            print(record, flush=True)
            last = perf_counter()
            if checkpoint_due:
                _save(learner, args, decision, totals, f"checkpoint-{learner.updates:07d}.pt")
                next_checkpoint += args.checkpoint_every
                while next_checkpoint <= learner.updates:
                    next_checkpoint += args.checkpoint_every
            if monitor is not None:
                before = perf_counter()
                stopped_by_monitor = monitor(learner, args, decision, totals, target_met or decision == args.decisions) is True
                if episode_check_due:
                    next_episode_check = initial_counts[0] + (
                        (episodes - initial_counts[0]) // monitor_interval + 1) * monitor_interval
                spent = perf_counter() - before
                begin += spent
                last += spent
            if target_met or stopped_by_monitor:
                break
    _save(learner, args, decision, totals, "checkpoint.pt")
    if (args.target_episodes or args.target_updates) and not (target_met or stopped_by_monitor):
        raise RuntimeError("Decision limit reached before both training targets were met")


def train(args, *, monitor=None):
    if args.initialize_critic and not args.initialize:
        raise ValueError("Critic initialization needs an input checkpoint via --initialize")
    if args.timed_departures and args.family != "full_mixture":
        raise ValueError("Timed departures require the full mixture")
    if args.correlated_yaw and args.family != "full_mixture":
        raise ValueError("Correlated yaw requires the full mixture")
    if args.training_yaw_degrees is not None and (
            args.plan == "b" or args.family != "full_mixture" or not 0 < args.training_yaw_degrees <= 20):
        raise ValueError("Fixed training yaw requires the native full mixture and a magnitude in (0, 20]")
    if args.stratified_replay and args.family != "full_mixture":
        raise ValueError("Replay stratification requires full-mixture scenario labels")
    if args.terminal_replay and not args.stratified_replay:
        raise ValueError("Terminal replay requires family-stratified replay")
    if args.topology_replay and not args.stratified_replay:
        raise ValueError("Topology replay requires family-stratified replay")
    if args.head_on_anchor and (args.family != "full_mixture" or not args.anchor or not args.anchor_weight):
        raise ValueError("Head-on anchoring requires a weighted anchor in the full mixture")
    if args.anchor_families and (args.head_on_anchor or args.family != "full_mixture"
                                or not args.anchor or not args.anchor_weight):
        raise ValueError("Family-specific anchoring requires one weighted anchor in the full mixture")
    if args.mirror_replay and (args.batch < 2 or args.batch % 2):
        raise ValueError("Mirrored replay needs an even batch with at least two airspaces")
    if args.safe_navigation_weight and not args.navigation_weight:
        raise ValueError("Safe-turn supervision requires a positive navigation weight")
    if not isfinite(args.rollout_router_temperature) or args.rollout_router_temperature <= 0:
        raise ValueError("Rollout router sampling temperature must be positive and finite")
    if args.crossing_exploration_steps < 1 or (args.crossing_exploration_steps > 1
            and (args.plan != "b" or args.family != "full_mixture")):
        raise ValueError("Persistent crossing exploration requires Plan B full-mixture traffic")
    if not isfinite(args.cross_track_weight) or args.cross_track_weight < 0:
        raise ValueError("Cross-track weight must be nonnegative and finite")
    if args.cross_track_weight and args.family != "full_mixture":
        raise ValueError("Cross-track reward requires the full mixture")
    if args.batched_layer_search and not args.admm_k:
        raise ValueError("Batched layer search requires ADMM projection")
    if args.joint_layer_search and not args.batched_layer_search:
        raise ValueError("Three-aircraft layer search requires batched layer search")
    margin = getattr(args, "batched_layer_margin", 0.0)
    if not isfinite(margin) or margin < 0:
        raise ValueError("Vertical clearance margin must be finite and nonnegative")
    if (margin or getattr(args, "early_layer_search", False) or getattr(args, "exact_layer_forecast", False)
            or getattr(args, "preserve_imminent_clearance", False)) and not args.batched_layer_search:
        raise ValueError("Layer forecast refinements require batched layer search")
    if not isfinite(args.joint_layer_reserve) or args.joint_layer_reserve < 0 or (
            args.joint_layer_reserve and not args.joint_layer_search):
        raise ValueError("The reserve must be nonnegative and requires three-aircraft layer search")
    if not isfinite(args.coordination_weight) or args.coordination_weight < 0 or (
            args.coordination_weight and (args.admm_k < 1 or args.dense_extra_rounds < 0)):
        raise ValueError("Coordination training needs a finite nonnegative weight and ADMM projection")
    if not 0 <= args.unprotected_rollout_fraction <= 1:
        raise ValueError("Unprotected rollout fraction must be between zero and one")
    if args.unprotected_rollout_fraction and (not args.admm_k or not args.executed_action_critic):
        raise ValueError("Mixed projection sampling requires ADMM and execution-aligned critic actions")
    if (args.projected_target_action or args.projected_actor_ste) and (not args.admm_k or not args.executed_action_critic):
        raise ValueError("Projected critic evaluations require ADMM and execution-aligned critic actions")
    if not isfinite(args.collision_critic_weight) or args.collision_critic_weight < 0 or args.collision_critic_warmup < 0:
        raise ValueError("Collision critic weight must be finite and nonnegative, and warmup nonnegative")
    if args.collision_label_horizon < 0 or (args.collision_label_horizon and not args.collision_critic_weight):
        raise ValueError("Delayed collision labels require a positive critic weight and horizon")
    if args.collision_critic_target == "observed" and not args.collision_label_horizon:
        raise ValueError("Observed collision targets require a label horizon")
    if args.return_window < 1 or (args.multi_step_return and args.return_window < 2):
        raise ValueError("Multi-step returns require a replay window of at least two decisions")
    if (not isfinite(args.teacher_weight) or args.teacher_weight < 0
            or bool(args.teacher_data) != (args.teacher_weight > 0)):
        raise ValueError("Teacher data and a positive teacher weight must be supplied together")
    if args.overlap_native and args.plan == "b":
        raise ValueError("Native rollout overlap requires CPU subprocesses")
    if args.native_contact_tolerance and args.plan != "b":
        raise ValueError("Native contact tolerance is an option for the tensor simulator")
    if args.event_checks_per_decision < 1 or (args.event_checks_per_decision != 1 and args.plan != "b"):
        raise ValueError("Additional event checks require the tensor simulator")
    if args.expected_mode_risk and (args.raw_risk_weight <= 0 or args.dense_risk or args.overlap_risk):
        raise ValueError("Expected mode risk requires positive raw risk weight and the standard risk geometry")
    if args.resume and args.initialize:
        raise ValueError("Choose either a fresh initialization or a training checkpoint resume")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "ppu" and importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    device = torch.device("cuda:0" if args.device == "ppu" else "cpu")
    if device.type == "cuda" and (torch.cuda.device_count() != 1 or "PPU" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Train on exactly one PPU")
    torch.set_num_threads(1)
    path = args.resume or args.initialize
    source = torch.load(path, map_location=device, weights_only=True) if path else None
    equivariant = getattr(args, "equivariant_actor", None)
    previous_equivariance = bool(source and source.get("equivariant_actor", False))
    args.equivariant_actor = previous_equivariance if equivariant is None else equivariant
    previous = ObservationProfile(**source.get("observation_profile", {})) if source else ObservationProfile()
    profile = ObservationProfile(previous.detection_metres if args.detection_km is None else 1000 * args.detection_km,
                                 previous.geodesic_navigation if args.geodesic_navigation is None
                                 else args.geodesic_navigation,
                                 previous.cross_track_observation if args.cross_track_observation is None
                                 else args.cross_track_observation,
                                 previous.far_conflict_observation if args.far_conflict_observation is None
                                 else args.far_conflict_observation,
                                 previous.selected_altitude_observation if args.selected_altitude_observation is None
                                 else args.selected_altitude_observation)
    if args.cross_track_weight and not profile.cross_track_observation:
        raise ValueError("Cross-track reward requires the cross-track observation")
    args.detection_km = profile.detection_metres / 1000
    args.geodesic_navigation = profile.geodesic_navigation
    args.cross_track_observation = profile.cross_track_observation
    args.far_conflict_observation = profile.far_conflict_observation
    args.selected_altitude_observation = profile.selected_altitude_observation
    resume = source if args.resume else None
    if resume:
        expected = ("decisions", "episodes", "updates", "target_actor", "target_critic",
                    "actor_optimizer", "critic_optimizer")
        if any(key not in resume for key in expected):
            raise ValueError("Resume requires a saved optimizer and target-network checkpoint")
        # Start fresh traffic and replay after the durable checkpoint.
        args.seed += int(resume["decisions"])
    torch.manual_seed(args.seed)
    if source is not None and source["width"] != args.width:
        raise ValueError("Checkpoint and learner widths differ")
    anchor = torch.load(args.anchor, map_location=device, weights_only=True) if args.anchor else None
    if anchor is not None and anchor["width"] != args.width:
        raise ValueError("Anchor checkpoint and learner widths differ")
    learner = TD3(device=device, width=args.width, actor_state=source["actor"] if source else None,
                  equivariant_actor=args.equivariant_actor,
                  learning_rate=args.learning_rate, raw_risk_weight=args.raw_risk_weight, dense_risk=args.dense_risk,
                  overlap_risk=args.overlap_risk,
                  navigation_weight=args.navigation_weight, projection_weight=args.projection_weight,
                  anchor_state=anchor["actor"] if anchor else None, anchor_weight=args.anchor_weight,
                  head_on_anchor=args.head_on_anchor, safe_navigation_weight=args.safe_navigation_weight,
                  anchor_categories=tuple(MIXTURE_NAMES.index(name) for name in args.anchor_families),
                  teacher_data=load_teacher_data(args.teacher_data, device, profile), teacher_weight=args.teacher_weight,
                  rollout_router_temperature=args.rollout_router_temperature,
                  collision_critic_weight=args.collision_critic_weight, collision_critic_warmup=args.collision_critic_warmup,
                  expected_mode_risk_loss=args.expected_mode_risk, collision_critic_target=args.collision_critic_target,
                  return_steps=args.return_window if args.multi_step_return else 1,
                  coordination_weight=args.coordination_weight,
                  coordination_rounds=(args.admm_k, args.dense_extra_rounds))
    learner.actor.observation_profile = profile
    if learner.anchor is not None:
        learner.anchor.equivariant_actor = bool(anchor.get("equivariant_actor", False))
    if args.initialize_critic:
        learner.critic.load_state_dict(source["critic"])
        learner.target_critic.load_state_dict(source["critic"])
    if resume:
        learner.critic.load_state_dict(resume["critic"])
        learner.target_actor.load_state_dict(resume["target_actor"])
        learner.target_critic.load_state_dict(resume["target_critic"])
        learner.actor_optimizer.load_state_dict(resume["actor_optimizer"])
        learner.critic_optimizer.load_state_dict(resume["critic_optimizer"])
        learner.updates = int(resume["updates"])
        if args.collision_critic_weight and "collision_critic" in resume:
            learner.load_collision_state(resume["collision_critic"])
    if profile.far_conflict_observation:
        if source is not None and not previous.far_conflict_observation:
            _reset_far_conflict_weights(learner.actor, learner.actor_optimizer)
            _reset_far_conflict_weights(learner.target_actor)
            for critic in (learner.critic.first, learner.critic.second):
                _reset_far_conflict_weights(critic, learner.critic_optimizer)
            for critic in (learner.target_critic.first, learner.target_critic.second):
                _reset_far_conflict_weights(critic)
        if learner.anchor is not None and not ObservationProfile(**anchor.get("observation_profile", {})).far_conflict_observation:
            _reset_far_conflict_weights(learner.anchor)
    if profile.selected_altitude_observation:
        if source is not None and not previous.selected_altitude_observation:
            from_plan_b = _training_plan(source, path) == "b"
            _initialize_selected_altitude_weights(learner.actor, learner.actor_optimizer, from_plan_b=from_plan_b)
            _initialize_selected_altitude_weights(learner.target_actor, from_plan_b=from_plan_b)
            for critic in (learner.critic.first, learner.critic.second):
                _initialize_selected_altitude_weights(critic, learner.critic_optimizer, from_plan_b=from_plan_b)
            for critic in (learner.target_critic.first, learner.target_critic.second):
                _initialize_selected_altitude_weights(critic, from_plan_b=from_plan_b)
        if learner.anchor is not None and not ObservationProfile(
                **anchor.get("observation_profile", {})).selected_altitude_observation:
            _initialize_selected_altitude_weights(learner.anchor, from_plan_b=_training_plan(anchor, args.anchor) == "b")
    if args.freeze_actor_encoder:
        learner.actor.encoder.requires_grad_(False)
    if getattr(args, "freeze_actor_router", False):
        learner.actor.router.requires_grad_(False)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    env = _environment(args, device)
    try:
        _run(args, device, learner, env, monitor, resume)
    finally:
        if hasattr(env, "close"):
            env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=("a", "b", "native"), required=True)
    parser.add_argument("--device", choices=("cpu", "ppu"), default="ppu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--initialize-critic", action="store_true",
                        help="Load the input checkpoint's critics while starting fresh optimizer states and counters")
    parser.add_argument("--freeze-actor-encoder", action="store_true",
                        help="Update the actor's action heads while keeping its shared encoder fixed")
    parser.add_argument("--freeze-actor-router", action="store_true",
                        help="Keep the actor's action routing head fixed during training")
    parser.add_argument("--equivariant-actor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", type=Path, help="Resume counters and optimizer with fresh simulator and replay")
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--head-on-anchor", action="store_true", help="Preserve the reference on head-on traffic")
    parser.add_argument("--anchor-families", choices=MIXTURE_NAMES, nargs="+", default=[],
                        help="Apply the reference loss only to the specified training traffic families")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--envs", type=int, default=16, help="For Plan A this is the number of airspaces per worker")
    parser.add_argument("--aircraft", type=int, default=12)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--detection-km", type=float)
    parser.add_argument("--geodesic-navigation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cross-track-observation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--far-conflict-observation", action=argparse.BooleanOptionalAction, default=None,
                        help="Expose the earliest distant conflict candidate as local actor features")
    parser.add_argument("--selected-altitude-observation", action=argparse.BooleanOptionalAction, default=None,
                        help="Expose the selected altitude and align nominal altitude across the two simulators")
    parser.add_argument("--cross-track-weight", type=float, default=0.0,
                        help="Penalize lateral route deviation of conflict-free yaw aircraft beyond 3 km")
    parser.add_argument("--control-dt", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--target-episodes", type=int, default=0)
    parser.add_argument("--target-updates", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoints by learner update count")
    parser.add_argument("--update-schedule-from-resume", action="store_true",
                        help="Start checkpoint intervals and cosine decay at the incoming update count")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-router-temperature", type=float, default=1.0,
                        help="Scale policy mode probabilities during exploration")
    parser.add_argument("--crossing-exploration-steps", type=int, default=1,
                        help="Reuse the router and control exploration draws in crossing traffic for this many decisions")
    parser.add_argument("--cosine-updates", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--inference-batch", type=int, default=32)
    parser.add_argument("--replay", type=int, default=4096)
    parser.add_argument("--stratified-replay", action="store_true")
    parser.add_argument("--terminal-replay", action="store_true",
                        help="Reserve part of each family in critic batches for arrival and collision transitions")
    parser.add_argument("--topology-replay", action="store_true",
                        help="Reserve one quarter of crossing and yaw samples for high-route-count topologies")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--admm-k", type=int, default=0)
    parser.add_argument("--admm-fallback", type=int, default=0)
    parser.add_argument("--component-projection", action="store_true")
    parser.add_argument("--vertical-heading", action="store_true")
    parser.add_argument("--directional-heading", action="store_true")
    parser.add_argument("--dense-extra-rounds", type=int, default=0)
    parser.add_argument("--batched-layer-search", action="store_true",
                        help="Use device-resident paired altitude choices during rollouts")
    parser.add_argument("--joint-layer-search", action="store_true")
    parser.add_argument("--batched-layer-margin", type=float, default=0.0)
    parser.add_argument("--early-layer-search", action="store_true")
    parser.add_argument("--exact-layer-forecast", action="store_true")
    parser.add_argument("--preserve-imminent-clearance", action="store_true")
    parser.add_argument("--joint-layer-reserve", type=float, default=0.0)
    parser.add_argument("--executed-action-critic", action="store_true",
                        help="Match the critic's replay action to the command delivered by projection")
    parser.add_argument("--projected-target-action", action="store_true",
                        help="Use the successor state's projection for protected critic targets")
    parser.add_argument("--projected-actor-ste", action="store_true",
                        help="Evaluate actor Q after projection with a straight-through action gradient")
    parser.add_argument("--mirror-replay", action="store_true",
                        help="Pair each replay airspace with its left-right reflection")
    parser.add_argument("--unprotected-rollout-fraction", type=float, default=0.0,
                        help="Fraction of simulated episodes that execute only the learned policy's actions")
    parser.add_argument("--family", choices=FAMILIES, default="upstream")
    parser.add_argument("--timed-departures", action="store_true")
    parser.add_argument("--correlated-yaw", action="store_true",
                        help="Align the sign of initial yaw for half of the yaw airspaces in the full mixture")
    parser.add_argument("--training-yaw-degrees", type=float)
    parser.add_argument("--physics64", action="store_true")
    parser.add_argument("--native-contact-tolerance", action="store_true",
                        help="Use the native vertical contact tolerance in the tensor simulator")
    parser.add_argument("--event-checks-per-decision", type=int, default=1,
                        help="Divide a tensor simulator decision into this many event detection intervals")
    parser.add_argument("--performance-calibration", type=Path)
    parser.add_argument("--shaped-reward", action="store_true")
    parser.add_argument("--lookahead-weight", type=float, default=0.0)
    parser.add_argument("--raw-risk-weight", type=float, default=0.0)
    parser.add_argument("--expected-mode-risk", action="store_true",
                        help="Average the actor's pair risk over both aircraft's action mode probabilities")
    parser.add_argument("--collision-critic-weight", type=float, default=0.0,
                        help="Actor penalty for learned per-aircraft discounted collision risk")
    parser.add_argument("--collision-critic-warmup", type=int, default=600)
    parser.add_argument("--collision-label-horizon", type=int, default=0,
                        help="Delay replay until the next given number of decisions have been observed")
    parser.add_argument("--collision-critic-target", choices=("td", "observed"), default="td",
                        help="Learn collision risk by bootstrapping or from exact per-aircraft window outcomes")
    parser.add_argument("--return-window", type=int, default=1,
                        help="Wait for this many decisions before value replay")
    parser.add_argument("--multi-step-return", action="store_true",
                        help="Use all observed rewards in the replay window for TD value targets")
    parser.add_argument("--dense-risk", action="store_true")
    parser.add_argument("--overlap-risk", action="store_true",
                        help="Also penalize loss of vertical separation during the horizontal conflict interval")
    parser.add_argument("--overlap-native", action="store_true")
    parser.add_argument("--audit-frames", action="store_true", help="Audit native BlueSky outcomes each physics frame")
    parser.add_argument("--navigation-weight", type=float, default=0.0)
    parser.add_argument("--safe-navigation-weight", type=float, default=0.0,
                        help="Relative supervision weight for safe return turns inside the conflict graph")
    parser.add_argument("--projection-weight", type=float, default=0.0)
    parser.add_argument("--coordination-weight", type=float, default=0.0,
                        help="Differentiate the continuous projection to train the ADMM warm start")
    parser.add_argument("--teacher-data", type=Path, nargs="+", default=[])
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=82701)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
