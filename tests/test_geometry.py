from contextlib import contextmanager
from dataclasses import replace
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.admm_layer import _consensus_updates, altitude_admm, speed_admm
from core.conflict_graph import (
    _pair_clearance_torch, conflict_graph, connected_components, cylinder_interval, greedy_colors, native_vertical_radius,
)
from core.layer_search import improve_layer_choices, improve_three_aircraft_choices, layer_clearance
from core.native_vertical import altitude_paths
from core.rstd import speed_forbidden_interval
from envs.bluesky_graph import (
    FAR_CONFLICT_FEATURES, SELECTED_ALTITUDE_EDGE_FEATURES, SELECTED_ALTITUDE_NODE_FEATURE,
    cross_track_km, traffic_graph, with_far_horizontal_observation,
)
from envs.bluesky_safety import (
    PhysicalState, _directional_heading, _joint_clearance_reserve, _protect_heading, coordination_loss, expected_mode_risk,
    lookahead_risk, protected_command, single_axis_commands,
)
from envs.bluesky_traffic import sample_routes
from envs.observations import observe
from eval.admm_communication import DelayedPairExchange, PairExchange
from eval.admm_gap_mpc import plan_gap_maneuver
from eval.bluesky_collision_trace import _altitude_options, _classify, _component_plan
from eval.bluesky_layer_fallback import _partial_layer_choices
from eval.graph_statistics import GraphStatistics
from eval.bluesky_pair import FAMILIES, _checksum
from eval.stage_acceptance import (
    EXACT_NATIVE_PROTOCOL, MPC_COMMUNICATION, assess, exact_refinement_acceptance, paper_checkpoint_digest,
)
from models.graphformer import Actor, mirror_observations
from train import bluesky_td3, td3
from train.bluesky_td3 import (
    _cross_track_penalty, _initialize_selected_altitude_weights, _reset_far_conflict_weights, _unprotected_worlds,
)
from train.bluesky_stage2 import _meets_development_floors, passes_native_development
from train.td3 import (
    CollisionHorizon, DeviceReplay, MultiStepReturns, Rollout, TD3, _collision_target, _navigation_loss, _teacher_loss,
    encode_action, encode_executed_action, load_teacher_data,
)


def test_existing_only_graph_recomputes_degree_without_removing_existing_edges():
    from envs.bluesky_graph import SCALE

    position = torch.tensor([[[0., 0., 0.], [9000., 0., 0.], [-9000., 0., 130.]]])
    direction = torch.tensor([[[1., 0.], [-1., 0.], [1., 0.]]])
    speed = torch.full((1, 3), 185.)
    active = torch.ones(1, 3, dtype=torch.bool)
    options = dict(scale=SCALE, reference_altitude=0.)
    baseline, _, full = observe(position, direction, speed, active, **options)
    nodes, edges, stripped = observe(position, direction, speed, active, include_potential=False, **options)
    assert full.existing.any() and full.potential.any()
    assert torch.equal(stripped.adjacency, full.existing)
    assert not stripped.potential.any() and not stripped.cross_layer.any()
    assert not edges[..., [12, 16]].any()
    assert (nodes[..., 21] < baseline[..., 21]).any()


def test_native_development_gate_uses_aggregate_gain_and_family_rate_floors():
    names = ("crossing3", "crossing4", "crossing6", "head_on", "yaw_plus15", "yaw_minus15")
    def panel(arrivals):
        return dict(zip(names, (dict(arrivals=total, aircraft=36) for total in arrivals), strict=True))
    incoming = panel((36, 34, 30, 36, 36, 32))
    candidate = panel((36, 34, 34, 36, 34, 34))
    assert passes_native_development(candidate, incoming)
    assert passes_native_development(panel((36, 33, 34, 36, 34, 33)), incoming)
    assert not passes_native_development(panel((36, 35, 36, 36, 31, 36)), incoming)
    assert not passes_native_development(panel((36, 35, 36, 34, 35, 36)), incoming)
    assert not passes_native_development(panel((36, 33, 32, 36, 34, 34)), incoming)


def test_gap_mpc_replans_an_approaching_contact_and_holds_clear_traffic():
    position = torch.tensor([[[0., 0., 304.], [12000., 0., 224.]]])
    direction = torch.tensor([[[1., 0.], [-1., 0.]]])
    speed = torch.full((1, 2), 100.)
    active = torch.ones(1, 2, dtype=torch.bool)
    proposal = torch.zeros(1, 2, 3)
    state = (position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    commands, hold_contacts, planned_contacts = plan_gap_maneuver(state, active, proposal)
    assert hold_contacts.item() == 1 and planned_contacts.item() == 0
    assert commands[0, 1, 1] < 0
    separated = position.clone()
    separated[0, 1, 0] = 50000
    commands, hold_contacts, planned_contacts = plan_gap_maneuver((separated, *state[1:]), active, proposal)
    torch.testing.assert_close(commands, proposal)
    assert hold_contacts.item() == planned_contacts.item() == 0


def test_delayed_exchange_uses_last_matured_state_in_current_coordinates():
    exchange = DelayedPairExchange(0., 70111, 1, "cpu", delay=2)
    active = torch.tensor([[True, True]])
    live = torch.tensor([True])
    edges = torch.tensor([[[False, True], [False, False]]])
    baseline = torch.zeros(1, 2, 2, 2)
    exchange.begin_step(0, active, live)
    speed = exchange.layer(("primary", "speed"))
    layer = exchange.layer(("primary", "altitude"), torch.tensor([[100., 200.]]))
    speed.send(edges, baseline + 255., baseline + 1.)
    speed.send(edges, baseline + 250., baseline + 2.)
    layer.send(edges, baseline + 15., baseline + 3.)
    for tick in (0, 1):
        if tick:
            exchange.begin_step(tick, active, live)
        torch.testing.assert_close(speed.receive(edges, baseline, baseline)[0], baseline)
    exchange.begin_step(2, active, live)
    received, dual = speed.receive(edges, baseline, baseline)
    torch.testing.assert_close(received[edges], torch.tensor([[250., 250.]]))
    torch.testing.assert_close(dual[edges], torch.tensor([[2., 2.]]))
    layer = exchange.layer(("primary", "altitude"), torch.tensor([[103., 208.]]))
    received, dual = layer.receive(edges, baseline, baseline)
    torch.testing.assert_close(received[edges], torch.tensor([[12., 7.]]))
    torch.testing.assert_close(dual[edges], torch.tensor([[3., 3.]]))
    row, = exchange.report()
    assert row["attempted_pair_exchanges"] == row["delivered_pair_exchanges"] == 3
    assert row["in_flight_pair_exchanges"] == 0
    assert row["used_delayed_pair_states"] == row["mean_used_message_age_steps"] == 2
    assert row["min_used_message_age_steps"] == row["max_used_message_age_steps"] == 2


def test_delayed_packets_do_not_cross_inactive_slots_or_terminal_worlds():
    exchange = DelayedPairExchange(0., 70111, 2, "cpu", delay=2)
    active = torch.ones((2, 2), dtype=torch.bool)
    live = torch.tensor([True, True])
    edges = torch.ones((2, 2, 2), dtype=torch.bool).triu(1)
    baseline = torch.zeros(2, 2, 2, 2)
    exchange.begin_step(0, active, live)
    layer = exchange.layer(("primary", "speed"))
    layer.send(edges, baseline + 5., baseline)
    active[0, 1] = False
    live[1] = False
    exchange.begin_step(1, active, live)
    active[0, 1] = True
    exchange.begin_step(2, active, live)
    torch.testing.assert_close(layer.receive(edges, baseline, baseline)[0], baseline)
    first, second = exchange.report()
    assert first["delivered_pair_exchanges"] == 1 and first["used_delayed_pair_states"] == 0
    assert second["delivered_pair_exchanges"] == 0 and second["in_flight_pair_exchanges"] == 1


def test_lossy_consensus_preserves_the_previous_edge_state():
    reference = torch.tensor([[2., 4.]])
    edge = torch.tensor([[[False, True], [False, False]]])

    def project(proposal, _responsibility, _lower, _upper):
        return torch.zeros_like(proposal)

    def solve(rounds, exchange=None):
        return _consensus_updates(reference, 0., 10., edge, project, rounds, .5, None, None, exchange=exchange)

    reliable = PairExchange(0., 70111, 1, "cpu")
    for expected, observed in zip(solve(2), solve(2, reliable), strict=True):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    diagnostics, = reliable.report()
    assert diagnostics["attempted_pair_exchanges"] == diagnostics["delivered_pair_exchanges"] == 2
    assert diagnostics["observed_loss"] == 0.
    blocked = PairExchange(1., 70111, 1, "cpu")
    action, dual, _ = solve(2, blocked)
    torch.testing.assert_close(action, reference, rtol=0, atol=0)
    assert not dual.any()
    assert blocked.report()[0]["dropped_pair_exchanges"] == 2

    outcomes = iter((edge, torch.zeros_like(edge)))
    _, held_dual, residual = solve(2, lambda _: next(outcomes))
    torch.testing.assert_close(held_dual, solve(1)[1], rtol=0, atol=0)
    assert residual.item() > 0.


def test_pair_exchange_draws_follow_traffic_seed_and_exclude_completed_worlds():
    solo = PairExchange(.2, 70112, 1, "cpu")
    batch = PairExchange(.2, 70111, 2, "cpu")
    batch.live[0] = False
    graph = torch.ones(1, 20, 20, dtype=torch.bool).triu(1)
    for _ in range(5):
        torch.testing.assert_close(solo(graph)[0], batch(graph.expand(2, -1, -1))[1])
    assert solo.report()[0] == batch.report()[1]
    assert batch.report()[0]["attempted_pair_exchanges"] == 0
    assert 0.15 < solo.report()[0]["observed_loss"] < 0.25
    batch.observe_residuals(torch.tensor([3., 5.]), torch.tensor([10., 20.]))
    assert batch.report()[0]["projection_solves"] == 0
    assert batch.report()[1]["speed_primal_residual_mean_mps"] == 5.
    assert batch.report()[1]["altitude_primal_residual_max_m"] == 20.


def test_colored_admm_consumes_updates_from_previous_color_in_same_sweep():
    reference = torch.tensor([[2., 4.]])
    edge = torch.tensor([[[False, True], [False, False]]])

    def project(proposal, _responsibility, _lower, _upper):
        return torch.zeros_like(proposal)

    jacobi = PairExchange(0., 70111, 1, "cpu")
    colored = PairExchange(0., 70111, 1, "cpu")
    simultaneous, *_ = _consensus_updates(reference, 0., 10., edge, project, 1, .5, None, None,
                                          exchange=jacobi)
    sequential, *_ = _consensus_updates(reference, 0., 10., edge, project, 1, .5, None, None,
                                        exchange=colored, colored_updates=True)
    torch.testing.assert_close(simultaneous, reference, rtol=0, atol=0)
    torch.testing.assert_close(sequential, torch.tensor([[2., 4. / 3.]]))
    assert jacobi.report()[0]["attempted_pair_exchanges"] == 1
    assert colored.report()[0]["attempted_pair_exchanges"] == 2


def test_observed_graph_statistics_count_links_in_fleets_over_64():
    graph = np.zeros((80, 80), dtype=bool)
    for left, right in ((0, 71), (71, 79), (79, 0), (79, 50), (1, 0), (71, 71)):
        graph[left, right] = True
    active = np.zeros(80, dtype=bool)
    active[[0, 50, 71, 78, 79]] = True
    stats = GraphStatistics()
    stats.add(graph, active)
    assert stats.report() == dict(decisions=1, aircraft_decisions=5, peak_aircraft=5, degree_sum=8,
                                  mean_degree=1.6, max_degree=3, max_clique_size=3, max_component_size=4)


def test_observed_graph_clique_matches_exhaustive_subsets():
    rng = np.random.default_rng(202)
    for density in (0.0, 0.2, 0.5, 0.8, 1.0):
        graph = np.triu(rng.random((8, 8)) < density, 1)
        symmetric = graph | graph.T
        largest = max(size for size in range(1, 9)
                      if any(all(symmetric[left, right] for left, right in combinations(nodes, 2))
                             for nodes in combinations(range(8), size)))
        stats = GraphStatistics()
        stats.add(graph, np.ones(8, dtype=bool))
        assert stats.report()["max_clique_size"] == largest


def test_batched_layer_search_respects_capture_time_and_does_not_create_conflicts():
    z = torch.zeros(1, 2)
    targets = torch.tensor([[[0.0, 100.0], [0.0, 100.0]]])
    exit = torch.full((1, 2, 2), 60.0)
    early = layer_clearance(z, z, targets, torch.zeros_like(exit), exit) >= 91.44
    late = layer_clearance(z, z, targets, torch.full_like(exit, 25.0), exit) >= 91.44
    assert not early[0, 0, 1].any()
    assert torch.equal(late[0, 0, 1], torch.tensor([[False, True], [True, False]]))

    allowed = torch.ones(3, 3, 3, 2, 2, dtype=torch.bool)
    allowed[0, 0, 1, 0, 0] = False
    allowed[0, 1, 0, 0, 0] = False
    allowed[0, 0, 2, 1, :] = False
    allowed[0, 2, 0, :, 1] = False
    allowed[2, 0, 1] = allowed[2, 1, 0] = torch.tensor([[False, False], [False, True]])
    assert improve_layer_choices(allowed).tolist() == [[0, 1, 0], [0, 0, 0], [1, 1, 0]]

    preferred = allowed[1:2].clone()
    preferred[0, 0, 1, 0, 0] = preferred[0, 1, 0, 0, 0] = False
    assert improve_layer_choices(allowed[1:2], preferred=preferred).tolist() == [[0, 1, 0]]


def test_executed_actions_keep_unchanged_modes_and_encode_combined_commands():
    proposal = torch.tensor([[[0, 1, 0, 0, .3, 0, 0], [0, 0, 0, 1, 0, 0, .8], [0, 0, 0, 1, 0, 0, -.2]]])
    commands = torch.tensor([[[.3, 0, 0], [.1, -.7, 0], [0, 0, 0]]])
    expected = torch.tensor([[[0, 1, 0, 0, .3, 0, 0], [0, 1, 1, 0, .1, -.7, 0], [1, 0, 0, 0, 0, 0, 0]]])
    torch.testing.assert_close(encode_executed_action(proposal, commands), expected)


def test_accelerated_layer_clearance_detects_native_unequal_descent_contact():
    height = torch.tensor([[103.8628, 201.0156]], dtype=torch.float64)
    speed = torch.tensor([[0.0, -7.62]], dtype=torch.float64)
    target = torch.tensor([[[30.435], [185.341]]], dtype=torch.float64)
    entry = torch.zeros(1, 2, 2, dtype=torch.float64)
    exit = torch.full_like(entry, 6.0)
    linear = layer_clearance(height, speed, target, entry, exit)
    accelerated = layer_clearance(height, speed, target, entry, exit, accelerated=True)
    assert linear[0, 0, 1, 0, 0] > 91.44
    assert accelerated[0, 0, 1, 0, 0] < 91.44
    torch.testing.assert_close(accelerated, accelerated.transpose(1, 2))


def test_joint_layer_search_repairs_three_aircraft_without_trading_away_clearance():
    allowed = torch.ones(2, 4, 4, 2, 2, dtype=torch.bool)
    separation = torch.eye(2, dtype=torch.bool)
    only_second = torch.tensor([[False, False], [False, True]])
    for source, target, table in ((0, 1, only_second), (0, 2, separation), (1, 2, separation)):
        allowed[:, source, target] = table
        allowed[:, target, source] = table.T
    allowed[1, 2, 3, 1, :] = False
    allowed[1, 3, 2, :, 1] = False
    pairs = improve_layer_choices(allowed)
    assert pairs.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    components = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])
    joint = improve_three_aircraft_choices(allowed, pairs, components)
    assert joint.tolist() == [[1, 1, 1, 0], [0, 0, 0, 0]]
    reserve = torch.ones_like(allowed)
    reserve[0, 0, 2, 1, 1] = False
    reserve[0, 2, 0, 1, 1] = False
    assert torch.equal(improve_three_aircraft_choices(allowed, pairs, components, preserve=reserve), pairs)


def test_joint_layer_search_can_improve_margin_without_a_nominal_conflict():
    allowed = torch.ones(1, 3, 3, 2, 2, dtype=torch.bool)
    preferred = allowed.clone()
    for left, right, table in (
        (0, 1, torch.tensor([[False, False], [False, True]])),
        (0, 2, torch.eye(2, dtype=torch.bool)),
        (1, 2, torch.eye(2, dtype=torch.bool)),
    ):
        preferred[0, left, right], preferred[0, right, left] = table, table.T
    initial = improve_layer_choices(allowed, preferred=preferred)
    assert initial.tolist() == [[0, 0, 0]]
    component = torch.zeros_like(initial)
    assert improve_three_aircraft_choices(allowed, initial, component, preferred=preferred).tolist() == [[1, 1, 1]]
    preserve = allowed.clone()
    preserve[0, 0, 2, 1, 1] = preserve[0, 2, 0, 1, 1] = False
    assert torch.equal(
        improve_three_aircraft_choices(allowed, initial, component, preferred=preferred, preserve=preserve),
        initial,
    )


def test_native_vertical_paths_accelerate_on_grid_and_capture_the_selected_altitude():
    height = torch.tensor([100.0, 100.0], dtype=torch.float64)
    vertical = torch.tensor([0.0, -7.62], dtype=torch.float64)
    target = torch.tensor([100.0, 100.25], dtype=torch.float64)
    commands = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    actual = altitude_paths(height, vertical, target, commands, 3)
    expected = torch.tensor([[[6196.0, 6196.003810, 6196.011430, 6196.022860]],
                             [[6196.0, 6196.25, 6196.25, 6196.25]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


def test_equivariant_actor_reflects_heading_and_edge_duals():
    torch.manual_seed(7)
    actor = Actor(width=16, equivariant_actor=True)
    for head in (actor.consensus, actor.altitude_consensus):
        torch.nn.init.normal_(head.weight, std=0.1)
    nodes, edges = torch.randn(2, 3, 24), torch.randn(2, 3, 3, 18)
    active = torch.tensor([[True, True, True], [True, True, False]])
    adjacency = (~torch.eye(3, dtype=torch.bool)).expand(2, -1, -1)
    mirror_nodes, mirror_edges = mirror_observations(nodes, edges)
    original = actor(nodes, edges, adjacency, active, deterministic=True)
    reflection = actor(mirror_nodes, mirror_edges, adjacency, active, deterministic=True)
    for key in ("logits", "mode", "speed", "altitude", "rho", "mobility", "altitude_rho"):
        torch.testing.assert_close(getattr(original, key), getattr(reflection, key))
    for key in ("heading", "dual", "altitude_dual"):
        torch.testing.assert_close(getattr(original, key), -getattr(reflection, key))
    torch.testing.assert_close(original.dual, original.dual.transpose(-2, -3).flip(-1))
    for value, mirrored in zip(original.consensus_inputs, reflection.consensus_inputs, strict=True):
        torch.testing.assert_close(value, mirrored, rtol=0., atol=0.)
    speed_dual, _, _, altitude_dual, _ = original.consensus_inputs
    for value in (speed_dual, altitude_dual):
        assert value.count_nonzero()
        torch.testing.assert_close(value, value.transpose(-2, -3).flip(-1))
    assert not torch.equal(speed_dual, original.dual)
    assert not torch.equal(altitude_dual, original.altitude_dual)
    (original.heading.square().sum() + original.speed.square().sum()).backward()
    assert torch.isfinite(actor.encoder.embed_nodes.weight.grad).all()


def test_joint_clearance_reserve_preserves_near_term_gap_and_future_buffer():
    clearance = torch.zeros((3, 2, 2, 2, 2))
    clearance[:, 0, 1, 0, 0] = torch.tensor([134.0, 183.0, 0.65])
    clearance[:, 0, 1, 1, 1] = torch.tensor([119.0, 91.47, 91.44])
    clearance[:, 1, 0] = clearance[:, 0, 1].transpose(-1, -2)
    overlaps = (~torch.eye(2, dtype=torch.bool)).expand(3, -1, -1)
    entry = torch.tensor([0.0, 104.0, 104.0]).reshape(-1, 1, 1).expand(3, 2, 2)
    choice = torch.zeros((3, 2), dtype=torch.long)
    allowed = _joint_clearance_reserve(clearance, overlaps, entry, choice, 10.0)
    assert allowed[:, 0, 1, 0, 0].all()
    assert allowed[:, 0, 1, 1, 1].tolist() == [False, False, True]


def test_coordination_loss_updates_the_admm_heads_without_changing_proposal_gradients():
    position = torch.tensor([[[-6000.0, 0.0, 30.0], [6000.0, 2200.0, 30.0], [0.0, -3500.0, 80.0]]])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    speed = torch.full((1, 3), 180.0)
    active = torch.ones_like(speed, dtype=torch.bool)
    adjacency = (~torch.eye(3, dtype=torch.bool)).unsqueeze(0)
    graph = torch.zeros((1, 3, 24)), torch.zeros((1, 3, 3, 18)), adjacency, active
    physical = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    actor = Actor(width=16)
    policy = actor(*graph, deterministic=True, detach_coordination=True)
    tuning = policy.consensus_inputs
    loss = coordination_loss(encode_action(policy)[..., 4:], physical.tensors(), adjacency, active, tuning, 2, 2)
    loss.backward()
    heads = (actor.consensus, actor.altitude_consensus, actor.willingness)
    gradients = [param.grad for head in heads for param in head.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().max() > 0 for value in gradients)
    assert all(param.grad is None for head in (actor.router, actor.speed, actor.altitude, actor.heading, actor.encoder)
               for param in head.parameters())


def test_layer_search_cannot_buy_a_resolution_by_losing_imminent_clearance():
    allowed = torch.ones((1, 3, 3, 2, 2), dtype=torch.bool)
    allowed[0, 0, 1] = False
    allowed[0, 1, 0] = False
    allowed[0, 0, 2, 0] = False
    allowed[0, 2, 0, :, 0] = False
    reserve = torch.ones_like(allowed)
    reserve[0, 0, 1, 1] = False
    reserve[0, 1, 0, :, 1] = False
    assert improve_layer_choices(allowed)[0, 0] == 1
    assert improve_layer_choices(allowed, preserve=reserve)[0, 0] == 0


def test_collision_target_and_replay_keep_per_aircraft_terminal_labels():
    collision = torch.tensor([[True, False, False, False]])
    bootstrap = torch.tensor([[False, False, True, True]])
    risk = _collision_target(collision, bootstrap, torch.tensor([[.2, .8, .3, .9]]),
                             torch.tensor([[.9, .5, .8, .4]]), .9)
    torch.testing.assert_close(risk, torch.tensor([[1., 0., .72, .81]]))

    counts = torch.zeros(1, dtype=torch.long)
    graph = (counts,) * 4
    data = Rollout(graph, counts, counts, graph, ~bootstrap, counts, counts, (counts,), counts,
                   collision_aircraft=collision)
    replay = DeviceReplay(2, "cpu", 11, collision_labels=True)
    replay.append(data)
    sample = replay.sample(1)
    assert torch.equal(sample.collision_aircraft, collision)
    assert len(sample.physical) == 1


def test_observed_collision_horizon_includes_endpoints_and_stops_at_identity_boundaries():
    def transition(active, terminal, collision):
        mask = torch.tensor([active], dtype=torch.bool)
        graph = (mask, mask, mask, mask)
        counts = torch.zeros(1, dtype=torch.long)
        return Rollout(graph, counts, counts, graph, torch.tensor([terminal]), counts, counts, (counts,), counts,
                       collision_aircraft=torch.tensor([collision]), route_count=counts + 6)

    horizon = CollisionHorizon(3)
    assert horizon.append(transition([1, 1, 1, 0], [False, True, False, False],
                                     [False, False, False, False])) is None
    assert horizon.append(transition([1, 1, 1, 1], [False, True, False, False],
                                     [False, True, False, False])) is None
    first = horizon.append(transition([1, 0, 1, 1], [True, False, False, True],
                                       [True, False, False, True]))
    assert first.observed_collision.tolist() == [[True, False, False, False]]

    second = horizon.append(transition([1, 0, 1, 0], [True, False, True, False],
                                        [True, False, True, False]))
    assert second.observed_collision.tolist() == [[True, True, True, True]]
    replay = DeviceReplay(2, "cpu", 17, collision_labels=True, observed_collision_labels=True, route_counts=True)
    replay.append(first)
    replay.append(second)
    restored = replay._take(torch.tensor([0, 1]))
    assert restored.observed_collision.tolist() == [[True, False, False, False], [True, True, True, True]]
    assert restored.collision_aircraft.tolist() == [[False, False, False, False], [False, True, False, False]]
    assert restored.route_count.tolist() == [6, 6]
    assert len(restored.physical) == 1


def test_multi_step_return_does_not_use_rewards_from_a_restarted_world():
    def transition(rewards, terminal, collision, marker):
        value = torch.tensor([[marker, marker]])
        graph = (value,) * 4
        counts = torch.zeros(1, dtype=torch.long)
        return Rollout(graph, value, torch.tensor([rewards]), graph, torch.tensor([terminal]), value, value,
                       (value,), counts, collision_aircraft=torch.tensor([collision]),
                       route_count=counts + 6, unprotected=counts == 0, next_physical=(value + 1,))

    data = (
        transition([1., 2.], [False, False], [False, False], 0),
        transition([-50., 4.], [True, False], [True, False], 1),
        transition([500., 8.], [False, False], [False, False], 2),
    )
    delayed = MultiStepReturns(3, 0.5, aggregate=False)
    aggregated = MultiStepReturns(3, 0.5, aggregate=True, collision_returns=True)
    for step in data[:-1]:
        assert delayed.append(step) is None and aggregated.append(step) is None
    assert delayed.append(data[-1]) is data[0]
    observed = aggregated.append(data[-1])
    assert observed.reward.tolist() == [[-24., 6.]]
    assert observed.terminal.tolist() == [[True, False]]
    assert observed.collision_return.tolist() == [[0.5, 0.]]
    assert observed.next_graph is data[-1].next_graph
    assert observed.next_physical is data[-1].next_physical
    assert data[0].terminal.tolist() == [[False, False]]
    replay = DeviceReplay(2, "cpu", 14, collision_labels=True, route_counts=True, collision_returns=True,
                          protection_labels=True, successor_geometry=True)
    replay.append(observed)
    saved = replay.sample(1)
    assert saved.collision_return.tolist() == [[0.5, 0.]]
    assert saved.collision_aircraft.tolist() == [[False, False]] and saved.route_count.tolist() == [6]
    assert saved.unprotected.tolist() == [True]
    assert saved.physical[0].tolist() == [[0, 0]] and saved.next_physical[0].tolist() == [[3, 3]]


def test_critic_bootstrap_projects_protected_worlds_only():
    learner = TD3("cpu", width=32)
    active = torch.ones(2, 1, dtype=torch.bool)
    adjacency = torch.zeros(2, 1, 1, dtype=torch.bool)
    graph = (torch.zeros(2, 1, 24), torch.zeros(2, 1, 1, 18), adjacency, active)
    future = (torch.ones_like(graph[0]), *graph[1:])
    successor = (torch.ones(2, 1),)
    batch = Rollout(graph, torch.zeros(2, 1, 7), torch.zeros(2, 1), future, ~active,
                    torch.zeros(2, 1, 3), ~active, (torch.zeros(2, 1),), torch.zeros(2, dtype=torch.long),
                    unprotected=torch.tensor([False, True]), next_physical=successor)
    candidate, received = [], []

    def project(proposal, physical, edges, mask, consensus):
        assert physical is successor and edges is future[2] and mask is future[3]
        command = torch.zeros_like(proposal[..., 4:])
        command[..., 2] = 0.75
        expected = encode_executed_action(proposal, command)
        expected[1] = proposal[1]
        candidate.append(expected)
        return command, mask

    def capture(_critic, inputs):
        received.append(inputs[-1])

    with learner.target_critic.register_forward_pre_hook(capture):
        learner.update(batch, target_projector=project)
    torch.testing.assert_close(received[0], candidate[0], rtol=0, atol=0)


@pytest.mark.parametrize(("categories", "head_on", "expected"),
                         [((1, 2), False, [False, True, True]), ((), True, [False, True, False]),
                          ((), False, None)])
def test_learner_applies_anchor_to_requested_traffic_only(monkeypatch, categories, head_on, expected):
    actor = Actor(width=32)
    learner = TD3("cpu", width=32, anchor_state=actor.state_dict(), anchor_weight=.65,
                  head_on_anchor=head_on, anchor_categories=categories)
    learner.updates = 1
    active = torch.ones(3, 1, dtype=torch.bool)
    graph = (torch.zeros(3, 1, 24), torch.zeros(3, 1, 1, 18),
             torch.zeros(3, 1, 1, dtype=torch.bool), active)
    graph[0][..., 8] = 1
    batch = Rollout(graph, torch.zeros(3, 1, 7), torch.zeros(3, 1), graph, ~active,
                    torch.zeros(3, 1, 3), ~active, (torch.zeros(3, 1),), torch.arange(3))
    original = td3._anchor_loss
    masks = []

    def capture(policy, reference, observation, airspaces):
        masks.append(airspaces.tolist() if airspaces is not None else None)
        return original(policy, reference, observation, airspaces)

    monkeypatch.setattr(td3, "_anchor_loss", capture)
    learner.update(batch)
    assert masks == [expected]


def test_topology_replay_preserves_family_budgets_and_head_on_draws():
    category = torch.arange(3).repeat_interleave(32)
    ids = torch.arange(len(category))
    routes = torch.full_like(category, 3)
    routes[0] = 6
    routes[category == 1] = 2
    graph = (ids,) * 4
    rollout = Rollout(graph, ids, ids, graph, ids, ids, ids, (ids,), category, route_count=routes)
    baseline = DeviceReplay(96, "cpu", 71, stratified=True, route_counts=True)
    candidate = DeviceReplay(96, "cpu", 71, stratified=True, topology_replay=True)
    for buffer in (baseline, candidate):
        buffer.append(rollout)
    original = baseline.sample(64)
    sampled = candidate.sample(64)
    assert sampled.category.bincount().tolist() == [26, 19, 19]
    assert (sampled.action[20:26] == 0).all()
    assert (sampled.route_count[20:26] == 6).all()
    assert torch.equal(sampled.action[26:], original.action[26:])


def test_projection_mixture_changes_only_finished_worlds_and_leaves_global_rng_alone():
    previous = torch.tensor([True, False, True, False])
    restarted = torch.tensor([False, False, True, True])
    generator = torch.Generator().manual_seed(19)
    global_rng = torch.get_rng_state()
    assert _unprotected_worlds(previous, restarted, 0.0, generator).tolist() == [True, False, False, False]
    assert _unprotected_worlds(previous, restarted, 1.0, generator).tolist() == [True, False, True, True]
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_terminal_replay_preserves_actor_samples_and_family_budgets():
    category = torch.arange(3).repeat_interleave(20)
    events = torch.zeros_like(category)
    events[torch.tensor([0, 20, 40])] = 1
    events[torch.tensor([1, 41])] = 2
    ids = torch.arange(len(category))
    graph = (ids,) * 4
    rollout = Rollout(graph, ids, ids, graph, ids, ids, ids, (ids,), category, events)
    ordinary = DeviceReplay(60, "cpu", 71, stratified=True)
    enriched = DeviceReplay(60, "cpu", 71, stratified=True, terminal_replay=True, protection_labels=True)
    unprotected = ids % 2 == 1
    ordinary.append(rollout)
    enriched.append(replace(rollout, unprotected=unprotected))

    for update_actor in (False, True, False, True):
        usual = ordinary.sample(64)
        critic, actor = enriched.sample_for_learning(64, update_actor=update_actor)
        assert torch.equal(critic.category, usual.category)
        assert critic.category.bincount().tolist() == [26, 19, 19]
        assert critic.events[:6].tolist() == [1, 1, 1, 2, 2, 2]
        assert critic.events[26:28].tolist() == [1, 1]
        assert torch.equal(critic.action[28:30], usual.action[28:30])
        assert critic.events[45:49].tolist() == [1, 1, 2, 2]
        assert torch.equal(critic.events, events[critic.action])
        assert torch.equal(critic.physical[0], critic.action)
        assert torch.equal(critic.unprotected, unprotected[critic.action])
        assert critic.unprotected[critic.events == 2][:3].all()
        assert usual.unprotected is None
        if update_actor:
            assert torch.equal(actor.action, usual.action)
            assert torch.equal(actor.events, usual.events)
            assert torch.equal(actor.unprotected, unprotected[actor.action])
        else:
            assert actor is None


def test_early_plan_uses_single_axis_with_the_heading_deadband():
    raw = torch.tensor([[.8, .5, .2], [.8, .5, .04], [.8, 0, .04]])
    expected = torch.tensor([[0, 0, .2], [0, .5, 0], [.8, 0, 0]])
    torch.testing.assert_close(single_axis_commands(raw), expected)


def test_native_collision_trace_combines_vertical_and_overlap_diagnostics():
    position = torch.tensor([[[-5000.0, 0.0, 182.88], [5000.0, 0.0, 182.88]]])
    direction = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    speed = torch.full((1, 2), 210.0)
    state = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    active = torch.ones_like(speed, dtype=torch.bool)
    adjacency = ~torch.eye(2, dtype=torch.bool).unsqueeze(0)
    controls = torch.zeros_like(position)
    contact = {"env": 0, "pair": [0, 1], "initial_vertical_m": 0.0, "vertical_rates_mps": [0.0, 0.0]}

    _classify(state.tensors(), active, adjacency, controls, (None, 8.0, None, None, 8.0), [contact],
              altitude_oracle=True, native_vertical=True, risk_controls=controls)

    assert contact["observed_with_35km"]
    assert contact["observed_with_50km"]
    assert contact["nominal_conflict_within_50km"]
    assert contact["nominal_horizontal_overlap_within_50km"]
    assert not contact["potential_conflict_within_50km"]
    assert "status" in contact["discrete_altitude_with_inertia"]
    assert contact["overlap_risk"]["executed"]["60"]["horizontal_min_m"] < 1.0


def test_partial_layer_choices_do_not_create_a_new_predicted_conflict():
    allowed = torch.ones(3, 3, 4, 4, dtype=torch.bool)
    allowed[0, 1] = False
    allowed[0, 1, 1, :] = True
    allowed[0, 1, :, 1] = True
    allowed[0, 2, 1, :] = False

    assert _partial_layer_choices(allowed.tolist(), [0, 1, 2]) == {0: 0, 1: 1, 2: 0}


def test_stage_acceptance_rejects_discrete_layer_diagnostics():
    with pytest.raises(ValueError, match="standard projection"):
        assess({"diagnostic_projection": True})


@pytest.mark.parametrize("mpc", [False, True])
def test_exact_refinement_requires_complete_paired_native_evidence(tmp_path, mpc):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    digest = _checksum(checkpoint)
    source = dict(sha256=digest, observation_profile=dict(detection_metres=35000.), equivariant_actor=True,
                  admm_warmstart_reflection="even")
    report = EXACT_NATIVE_PROTOCOL | {"sources": {"candidate_1": source}, "families": {}}
    if mpc:
        report.update(MPC_COMMUNICATION)
    arms = dict(baseline=10, candidate_1=12, candidate_2=11, baseline_without_projection=9,
                candidate_1_without_projection=11, candidate_2_without_projection=9)
    for family in FAMILIES:
        report["families"][family] = {
            arm: dict(admm_k=0 if arm.endswith("without_projection") else 2, admm_fallback=0,
                      rows=[dict(seed=seed, aircraft=12, arrivals=arrivals, collisions=12-arrivals,
                                 exits=0, unresolved=0, action_cost_per_aircraft=1.)
                            for seed in range(12001, 12006)])
            for arm, arrivals in arms.items()
        }
    comparison = tmp_path / "comparison.json"
    acceptance = tmp_path / "acceptance.json"
    comparison.write_text(json.dumps(report))
    result = exact_refinement_acceptance(comparison, mpc=mpc)
    with pytest.raises(ValueError, match="frozen exact safety profile"):
        exact_refinement_acceptance(comparison, mpc=not mpc)
    assert result["paper_evaluation_ready"] and result["passed"]
    acceptance.write_text(json.dumps(result))
    assert paper_checkpoint_digest(acceptance, checkpoint) == digest

    result["totals"]["candidate_1"]["arrivals"] = 361
    acceptance.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="release gate"):
        paper_checkpoint_digest(acceptance, checkpoint)

    group = report["families"]["crossing6"]["candidate_1_without_projection"]
    group["admm_k"] = 2
    comparison.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ADMM budget"):
        exact_refinement_acceptance(comparison, mpc=mpc)

    group["admm_k"] = 0
    report["event_sampling_seconds"] = 6.0
    comparison.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="frozen exact safety profile"):
        exact_refinement_acceptance(comparison, mpc=mpc)


def test_cross_track_is_signed_and_penalty_applies_only_to_free_yaw_aircraft():
    latitude = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)
    longitude = torch.tensor([0.01, -0.01, 0.0], dtype=torch.float64)
    origin = torch.zeros_like(latitude)
    offset = cross_track_km(latitude, longitude, torch.ones_like(latitude), origin, origin, origin)
    torch.testing.assert_close(offset, torch.tensor([1.11194, -1.11194, 0.0], dtype=torch.float64), atol=2e-5, rtol=0)

    nodes = torch.zeros(2, 5, 24)
    nodes[0, 2, 8] = 1
    future_nodes = nodes.clone()
    future_nodes[..., 9] = 0.65
    adjacency = torch.zeros(2, 5, 5, dtype=torch.bool)
    adjacency[0, 1, 4] = adjacency[0, 4, 1] = True
    active = torch.ones(2, 5, dtype=torch.bool)
    terminal = torch.zeros_like(active)
    terminal[0, 3] = True
    graph = (nodes, None, adjacency, active)
    future = (future_nodes, None, adjacency, active)
    penalty = _cross_track_penalty(graph, future, terminal, torch.tensor([2, 0]))
    torch.testing.assert_close(penalty, torch.tensor([[0.25, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))


def test_far_observation_keeps_protection_graph_and_ignores_inactive_aircraft():
    position = torch.tensor([[[0.0, 0, 0], [30_000.0, 0, 100], [31_000.0, 0, 200], [0, 8000.0, 0]]])
    direction = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0]]])
    speed = torch.full((1, 4), 200.0)
    active = torch.tensor([[True, True, False, True]])
    adjacency = torch.zeros(1, 4, 4, dtype=torch.bool)
    adjacency[0, 0, 3] = adjacency[0, 3, 0] = True
    nodes = torch.zeros(1, 4, 24)
    edges = torch.zeros(1, 4, 4, 18)
    graph = (nodes, edges, adjacency, active)

    new = with_far_horizontal_observation(graph, (position, direction, speed), 20_000.0)

    assert new[0] is nodes and new[1] is edges and new[3] is active
    assert not adjacency[0, 0, 1] and new[2][0, 0, 1] and new[2][0, 1, 0]
    assert new[2][0, 0, 3] and new[2][0, 3, 0]
    assert not new[2][0, 2].any() and not new[2][0, :, 2].any()


def test_far_conflict_features_choose_reachable_layer_and_preserve_migrated_actions():
    longitude = torch.tensor([[-0.12, 0.12, 0.09, 0.12]])
    latitude = torch.zeros_like(longitude)
    altitude = torch.tensor([[6096.0, 6096.0, 6370.32, 6096.0]])
    active = torch.tensor([[True, True, True, False]])
    options = dict(latitude=latitude, longitude=longitude, altitude=altitude,
                   heading=torch.tensor([[90., 270., 270., 270.]]),
                   east=torch.tensor([[170., -170., -170., -170.]]), north=latitude, vertical=latitude,
                   goal_latitude=latitude, goal_longitude=-longitude, planned_altitude=altitude,
                   lnav=active, active=active, center_latitude=0., center_longitude=0., airspace_half_km=60.)
    original = traffic_graph(**options)
    augmented = traffic_graph(**options, far_conflict_observation=True)
    assert torch.equal(original[2], augmented[2]) and not augmented[2][0, 0].any()
    torch.testing.assert_close(augmented[0][0, 0, FAR_CONFLICT_FEATURES][:4],
                               torch.cat((augmented[1][0, 0, 1, :2], augmented[1][0, 0, 1, 3:5] / 4)))
    assert augmented[0][0, 0, 19] > 0
    assert not augmented[0][0, 2:, FAR_CONFLICT_FEATURES].count_nonzero()

    actor = Actor(width=32).eval()
    optimizer = torch.optim.AdamW(actor.parameters())
    weight = actor.encoder.embed_nodes.weight
    momentum = torch.ones_like(weight)
    optimizer.state[weight]["exp_avg"] = momentum
    before = actor(*original)
    _reset_far_conflict_weights(actor, optimizer)
    after = actor(*augmented)
    torch.testing.assert_close(before.logits, after.logits, atol=0, rtol=0)
    torch.testing.assert_close(before.altitude, after.altitude, atol=0, rtol=0)
    assert not momentum[:, FAR_CONFLICT_FEATURES].count_nonzero()
    assert (momentum[:, :15] == 1).all()


def test_stage2_selection_accepts_rates_at_floors_and_rejects_below():
    score = dict(head_on=57, aircraft_by_family={"head_on": 60, "yaw_plus15": 60},
                 arrivals_by_family={"head_on": 57, "yaw_plus15": 53})
    assert _meets_development_floors(score)
    assert not _meets_development_floors(score | {"head_on": 56})
    assert not _meets_development_floors(score | {"arrivals_by_family": {"head_on": 57, "yaw_plus15": 52}})


def test_metrics_append_retries_partial_write_without_truncating_completed_records(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"decisions":600}\n')

    class PartiallyWritablePath:
        attempts = 0

        @contextmanager
        def open(self, *args, **kwargs):
            self.attempts += 1
            with metrics.open(*args, **kwargs) as handle:
                if self.attempts == 1:
                    def write_prefix(data):
                        return handle.write(data[:5])
                    yield SimpleNamespace(write=write_prefix, tell=handle.tell)
                else:
                    yield handle

    path = PartiallyWritablePath()
    delays = []
    monkeypatch.setattr(bluesky_td3, "sleep", delays.append)
    bluesky_td3._append_metrics(path, '{"decisions":650}')
    assert path.attempts == 2 and delays == [2]
    assert metrics.read_text().splitlines() == ['{"decisions":600}', '{"decisions":650}']


def test_selected_altitude_features_separate_autopilot_target_from_route_and_preserve_plan_b_policy():
    longitude = torch.tensor([[-0.02, 0.02, 0.21]], dtype=torch.float64)
    latitude = torch.zeros_like(longitude)
    altitude = torch.full_like(longitude, 6096.0)
    selected = altitude + torch.tensor([[91.44, 0.0, 0.0]], dtype=altitude.dtype)
    active = torch.tensor([[True, True, False]])
    options = dict(latitude=latitude, longitude=longitude, altitude=altitude,
                   heading=torch.tensor([[90.0, 270.0, 90.0]]),
                   east=torch.tensor([[170.0, -170.0, 0.0]]), north=latitude,
                   vertical=torch.tensor([[3.81, 0.0, 0.0]]), goal_latitude=latitude, goal_longitude=-longitude,
                   lnav=active, active=active, center_latitude=0.0, center_longitude=0.0, airspace_half_km=60.0)
    old_plan_b = traffic_graph(**options, planned_altitude=selected, selected_altitude=selected)
    new = traffic_graph(**options, planned_altitude=altitude, selected_altitude=selected,
                        selected_altitude_observation=True)
    assert new[2][0, 1, 0] and torch.equal(new[2], old_plan_b[2])
    torch.testing.assert_close(new[0][..., 3], torch.zeros_like(new[0][..., 3]))
    torch.testing.assert_close(new[0][..., SELECTED_ALTITUDE_NODE_FEATURE], -old_plan_b[0][..., 3])
    target, rate = SELECTED_ALTITUDE_EDGE_FEATURES
    torch.testing.assert_close(new[1][0, 1, 0, [target, rate]], torch.tensor([1.0, 0.5]))
    assert not new[1][0, :, 2, [target, rate]].count_nonzero()

    actor = Actor(width=32).eval()
    optimizer = torch.optim.AdamW(actor.parameters())
    weights = actor.encoder.embed_nodes.weight
    optimizer.state[weights]["exp_avg"] = torch.ones_like(weights)
    with torch.no_grad():
        original = actor(*old_plan_b)
        _initialize_selected_altitude_weights(actor, optimizer, from_plan_b=True)
        migrated = actor(*new)
    torch.testing.assert_close(migrated.logits, original.logits)
    torch.testing.assert_close(encode_action(migrated), encode_action(original))
    assert not optimizer.state[weights]["exp_avg"][:, (3, SELECTED_ALTITUDE_NODE_FEATURE)].count_nonzero()


def test_correlated_yaw_changes_only_yaw_signs_in_half_of_yaw_worlds():
    def routes(correlated_yaw):
        return sample_routes("full_mixture", 4096, 20, 45.0, 0.0, 60.0,
                             torch.Generator().manual_seed(92197), "cpu",
                             timed_departures=True, correlated_yaw=correlated_yaw)

    regular, coupled = routes(False), routes(True)
    for name in vars(regular).keys() - {"yaw_deg"}:
        assert torch.equal(getattr(regular, name), getattr(coupled, name))
    assert torch.equal(regular.yaw_deg.abs(), coupled.yaw_deg.abs())
    yaw_world = regular.category == 2
    assert torch.equal(regular.yaw_deg[~yaw_world], coupled.yaw_deg[~yaw_world])
    active = coupled.active[yaw_world]
    positive = coupled.yaw_deg[yaw_world] > 0
    aligned = ((positive == positive[:, :1]) | ~active).all(dim=-1)
    assert 0.46 < aligned.float().mean() < 0.54
    assert 0.44 < positive[aligned, 0].float().mean() < 0.56


def test_teacher_command_mask_keeps_unlabeled_aircraft_out_of_loss(tmp_path):
    nodes = torch.zeros(1, 2, 24)
    edges = torch.zeros(1, 2, 2, 18)
    adjacency = ~torch.eye(2, dtype=torch.bool).unsqueeze(0)
    active = torch.ones(1, 2, dtype=torch.bool)
    commands = torch.tensor([[[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]])
    example = {"graph": (nodes, edges, adjacency, active), "commands": commands}
    legacy, masked = tmp_path / "legacy.pt", tmp_path / "masked.pt"
    label_mask = torch.tensor([[True, False]])
    torch.save(example, legacy)
    torch.save(dict(example, command_mask=label_mask), masked)

    old_data, new_data = load_teacher_data((legacy, masked), "cpu")
    assert torch.equal(old_data[2], active)
    assert torch.equal(new_data[2], label_mask)
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    proposal = SimpleNamespace(logits=logits, speed=torch.zeros(1, 2, 2), altitude=torch.zeros(1, 2),
                               heading=torch.zeros(1, 2))
    _teacher_loss(proposal, new_data[1], new_data[2]).backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert torch.equal(logits.grad[0, 1], torch.zeros(4))

    torch.save(dict(example, command_mask=~label_mask), masked)
    with pytest.raises(ValueError, match="command mask"):
        load_teacher_data((masked,), "cpu")


def test_rollout_router_temperature_changes_hard_samples_only():
    actor = Actor(width=32)
    with torch.no_grad():
        actor.router.weight.zero_()
        actor.router.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
        worlds = 512
        graph = (torch.zeros(worlds, 1, 24), torch.zeros(worlds, 1, 1, 18),
                 torch.zeros(worlds, 1, 1, dtype=torch.bool), torch.ones(worlds, 1, dtype=torch.bool))
        with torch.random.fork_rng():
            torch.manual_seed(1823)
            ordinary = actor(*graph, temperature=0.35).mode[..., 0]
            torch.manual_seed(1823)
            reduced = actor(*graph, temperature=0.35, sampling_temperature=0.5).mode[..., 0]
        assert ordinary.sum() < worlds * 0.80
        assert reduced.sum() > worlds * 0.90
        assert actor.eval()(*graph, sampling_temperature=0.5).mode[..., 0].all()


def test_crossing_exploration_persists_until_window_or_aircraft_end():
    active = torch.ones(3, 2, dtype=torch.bool)
    category = torch.arange(3)
    exploration = td3.CrossingExploration(active, 3, 115)
    mask, mode, controls = exploration.draw(active, category)
    assert mask.tolist() == [[True, True], [False, False], [False, False]]
    exploration.end(torch.tensor([[False, True], [False, False], [False, False]]))
    _, next_mode, next_controls = exploration.draw(active, category)
    torch.testing.assert_close(next_mode[0, 0], mode[0, 0])
    torch.testing.assert_close(next_controls[0, 0], controls[0, 0])
    assert not torch.equal(next_mode[0, 1], mode[0, 1])
    _, third_mode, _ = exploration.draw(active, category)
    torch.testing.assert_close(third_mode, next_mode)
    _, fourth_mode, _ = exploration.draw(active, category)
    assert not torch.equal(fourth_mode[0, 0], third_mode[0, 0])
    torch.testing.assert_close(fourth_mode[0, 1], third_mode[0, 1])

    learner = TD3("cpu", width=32)
    graph = (torch.zeros(3, 2, 24), torch.zeros(3, 2, 2, 18), torch.zeros(3, 2, 2, dtype=torch.bool), active)
    with torch.random.fork_rng():
        torch.manual_seed(98)
        baseline = learner.proposal_and_consensus(graph, explore=True)[0]
        torch.manual_seed(98)
        variant = learner.proposal_and_consensus(graph, explore=True, crossing_exploration=(mask, mode, controls))[0]
    assert torch.equal(baseline[1:], variant[1:])
    assert not torch.equal(baseline[0], variant[0])


def test_projected_ste_forward_and_learner_anchor_warmup():
    proposal = torch.tensor([1., 2.], requires_grad=True)
    executed = torch.tensor([-2., 5.], requires_grad=True)
    result = td3.projected_action_ste(proposal, executed)
    assert torch.equal(result, executed)
    result.square().sum().backward()
    torch.testing.assert_close(proposal.grad, 2 * executed.detach())
    assert executed.grad is None

    learner = TD3("cpu", width=32, anchor_weight=.65)
    learner.anchor_start_update = 50_000
    for elapsed, expected in ((0, .65), (1000, .325), (2000, 0.), (5000, 0.)):
        learner.updates = 50_000 + elapsed
        assert learner.current_anchor_weight == expected


def test_mirrored_replay_maps_heading_and_local_lateral_geometry():
    nodes, edges = torch.rand(1, 2, 24), torch.rand(1, 2, 2, 18)
    active = torch.ones(1, 2, dtype=torch.bool)
    adjacency = torch.tensor([[[False, True], [True, False]]])
    graph = (nodes, edges, adjacency, active)
    position, direction = torch.rand(1, 2, 3), torch.rand(1, 2, 2)
    scalar = torch.ones(1, 2)
    physical = (position, direction, *(scalar for _ in range(5)))
    action, teacher = torch.rand(1, 2, 7), torch.rand(1, 2, 3)
    batch = Rollout(graph, action, scalar, graph, active, teacher, active, physical,
                    torch.ones(1, dtype=torch.long), next_physical=physical)
    mirrored = td3.mirror_replay_batch(batch)
    for original, paired in zip(batch.tensors(), mirrored.tensors(), strict=True):
        torch.testing.assert_close(original, paired[:1])
    torch.testing.assert_close(mirrored.graph[0][1, :, [5, 9, 16, 18]],
                               -nodes[0, :, [5, 9, 16, 18]])
    torch.testing.assert_close(mirrored.next_graph[1][1, ..., [1, 4]], -edges[0, ..., [1, 4]])
    torch.testing.assert_close(mirrored.next_physical[0][1, :, 0], -position[0, :, 0])
    torch.testing.assert_close(mirrored.physical[1][1, :, 0], -direction[0, :, 0])
    torch.testing.assert_close(mirrored.action[1, :, -1], -action[0, :, -1])
    torch.testing.assert_close(mirrored.teacher[1, :, -1], -teacher[0, :, -1])


def test_deterministic_teacher_forward_leaves_exploration_rng_unchanged():
    actor = Actor(width=32)
    graph = (torch.zeros(1, 2, 24), torch.zeros(1, 2, 2, 18),
             torch.zeros(1, 2, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    before = torch.random.get_rng_state()
    action = actor(*graph, deterministic=True)
    torch.nn.functional.cross_entropy(action.logits.flatten(0, 1), torch.ones(2, dtype=torch.long)).backward()
    assert torch.equal(torch.random.get_rng_state(), before)
    assert actor.router.bias.grad.abs().sum() > 0
    actor(*graph)
    assert not torch.equal(torch.random.get_rng_state(), before)


def test_geodesic_goal_observation_changes_only_target_angle():
    zero = torch.zeros(1, 1, dtype=torch.float64)
    latitude = zero + 45.0
    altitude = zero + 6150.0
    arguments = dict(latitude=latitude, longitude=zero, altitude=altitude, heading=zero, east=zero, north=zero + 180,
                     vertical=zero, goal_latitude=latitude, goal_longitude=zero + 90, planned_altitude=altitude,
                     lnav=zero.bool(), active=torch.ones_like(zero, dtype=torch.bool),
                     center_latitude=45.0, center_longitude=0.0, airspace_half_km=150)
    planar = traffic_graph(**arguments)
    measured = traffic_graph(**arguments, geodesic_navigation=True)
    torch.testing.assert_close(planar[0][..., :5], measured[0][..., :5])
    torch.testing.assert_close(planar[0][..., 7:], measured[0][..., 7:])
    for previous, current in zip(planar[1:], measured[1:], strict=True):
        torch.testing.assert_close(previous, current)
    angle = torch.atan2(measured[0][0, 0, 5], measured[0][0, 0, 6]).rad2deg()
    assert angle.item() == pytest.approx(54.7356, abs=1e-4)


@pytest.mark.parametrize("dimension", [2, 3])
def test_matches_original_six_cases_and_disjoint_windows(dimension):
    reference = Path(os.environ.get("GEO_CDR_REFERENCE", Path(__file__).resolve().parents[2] / "uav_cdr_python"))
    sys.path.insert(0, str(reference))
    from speed_adjustment.solver import _pair_clearance

    rng = np.random.default_rng(23)
    n = 768
    data = {key: rng.normal(0, 150, size=(n, dimension)) for key in ("p_io", "p_jo")}
    for key in ("n1", "n2"):
        d = rng.normal(size=(n, dimension))
        data[key] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    for key in ("D", "E"):
        data[key] = rng.uniform(10, 300, size=n)
    for key in ("t1_s", "t2_s"):
        data[key] = rng.uniform(0, 16, size=n)
    v_i, v_j = rng.uniform(5, 25, size=(2, n))
    target = [_pair_clearance({key: values[i] for key, values in data.items()}, v_i[i], v_j[i], 50.0)
              for i in range(n)]
    params = {key: torch.from_numpy(values) for key, values in data.items()}
    result = _pair_clearance_torch(params, torch.from_numpy(v_i), torch.from_numpy(v_j), 50.0)
    torch.testing.assert_close(result, torch.tensor(target, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    gradients = torch.autograd.gradcheck(
        lambda i, j: _pair_clearance_torch(params, i, j, 50.0),
        (torch.from_numpy(v_i).requires_grad_(), torch.from_numpy(v_j).requires_grad_()),
        fast_mode=True,
    )
    assert gradients


def test_cylinder_respects_vertical_overlap_and_tangent():
    delta = torch.tensor([[30.0, 0, 30], [30.0, 0, 30], [0, 20.0, 0]])
    velocity = torch.tensor([[-1.0, 0, -1], [-1.0, 0, 0], [1.0, 0, 0]])
    enter, leave, exists = cylinder_interval(delta, velocity, 20.0, 10.0, 60.0)
    assert exists.tolist() == [True, False, False]
    torch.testing.assert_close(enter, torch.tensor([20.0, 0, 0]))
    torch.testing.assert_close(leave, torch.tensor([40.0, 0, 0]))


def test_graph_symmetry_coloring_and_cross_layer_edges():
    points = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0], [0.0, 0, 120], [500.0, 0, 0]]])
    active = torch.ones(1, 4, dtype=torch.bool)
    graph = conflict_graph(points, torch.zeros_like(points), active, horizontal_radius=20.0,
                           detection=100.0, speed_slack=0.0)
    assert torch.equal(graph.adjacency, graph.adjacency.transpose(-1, -2))
    assert graph.existing[0, 0, 1] and graph.cross_layer[0, 0, 2]
    colors = greedy_colors(graph.adjacency)
    assert not (graph.adjacency & (colors.unsqueeze(-1) == colors.unsqueeze(-2))).any()
    assert connected_components(graph.adjacency).tolist() == [[0, 0, 0, 3]]


def test_speed_admm_audits_solved_and_impossible_pairs():
    points = torch.tensor([[[-210.0, 0], [0, -240.0]], [[-60.0, 0], [60.0, 0]]])
    directions = torch.tensor([[[1.0, 0], [0, 1.0]], [[1.0, 0], [-1.0, 0]]])
    speed = torch.full((2, 2), 12.0, requires_grad=True)
    active = torch.ones_like(speed, dtype=torch.bool)
    edges = torch.tensor([[[False, True], [True, False]]]).expand(2, -1, -1)
    bounds = dict(minimum_speed=4.0, maximum_speed=20.0, horizon=40.0, radius=50.0)
    assert not speed_admm(points, directions, speed, active, edges, iterations=0, **bounds).feasible.any()
    result = speed_admm(points, directions, speed, active, edges, **bounds)
    assert result.feasible.tolist() == [True, False]
    result.speed.square().sum().backward()
    assert torch.isfinite(speed.grad).all()


def test_altitude_verifier_keeps_neighbor_in_requested_layer():
    position = torch.tensor([[[-300.0, 0, 6000], [0, -300.0, 6000], [-300.0, 0, 6091.44]]],
                            dtype=torch.float64)
    horizontal_velocity = torch.tensor([[[10.0, 0], [0, 10], [10, 0]]], dtype=position.dtype)
    proposal = torch.tensor([[91.44, 0, 0]], dtype=position.dtype, requires_grad=True)
    bounds = dict(minimum_altitude=5900, maximum_altitude=6200, horizontal_radius=50)
    original_pair = torch.tensor([[True, True, False]])
    all_aircraft = torch.ones_like(original_pair)
    assert altitude_admm(position, horizontal_velocity, proposal, original_pair, iterations=0, **bounds).feasible
    assert not altitude_admm(position, horizontal_velocity, proposal, all_aircraft, iterations=0, **bounds).feasible
    result = altitude_admm(position, horizontal_velocity, proposal, all_aircraft, iterations=5, **bounds)
    assert result.minimum_clearance > -91.44 + 1e-5
    result.offset.square().sum().backward()
    assert torch.isfinite(proposal.grad).all()


def test_forbidden_speed_interval_agrees_with_direct_continuous_audit():
    rng = torch.Generator().manual_seed(25)
    position = torch.randn((1200, 3), dtype=torch.float64, generator=rng) * 120
    angle = torch.rand(1200, dtype=position.dtype, generator=rng) * (2 * torch.pi)
    heading = torch.stack((angle.cos(), angle.sin()), dim=-1)
    other = torch.randn((1200, 3), dtype=position.dtype, generator=rng) * position.new_tensor([8, 8, 1])
    speed = torch.linspace(5.001, 19.999, 401, dtype=position.dtype)
    lo, hi, exists = speed_forbidden_interval(
        position, heading, other[:, :2], -other[:, 2], lower=5.0, upper=20.0,
        horizontal_radius=50, vertical_radius=91.44, horizon=80,
    )
    horizontal = speed[None, :, None] * heading[:, None, :]
    ego = torch.cat((horizontal, torch.zeros_like(horizontal[..., :1])), dim=-1)
    _, _, audit = cylinder_interval(position[:, None, :], ego - other[:, None, :], 50, 91.44, 80)
    predicted = exists[:, None] & (speed > lo[:, None]) & (speed < hi[:, None])
    assert torch.equal(predicted, audit)


def test_forbidden_speed_initial_tangency_and_parallel_grazing():
    position = torch.tensor([[-50.0, 0, 0], [50.0, 0, 0], [-200.0, 50.0, 0]], dtype=torch.float64)
    heading = position.new_tensor([[1, 0]])
    other = position.new_tensor([[10, 0]])
    lo, hi, exists = speed_forbidden_interval(
        position, heading, other, position[:, 2], lower=5.0, upper=20.0,
        horizontal_radius=50, vertical_radius=91.44, horizon=80,
    )
    assert exists.tolist() == [True, True, False]
    assert lo.tolist() == [10, 5, 0]
    assert hi.tolist() == [20, 10, 0]


def test_colored_altitude_ties_resolve_interleaved_opposing_lanes():
    position = torch.tensor([[[-300.0, 0, 6000], [300, 40, 6000], [-300, 80, 6000], [300, 120, 6000]]])
    velocity = torch.tensor([[[10.0, 0], [-10, 0], [10, 0], [-10, 0]]])
    proposal = torch.zeros((1, 4))
    active = torch.ones((1, 4), dtype=torch.bool)
    bounds = dict(minimum_altitude=5900, maximum_altitude=6100, horizontal_radius=50, iterations=10)
    by_index = altitude_admm(position, velocity, proposal, active, **bounds)
    by_color = altitude_admm(position, velocity, proposal, active, colored_ties=True, **bounds)
    assert not by_index.feasible
    assert by_color.feasible


def test_heading_guard_keeps_isolated_aircraft_turn_during_a_conflict():
    position = torch.tensor([[[-12_000.0, 0, 182.88], [12_000.0, 0, 182.88], [0, 30_000.0, 182.88]]])
    direction = torch.tensor([[[1.0, 0], [-1, 0], [1, 0]]])
    speed = torch.full((1, 3), 180.0)
    active = torch.ones_like(speed, dtype=torch.bool)
    state = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    command = torch.tensor([[0.5, -0.5, 0.8]])
    result = _protect_heading(state, speed, command, active)
    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, 0.8]]))


def test_unresolvable_component_does_not_cancel_a_separate_feasible_plan():
    position = torch.tensor([[[-1000.0, 0, 182.88], [1000, 0, 182.88],
                              [-12000, 70000, 182.88], [12000, 70000, 182.88]]])
    direction = torch.tensor([[[1.0, 0], [-1, 0], [1, 0], [-1, 0]]])
    speed = torch.full((1, 4), 180.0)
    active = torch.ones_like(speed, dtype=torch.bool)
    state = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    actions = torch.zeros((1, 4, 7))
    adjacency = torch.zeros((1, 4, 4), dtype=torch.bool)
    _, scene_feasible = protected_command(actions, state.tensors(), adjacency, active, None, 2, 10)
    assert not scene_feasible.any()
    controls, feasible = protected_command(actions, state.tensors(), adjacency, active, None, 2, 10, by_component=True)
    assert feasible.tolist() == [[False, False, True, True]]
    assert (controls[0, :2] == 0).all()
    altitude_offsets = controls[0, 2:, 1] * 91.44
    assert abs(altitude_offsets[0] - altitude_offsets[1]) >= 91.44


def test_mixed_fleet_pads_variable_sizes_and_preserves_fixed_evaluation_size():
    rng = torch.Generator().manual_seed(9)
    mixed = sample_routes("mixed", 96, 30, 41.0, -92.5, 150.0, rng, "cpu")
    count = mixed.active.sum(dim=-1)
    assert count.amin() >= 6 and count.amax() <= 30
    assert count.unique().numel() > 10
    assert torch.equal(mixed.active, torch.arange(30) < count.unsqueeze(-1))
    fixed = sample_routes("crossing6", 3, 30, 41.0, -92.5, 150.0, rng, "cpu")
    assert fixed.active.all()


def test_heading_guard_checks_the_planned_altitude_envelope():
    position = torch.tensor([[[-12000.0, 0, 91.44], [12000, 0, 182.88]]])
    direction = torch.tensor([[[1.0, 0], [-1, 0]]])
    speed = torch.full((1, 2), 180.0)
    active = torch.ones_like(speed, dtype=torch.bool)
    vertical = torch.zeros_like(speed)
    state = PhysicalState(position, direction, speed, vertical, speed, speed, position[..., 2])
    heading = torch.tensor([[0.4, 0.5]])
    hold = _protect_heading(state, speed, heading, active, altitude_command=torch.zeros_like(speed))
    torch.testing.assert_close(hold, heading)
    converge = _protect_heading(state, speed, heading, active, altitude_command=torch.tensor([[0.1, 0.0]]))
    assert (converge == 0).all()


def test_directional_guard_keeps_isolated_and_widening_turns():
    point = torch.tensor([[[-10000.0, 0, 182.88], [10000, 2500, 182.88], [-11000, 75000, 182.88]]])
    direction = torch.tensor([[[1.0, 0.0], [-1, 0], [1, 0]]])
    speed = torch.full((1, 3), 180.0)
    zero = torch.zeros_like(speed)
    active = torch.ones_like(speed, dtype=torch.bool)
    state = PhysicalState(point, direction, speed, zero, speed, speed, point[..., 2])
    adjacency = torch.tensor([[[False, True, False], [True, False, False], [False, False, False]]])
    for sign in (1.0, -1.0):
        heading = torch.tensor([[sign, 0, -1.0]])
        output = _directional_heading(state, speed, heading, zero, adjacency, active)
        assert output[0, 0] == max(sign, 0)
        assert output[0, 2] == -1

    policy = SimpleNamespace(logits=torch.zeros(1, 3, 4), heading=zero.clone().requires_grad_())
    node = torch.zeros(1, 3, 24)
    node[..., 6:7] = 1
    node[..., 8] = torch.tensor([0.0, 1.0, 1.0])
    graph = (node, None, adjacency, active)
    for sign in (1.0, -1.0):
        node[0, 0, 5] = 0.176 * sign
        guidance = _navigation_loss(policy, graph, state.tensors(), 0.35)
        baseline = _navigation_loss(policy, graph, state.tensors(), 0.0)
        gradient, = torch.autograd.grad(guidance - baseline, policy.heading)
        if sign > 0:
            assert gradient[0, 0] < 0
        else:
            assert gradient[0, 0] == 0


def test_discrete_altitude_search_obeys_component_layer_and_time_constraints():
    position = torch.tensor([[[-10000.0, 0, 0], [10000, 0, 0], [0, -10000, 0]]])
    horizontal = torch.tensor([[[180.0, 0], [-180, 0], [0, 180]]])
    active = torch.ones(1, 3, dtype=torch.bool)
    for altitude, feasible in ((0.0, False), (91.44, True)):
        position[..., 2] = altitude
        overlap, allowed = _altitude_options(position, horizontal, position[..., 2], active)
        assert overlap.sum() == 6
        assignment, _ = _component_plan(allowed[0].tolist(), [0, 1, 2])
        assert (assignment is not None) == feasible
        if feasible:
            assert set(assignment.values()) == {0, 1, 2}
    position[..., :2] /= 5
    _, allowed = _altitude_options(position, horizontal, position[..., 2], active)
    assignment, _ = _component_plan(allowed[0].tolist(), [0, 1, 2])
    assert assignment is None
    valid = torch.ones((2, 2, 4, 4), dtype=torch.bool)
    assignment, _ = _component_plan(valid.tolist(), [0, 1], choices=[[3], [2]])
    assert assignment == {0: 3, 1: 2}


def test_vertical_replay_rejects_an_instantaneous_reversal_under_separation():
    position = torch.tensor([[[0.0, 0.0, 180.0], [1000.0, 0.0, 55.0]]])
    horizontal = torch.zeros(1, 2, 2)
    selected = torch.tensor([[90.0, 150.0]])
    active = torch.ones(1, 2, dtype=torch.bool)
    rates = torch.tensor([[-7.62, 7.62]])
    _, ramp = _altitude_options(position, horizontal, selected, active)
    _, inertia = _altitude_options(position, horizontal, selected, active, vertical_speed=rates)
    assert ramp[0, 0, 1, 2, 1]
    assert not inertia[0, 0, 1, 2, 1]


def test_lookahead_risk_checks_vertical_separation_at_horizontal_entry():
    position = torch.tensor([[[-5000.0, 0.0, 170.0], [5000.0, 0.0, 120.0]]])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    ground_speed = torch.full((1, 2), 180.0)
    state = PhysicalState(position, direction, ground_speed, torch.tensor([[2.0, 0.0]]),
                          ground_speed, ground_speed, torch.tensor([[250.0, 120.0]]))
    active = torch.ones(1, 2, dtype=torch.bool)
    action = torch.zeros(1, 2, 3, requires_grad=True)
    cpa = lookahead_risk(state.tensors(), active, action, dense=True)
    interval = lookahead_risk(state.tensors(), active, action, dense=True, overlap=True)
    assert (interval > cpa + 0.5).all()
    gradient, = torch.autograd.grad(interval.sum(), action)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 0].abs() > 0
    active[:, 1] = False
    assert lookahead_risk(state.tensors(), active, overlap=True).count_nonzero() == 0


def test_adaptive_altitude_rounds_only_apply_to_triangle_components():
    p = torch.tensor([[[-1000.0, 0, 182.88], [1000, 0, 182.88], [0, 2000, 182.88],
                      [-12000, 65000, 182.88], [12000, 65000, 182.88]]])
    v = torch.tensor([[[1.0, 0], [-1, 0], [0, -1], [180, 0], [-180, 0]]])
    active = torch.ones(1, 5, dtype=torch.bool)
    action = torch.zeros(1, 5)
    params = dict(minimum_altitude=0.0, maximum_altitude=304.8, iterations=2)
    base = altitude_admm(p, v, action, active, **params)
    adaptive = altitude_admm(p, v, action, active, extra_rounds=2, **params)
    assert adaptive.adaptive_nodes.tolist() == [[True, True, True, False, False]]
    torch.testing.assert_close(adaptive.offset[:, 3:], base.offset[:, 3:])


def test_mixture_proportions_and_lookahead_reward_mask():
    rng = torch.Generator().manual_seed(621)
    traffic = sample_routes("full_mixture", 4096, 20, 41.0, -92.5, 150.0, rng, "cpu")
    proportions = torch.bincount(traffic.category, minlength=3) / 4096
    torch.testing.assert_close(proportions, torch.tensor([0.4, 0.3, 0.3]), atol=0.025, rtol=0)
    assert traffic.yaw_deg[traffic.category == 2].abs().min() >= 1
    assert traffic.yaw_deg[traffic.category != 2].abs().max() == 0

    p = torch.tensor([[[-9000.0, 0, 182.88], [9000, 3000, 182.88], [500, 500, 182.88]]])
    d = torch.tensor([[[1.0, 0], [-1, 0], [1, 0]]])
    speed = torch.full((1, 3), 180.0)
    active = torch.tensor([[True, True, False]])
    state = PhysicalState(p, d, speed, torch.zeros_like(speed), speed, speed, p[..., 2])
    action = torch.zeros((1, 3, 3), requires_grad=True)
    risk = lookahead_risk(state.tensors(), active, action)
    assert risk[0, 0] > 0.8 and risk[0, 2] == 0
    risk.sum().backward()
    assert torch.isfinite(action.grad).all()


def test_scheduled_mixture_keeps_only_lane_leaders_at_time_zero():
    rng = torch.Generator().manual_seed(62)
    traffic = sample_routes("full_mixture", 64, 20, 41.0, -92.5, 150.0, rng, "cpu",
                            timed_departures=True)
    initial = (traffic.departure_seconds == 0) & traffic.active
    delayed = (traffic.departure_seconds > 0) & traffic.active
    assert (initial.sum(dim=-1) >= 2).all()
    assert (initial.sum(dim=-1) <= 8).all()
    assert (traffic.departure_seconds[delayed] >= 48).all()
    assert delayed.any()


def test_dense_risk_keeps_signal_from_a_second_conflicting_neighbor():
    p = torch.tensor([[[-9000.0, 0, 180.0], [9000, 1800, 180.0], [9000, -3800, 180.0]]])
    direction = torch.tensor([[[1.0, 0], [-1, 0], [-1, 0]]])
    speed = torch.full((1, 3), 180.0)
    active = torch.ones_like(speed, dtype=torch.bool)
    state = PhysicalState(p, direction, speed, torch.zeros_like(speed), speed, speed, p[..., 2])
    action = torch.zeros((1, 3, 3), requires_grad=True)
    for dense in (False, True):
        risk = lookahead_risk(state.tensors(), active, action, dense=dense)[0, 0]
        gradient, = torch.autograd.grad(risk, action)
        assert torch.isfinite(gradient).all()
        if dense:
            assert gradient[0, 2, 2].abs() > 1e-4
        else:
            assert gradient[0, 2, 2] == 0


def test_native_vertical_boundary_detects_a_true_shortfall_without_flagging_equal_layers():
    levels = torch.tensor([[6096., 6187.439998], [6096., 6187.44]], dtype=torch.float64)
    difference = levels[:, 0] - levels[:, 1]
    relative = torch.stack((torch.full_like(difference, 4998.8), torch.zeros_like(difference), difference), dim=-1)
    velocity = torch.zeros_like(relative)
    _, _, previous = cylinder_interval(relative, velocity, 5000., 91.44 - 1e-4, 0.05)
    _, _, matched = cylinder_interval(relative, velocity, 5000., native_vertical_radius(levels, 91.44)[:, 0, 1], 0.05)
    assert previous.tolist() == [False, False]
    assert matched.tolist() == [True, False]


def test_plan_b_event_checks_preserve_departure_time(monkeypatch):
    rl_gpu = pytest.importorskip("bluesky.rl_gpu")
    from envs.bluesky_plan_b import GeoTorchBlueSky

    config = rl_gpu.TorchVecEnvConfig(num_envs=1, max_aircraft=20, sim_dt=0.05, control_dt=6.0,
                                      device="cpu", dtype=torch.float64, seed=100)
    with pytest.raises(ValueError, match="divide the physics substeps"):
        GeoTorchBlueSky(config, event_checks_per_decision=7)
    env = GeoTorchBlueSky(config, family="full_mixture", timed_departures=True, event_checks_per_decision=3)
    env.reset()
    env.state.active[0, 1] = False
    env.pending[0, 1] = True
    env.departures[0, 1] = 2.0
    snapshots = []

    def record_events(start, end, seconds, route):
        snapshots.append((seconds, bool(start.active[0, 1])))
        fault = torch.zeros_like(start.active)
        return fault, fault, fault

    monkeypatch.setattr(env, "_events", record_events)
    env.step(torch.zeros(1, 20, 3))
    assert snapshots == [(2.0, False)] * 3
    assert env.state.active[0, 1] and not env.pending[0, 1]


def test_expected_mode_risk_matches_fixed_modes_and_scores_an_unselected_climb():
    position = torch.tensor([[[-6500.0, 0.0, 0.0], [6500.0, 0.0, 0.0], [0.0, 5000.0, 30.0]]])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    speed = torch.full((1, 3), 140.0)
    active = torch.tensor([[True, True, False]])
    state = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    control = torch.tensor([[[0.1, 0.9, -0.4], [-0.3, 0.0, 0.2], [0.5, 0.5, 0.5]]], requires_grad=True)
    hard = torch.nn.functional.one_hot(torch.tensor([[2, 1, 0]]), 4).float()
    direct = lookahead_risk(state.tensors(), active, hard[..., 1:] * control)
    averaged = expected_mode_risk(state.tensors(), active, hard, control)
    torch.testing.assert_close(averaged, direct)

    logits = torch.tensor([[[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
                          requires_grad=True)
    risk = expected_mode_risk(state.tensors(), active, logits.softmax(-1), control)
    route_gradient, action_gradient = torch.autograd.grad(risk.sum(), (logits, control))
    assert torch.isfinite(route_gradient).all() and torch.isfinite(action_gradient).all()
    assert route_gradient[0, 0, 2] < 0 and action_gradient[0, 0, 1] < 0
    assert risk[0, 2] == 0 and route_gradient[0, 2].count_nonzero() == 0


def test_delayed_consensus_transmits_once_per_edge_per_tick_without_amplifying_delivery():
    reference = torch.tensor([[2.0, 4.0]])
    edge = torch.tensor([[[False, True], [False, False]]])
    active = torch.tensor([[True, True]])
    live = torch.tensor([True])

    def project(proposal, _responsibility, _lower, _upper):
        return torch.zeros_like(proposal)

    for colored in (False, True):
        exchange = DelayedPairExchange(0.0, 70111, 1, "cpu", delay=2)
        for tick in range(4):
            exchange.begin_step(tick, active, live)
            layer = exchange.layer(("primary", "speed"))
            action, dual, residual = _consensus_updates(
                reference, 0.0, 10.0, edge, project, iterations=5, rho=0.5, mobility=None,
                warm_dual=None, exchange=layer, colored_updates=colored,
            )
            assert torch.isfinite(action).all() and torch.isfinite(dual).all() and torch.isfinite(residual).all()
        report, = exchange.report()
        # Exactly one packet attempt per active edge per tick (4 ticks -> 4 attempts, not 4*5 or 4*5*2).
        assert report["attempted_pair_exchanges"] == 4
        assert report["delivered_pair_exchanges"] == 2
        assert report["in_flight_pair_exchanges"] == 2
        assert report["used_delayed_pair_states"] == 2
        # Local dual was updated on the final tick rather than staying stuck at zero.
        assert dual[edge].abs().max() > 0


def test_altitude_admm_with_vertical_inertia_detects_reversing_overshoot():
    position = torch.tensor([[[0.0, 0.0, 100.0], [500.0, 0.0, 0.0]]], dtype=torch.float64)
    velocity = torch.tensor([[[10.0, 0.0], [-10.0, 0.0]]], dtype=torch.float64)
    proposal = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    active = torch.ones((1, 2), dtype=torch.bool)
    bounds = dict(minimum_altitude=0.0, maximum_altitude=304.8, horizontal_radius=5000.0, iterations=0)
    # Without vertical inertia, 100 m initial gap >= 91.44 m looks feasible at K=0.
    no_inertia = altitude_admm(position, velocity, proposal, active, **bounds)
    assert no_inertia.feasible.item()
    # With upper aircraft descending at -7.62 m/s, 19.05 m braking overshoot breaches 91.44 m.
    descending = torch.tensor([[-7.62, 0.0]], dtype=torch.float64)
    with_inertia = altitude_admm(position, velocity, proposal, active, vertical_speed=descending, **bounds)
    assert not with_inertia.feasible.item()
    # K=2 gradients through inertia-aware altitude_admm remain finite.
    solved = altitude_admm(position, velocity, proposal, active, vertical_speed=descending,
                           minimum_altitude=0.0, maximum_altitude=304.8, horizontal_radius=5000.0, iterations=2)
    solved.offset.square().sum().backward()
    assert torch.isfinite(proposal.grad).all()


def test_protected_command_marks_discrete_layer_search_resolutions_feasible():
    # Two aircraft colliding on the same altitude where speed-only fails and ADMM K=0 leaves conflict unresolved,
    # while batched_layer_search resolves it cleanly via vertical layer separation.
    position = torch.tensor([[[-6000.0, 0.0, 182.88], [6000.0, 0.0, 182.88]]])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    speed = torch.full((1, 2), 180.0)
    state = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    active = torch.ones((1, 2), dtype=torch.bool)
    adjacency = ~torch.eye(2, dtype=torch.bool).unsqueeze(0)
    actions = torch.zeros((1, 2, 7))
    _, admm_only_feasible = protected_command(actions, state.tensors(), adjacency, active, None, 0, 0)
    assert not admm_only_feasible.any()
    controls, resolved_feasible = protected_command(
        actions, state.tensors(), adjacency, active, None, 0, 0,
        batched_layer_search=True, batched_layer_margin=6.0, early_layer_search=True, exact_layer_forecast=True,
    )
    assert resolved_feasible.all()
    assert (controls[0, 0, 1] - controls[0, 1, 1]).abs() >= 0.99


def test_coordination_loss_trains_equivariant_admm_heads_without_nans_on_clear_worlds():
    # World 0 has a live conflict; World 1 is completely separated (minimum_clearance == +inf).
    position = torch.tensor([
        [[-6000.0, 0.0, 30.0], [6000.0, 2200.0, 30.0], [0.0, -3500.0, 80.0]],
        [[-40000.0, 0.0, 0.0], [40000.0, 40000.0, 300.0], [0.0, -40000.0, 180.0]],
    ])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]] * 2)
    speed = torch.full((2, 3), 180.0)
    active = torch.ones((2, 3), dtype=torch.bool)
    adjacency = torch.stack((~torch.eye(3, dtype=torch.bool), torch.zeros(3, 3, dtype=torch.bool)))
    graph = torch.randn((2, 3, 24)), torch.randn((2, 3, 3, 18)), adjacency, active
    physical = PhysicalState(position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    actor = Actor(width=16, equivariant_actor=True)
    policy = actor(*graph, deterministic=True, detach_coordination=True)
    loss = coordination_loss(
        encode_action(policy)[..., 4:], physical.tensors(), adjacency, active,
        policy.consensus_inputs, 2, 2, vertical_margin=6.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for head in (actor.consensus, actor.altitude_consensus, actor.willingness):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert any(p.grad.abs().max() > 0 for head in (actor.consensus, actor.altitude_consensus, actor.willingness)
               for p in head.parameters())
    assert all(p.grad is None for head in (actor.router, actor.speed, actor.altitude, actor.heading, actor.encoder)
               for p in head.parameters())


def test_decentralized_gap_mpc_breaks_symmetry_preserves_yaw_and_checks_inertia():
    # (1) Symmetric equal-altitude head-on conflict separates cleanly without communication.
    position = torch.tensor([[[-6000.0, 0.0, 180.0], [6000.0, 0.0, 180.0]]])
    direction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    speed = torch.full((1, 2), 180.0)
    active = torch.ones((1, 2), dtype=torch.bool)
    proposal = torch.zeros((1, 2, 3))
    state = (position, direction, speed, torch.zeros_like(speed), speed, speed, position[..., 2])
    commands, hold_contacts, planned_contacts = plan_gap_maneuver(state, active, proposal)
    assert hold_contacts.item() == 1 and planned_contacts.item() == 0
    assert commands[0, 0, 1] > 0 and commands[0, 1, 1] < 0

    # (2) Changing aircraft 1's private proposal never changes aircraft 0's chosen command (true decentralization).
    alt_proposal = proposal.clone()
    alt_proposal[0, 1] = torch.tensor([-0.9, 0.0, 0.8])
    alt_commands, _, _ = plan_gap_maneuver(state, active, alt_proposal)
    torch.testing.assert_close(commands[0, 0], alt_commands[0, 0], rtol=0.0, atol=0.0)

    # (3) Conflict-free off-route aircraft keeps safe pure heading-recovery proposal instead of holding.
    clear_pos = torch.tensor([[[-40000.0, 0.0, 180.0], [40000.0, 40000.0, 180.0]]])
    yaw_proposal = torch.tensor([[[0.0, 0.0, -0.8], [0.0, 0.0, 0.0]]])
    clear_state = (clear_pos, *state[1:])
    yaw_cmd, hold_c, plan_c = plan_gap_maneuver(clear_state, active, yaw_proposal)
    assert hold_c.item() == plan_c.item() == 0
    torch.testing.assert_close(yaw_cmd[0, 0], yaw_proposal[0, 0])

    # (4) Accelerated vertical inertia contact is detected even when linear forecast would miss it.
    trap_pos = torch.tensor([[[0.0, 0.0, 103.8628], [1000.0, 0.0, 201.0156]]])
    trap_vs = torch.tensor([[0.0, -7.62]])
    trap_sel = torch.tensor([[30.435, 185.341]])
    trap_state = (trap_pos, direction, speed, trap_vs, speed, speed, trap_sel)
    _, trap_hold, _ = plan_gap_maneuver(trap_state, active, proposal)
    assert trap_hold.item() == 1


def test_exp6_one_step_config_verification_tolerates_legacy_configs(tmp_path):
    from eval.bluesky_paper_exp6 import _verify_one_step_control

    base_cfg = {"plan": "b", "return_window": 24, "multi_step_return": True, "output": "a", "decisions": 2200,
                "target_episodes": 0, "target_updates": 20000}
    new_cfg = {**base_cfg, "multi_step_return": False, "output": "b", "decisions": 1523, "target_updates": 0,
               "admm_communication_period": None, "admm_gap_policy": "hold"}
    p_ckpt, o_ckpt = tmp_path / "primary.pt", tmp_path / "one_step.pt"
    torch.save({"updates": 15000, "decisions": 1523, "training_plan": "b", "tag": "p"}, p_ckpt)
    torch.save({"updates": 15000, "decisions": 1523, "training_plan": "b", "tag": "o"}, o_ckpt)
    c_cfg, s_cfg = tmp_path / "control.json", tmp_path / "one.json"
    c_cfg.write_text(json.dumps(base_cfg))
    s_cfg.write_text(json.dumps(new_cfg))
    args = SimpleNamespace(checkpoint=p_ckpt, one_step_checkpoint=o_ckpt, control_config=c_cfg, one_step_config=s_cfg)
    verified = _verify_one_step_control(args)
    assert verified["updates"] == 15000 and verified["decisions"] == 1523



