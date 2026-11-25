"""Rate-limited Geo-ADMM projections for native BlueSky action coordinates."""

from dataclasses import dataclass, replace
from math import isfinite, radians

import torch
from torch.nn.functional import softplus

from core.admm_layer import altitude_admm, speed_admm
from core.conflict_graph import _ball_interval, closest_approach, connected_components
from core.layer_search import improve_layer_choices, improve_three_aircraft_choices, layer_clearance
from core.native_vertical import exact_imminent_clearance
from envs.bluesky_graph import SCALE
from envs.observations import LAYER

KNOT = 0.514444
CAS_DELTA = 4.5 * KNOT


def single_axis_commands(controls):
    """Keep the heading, altitude or speed axis in that order of priority."""
    magnitude = controls.abs()
    heading = magnitude[..., 2] > 0.05
    altitude = ~heading & (magnitude[..., 1] > 0.01)
    speed = ~(heading | altitude)
    return controls * torch.stack((speed, altitude, heading), dim=-1)


@dataclass
class PhysicalState:
    position: torch.Tensor
    direction: torch.Tensor
    ground_speed: torch.Tensor
    vertical_speed: torch.Tensor
    cas: torch.Tensor
    tas: torch.Tensor
    selected_altitude: torch.Tensor

    def tensors(self):
        return (self.position, self.direction, self.ground_speed, self.vertical_speed, self.cas, self.tas,
                self.selected_altitude)


def _horizontal_clearance(state, planned_speed):
    horizontal = state.position[..., :2]
    velocity = state.direction[..., :2] * planned_speed.unsqueeze(-1)
    relative = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    relative_velocity = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    distance, _, _ = closest_approach(relative, relative_velocity, SCALE.horizon)
    return distance


def _active_pairs(active):
    distinct = ~torch.eye(active.shape[-1], dtype=torch.bool, device=active.device)
    return active.unsqueeze(-1) & active.unsqueeze(-2) & distinct


def _vertical_separation(state, altitude_command):
    altitude = state.position[..., 2]
    target = torch.where(altitude_command != 0.0, altitude + LAYER * altitude_command, state.selected_altitude)
    target = target.clamp(0.0, 304.8)
    vertical_speed = state.vertical_speed
    overshoot = vertical_speed.abs() * 0.25 + vertical_speed.square() / (2.0 * 1.524)
    lower = torch.minimum(altitude, target) - torch.where(vertical_speed < 0, overshoot, 0.0)
    upper = torch.maximum(altitude, target) + torch.where(vertical_speed > 0, overshoot, 0.0)
    gap = lower.unsqueeze(-1) - upper.unsqueeze(-2)
    return torch.maximum(gap, gap.transpose(-1, -2)) >= LAYER - 1e-4


def _directional_heading(state, speed, heading, altitude, adjacency, active):
    vector = state.direction[..., :2] * speed.unsqueeze(-1)
    east, north = vector.unbind(dim=-1)
    turn = heading * radians(5.0)
    turned = torch.stack((east * turn.cos() + north * turn.sin(), north * turn.cos() - east * turn.sin()), dim=-1)
    point = state.position[..., :2]
    relative = point.unsqueeze(-2) - point.unsqueeze(-3)
    original = vector.unsqueeze(-2) - vector.unsqueeze(-3)
    change = turned - vector
    own = change.unsqueeze(-2)
    other = change.unsqueeze(-3)
    def clearance(velocity):
        return closest_approach(relative, velocity, SCALE.horizon)[0]

    nominal = clearance(original)
    alone = clearance(original + own)
    both = clearance(original + own - other)
    neighbor_only = alone.transpose(-1, -2)
    dangerous = ((alone < SCALE.horizontal_radius + 20.0) & (alone < nominal - 0.25)) | (
        (both < SCALE.horizontal_radius + 20.0) & (both < neighbor_only - 0.25)
    )
    linked = (adjacency | adjacency.transpose(-1, -2)).any(dim=-1)
    watch = _active_pairs(active) & ~_vertical_separation(state, altitude)
    blocked = linked & (watch & dangerous).any(dim=-1)
    return heading.masked_fill(blocked, 0.0)


def _protect_heading(state, planned_speed, heading, active, *, altitude_command=None):
    horizon, radius = SCALE.horizon, SCALE.horizontal_radius
    distance = _horizontal_clearance(state, planned_speed)
    lateral_budget = (2.0 * planned_speed * horizon) * (heading.abs() * radians(2.5)).sin()
    margin = lateral_budget.unsqueeze(-1) + lateral_budget.unsqueeze(-2)
    unsafe = _active_pairs(active) & (distance < radius + margin + 20.0)
    if altitude_command is not None:
        unsafe &= ~_vertical_separation(state, altitude_command)
    return heading.masked_fill(unsafe.any(dim=-1), 0.0)


def _admm_plans(action, state, ground_delta, adjacency, active, consensus, iterations, extra_rounds,
                *, independent_tiers=False, exchange=None, colored_updates=False, observe_residuals=None,
                channel="primary", vertical_margin=0.0):
    speed = state.ground_speed
    initial_speed = speed + ground_delta * action[..., 0]
    remaining_altitude = (state.selected_altitude - state.position[..., 2]).clamp(-LAYER, LAYER)
    initial_altitude = torch.where(action[..., 1] != 0, LAYER * action[..., 1], remaining_altitude)
    vertical_rate = torch.where(initial_altitude.abs() > 1.0, state.vertical_speed, 0.0)
    dual, rho, mobility, altitude_dual, altitude_rho = consensus or (None, 8.0, None, None, 8.0)
    layer_exchange = getattr(exchange, "layer", None)
    speed_exchange = exchange if layer_exchange is None else layer_exchange((channel, "speed"))
    altitude_exchange = (exchange if layer_exchange is None
                         else layer_exchange((channel, "altitude"), state.position[..., 2]))
    first = speed_admm(state.position, state.direction, initial_speed, active, adjacency,
                       minimum_speed=(speed - ground_delta).clamp_min(0), maximum_speed=speed + ground_delta,
                       horizon=SCALE.horizon, radius=SCALE.horizontal_radius, iterations=iterations, margin=20.0,
                       vertical_radius=LAYER, vertical_rate=vertical_rate,
                       warm_dual=dual, rho=rho, mobility=mobility, exchange=speed_exchange,
                       colored_updates=colored_updates)
    tier1_speed = first.speed.detach() if independent_tiers else first.speed
    second = altitude_admm(state.position, state.direction[..., :2] * tier1_speed.unsqueeze(-1), initial_altitude,
                           active, minimum_altitude=0.0, maximum_altitude=304.8, horizon=SCALE.horizon,
                           horizontal_radius=SCALE.horizontal_radius, iterations=iterations,
                           warm_dual=altitude_dual, rho=altitude_rho, mobility=mobility, colored_ties=True,
                           margin=max(3.0, float(vertical_margin)), vertical_speed=state.vertical_speed,
                           extra_rounds=extra_rounds, exchange=altitude_exchange, colored_updates=colored_updates)
    if observe_residuals is not None:
        observe_residuals(first.primal_residual, second.primal_residual)
    return first, second, initial_altitude, remaining_altitude


def _project_components(action, state, ground_delta, adjacency, members, active, consensus, iterations, extra_rounds,
                        *, exchange=None, colored_updates=False, observe_residuals=None, channel="primary",
                        vertical_margin=0.0):
    first, second, initial_altitude, remaining_altitude = _admm_plans(
        action, state, ground_delta, adjacency, active, consensus, iterations, extra_rounds, exchange=exchange,
        colored_updates=colored_updates, observe_residuals=observe_residuals, channel=channel,
        vertical_margin=vertical_margin,
    )
    changes_altitude = active & (initial_altitude.abs() > 1.0)
    first_issues = first.conflicts.any(dim=-1) | changes_altitude
    second_issues = second.conflicts.any(dim=-1)
    first_certified = active & ~(members & first_issues.unsqueeze(-2)).any(dim=-1)
    second_certified = active & ~(members & second_issues.unsqueeze(-2)).any(dim=-1)
    use_altitude = ~first_certified & second_certified
    use_speed = first_certified | use_altitude
    speed_action = (first.speed - state.ground_speed) / ground_delta.clamp_min(1e-4)
    altitude_action = second.offset / LAYER
    # A zero increment leaves any pre-existing target in effect on both engines.
    retain_altitude = (second.offset - remaining_altitude).abs() < 0.03
    cancel = (altitude_action.abs() < 1e-5) & ~retain_altitude
    altitude_action = torch.where(retain_altitude, 0.0, altitude_action)
    altitude_action = torch.where(cancel, torch.full_like(altitude_action, 1e-3), altitude_action)
    projected = torch.stack((torch.where(use_speed, speed_action, action[..., 0]),
                             torch.where(use_altitude, altitude_action, action[..., 1]), action[..., 2]), dim=-1)
    return projected, use_speed


def coordination_loss(action, physical, adjacency, active, consensus, iterations, extra_rounds, *, vertical_margin=0.0):
    """Train the continuous warm start against the same constraints as the rollout projector."""
    state = PhysicalState(*physical)
    ground_delta = CAS_DELTA * (state.tas / state.cas.clamp_min(1)).clamp(0.8, 2.0)
    first, second, _, _ = _admm_plans(action.detach(), state, ground_delta, adjacency, active, consensus,
                                      iterations, extra_rounds, independent_tiers=True,
                                      vertical_margin=vertical_margin)
    speed_shortfall = torch.where(torch.isfinite(first.minimum_clearance), (-first.minimum_clearance).clamp_min(0.0), 0.0)
    layer_shortfall = torch.where(torch.isfinite(second.minimum_clearance), (-second.minimum_clearance).clamp_min(0.0), 0.0)
    speed_gap = speed_shortfall / SCALE.horizontal_radius
    layer_gap = layer_shortfall / LAYER
    speed_residual = first.primal_residual / CAS_DELTA
    layer_residual = second.primal_residual / LAYER
    return (speed_gap.square() + layer_gap.square() + 0.1 * (speed_residual.square() + layer_residual.square())).mean()


def protected_command(action, state, adjacency, active, consensus, iterations, fallback, *, by_component=False,
                      vertical_heading=False, directional_heading=False, extra_rounds=0, batched_layer_search=False,
                      batched_layer_margin=0.0, accelerated_layer_forecast=False, preserve_imminent_clearance=False,
                      early_layer_search=False, exact_layer_forecast=False,
                      joint_layer_search=False, joint_layer_reserve=0.0, exchange=None, colored_updates=False,
                      observe_residuals=None):
    action = action[..., 4:]
    state = PhysicalState(*state)
    ground_delta = CAS_DELTA * (state.tas / state.cas.clamp_min(1)).clamp(0.8, 2.0)
    members = active.unsqueeze(-2)
    if by_component:
        # Outside this envelope, bounded speed changes cannot couple the two aircraft.
        speed_budget = SCALE.horizon * (ground_delta.unsqueeze(-1) + ground_delta.unsqueeze(-2))
        reachable = _horizontal_clearance(state, state.ground_speed) < SCALE.horizontal_radius + speed_budget + 20.0
        coupled = _active_pairs(active) & (adjacency | adjacency.transpose(-1, -2) | reachable)
        labels = connected_components(coupled)
        members = labels.unsqueeze(-1) == labels.unsqueeze(-2)
    proposed, feasible = _project_components(action, state, ground_delta, adjacency, members, active, consensus,
                                             iterations, extra_rounds, exchange=exchange, colored_updates=colored_updates,
                                             observe_residuals=observe_residuals, vertical_margin=batched_layer_margin)
    if fallback:
        fixed, verified = _project_components(action, state, ground_delta, adjacency, members, active, None, fallback, 0,
                                              exchange=exchange, colored_updates=colored_updates,
                                              observe_residuals=observe_residuals, channel="fallback",
                                              vertical_margin=batched_layer_margin)
        proposed = torch.where((verified & ~feasible).unsqueeze(-1), fixed, proposed)
        feasible = feasible | verified
    proposed = proposed.clamp(-1, 1)
    speed = state.ground_speed + ground_delta * proposed[..., 0]
    altitude_command = proposed[..., 1] if vertical_heading else None
    if directional_heading:
        heading = _directional_heading(state, speed, proposed[..., 2], proposed[..., 1], adjacency, active)
    else:
        heading = _protect_heading(state, speed, proposed[..., 2], active, altitude_command=altitude_command)
    controls = torch.stack((proposed[..., 0], proposed[..., 1], heading), dim=-1)
    if batched_layer_search:
        controls, layer_feasible = _discrete_layer_search(
            state, controls, active, feasible, members=members, margin=batched_layer_margin,
            accelerated=accelerated_layer_forecast, early=early_layer_search, exact=exact_layer_forecast,
            preserve_imminent=preserve_imminent_clearance, joint=joint_layer_search,
            joint_reserve=joint_layer_reserve,
        )
        feasible = feasible | layer_feasible
    return controls.masked_fill(~active.unsqueeze(-1), 0.0), feasible


def _altitude_at(altitude, target, rate, time):
    return (altitude + rate * time).clamp(torch.minimum(altitude, target), torch.maximum(altitude, target))


def _vertical_difference(state, time):
    altitude = state.position[..., 2]
    fields = altitude, state.selected_altitude, state.vertical_speed
    first = _altitude_at(*(field.unsqueeze(-1) for field in fields), time)
    second = _altitude_at(*(field.unsqueeze(-2) for field in fields), time)
    return first - second


def _overlap_vertical_gap(state, relative, velocity, horizon, cpa_gap):
    """Find minimum vertical separation while horizontal separation is inside the radius."""
    entry, exit, intersects = _ball_interval(relative, velocity, SCALE.horizontal_radius, horizon)
    altitude = state.position[..., 2]
    rate = torch.where(state.vertical_speed != 0, state.vertical_speed, 1.0)
    level_capture = (state.selected_altitude - altitude) / rate
    capture_i = level_capture.unsqueeze(-1).clamp(entry, exit)
    capture_j = level_capture.unsqueeze(-2).clamp(entry, exit)
    gaps = _vertical_difference(state, torch.stack((entry, exit, capture_i, capture_j)))
    minimum = torch.maximum(gaps.amin(dim=0), -gaps.amax(dim=0)).clamp_min(0.0)
    return torch.where(intersects, torch.minimum(cpa_gap, minimum), cpa_gap)


def _forecast_command(state, action, *, dense):
    speed, altitude, heading = action.unbind(dim=-1)
    ratio = (state.tas / state.cas.clamp_min(1)).clamp(0.8, 2.0)
    east, north = state.direction[..., :2].unbind(dim=-1)
    angle = heading * radians(5.0)
    direction = torch.stack((east * angle.cos() + north * angle.sin(),
                             north * angle.cos() - east * angle.sin()), dim=-1)
    z = state.position[..., 2]
    offset = LAYER * altitude
    held_offset = offset if dense else 0.0
    target = torch.where(altitude != 0, (z + offset).clamp(0, 304.8), state.selected_altitude + held_offset)
    rate = torch.where(altitude != 0, (target - z).div(15.0).clamp(-7.62, 7.62),
                       state.vertical_speed + held_offset / 15.0)
    return replace(state, direction=direction, ground_speed=state.ground_speed + speed * CAS_DELTA * ratio,
                   selected_altitude=target, vertical_speed=rate)


def _joint_clearance_reserve(clearance, overlaps, entry, choice, reserve):
    batch, aircraft, _, options, _ = clearance.shape
    other = choice[:, None, :, None, None].expand(batch, aircraft, aircraft, options, 1)
    own = choice[:, :, None, None].expand(batch, aircraft, aircraft, 1)
    current = clearance.gather(-1, other).squeeze(-1).gather(-1, own).squeeze(-1)
    buffered = LAYER + reserve
    # Retain a reserve for separated traffic and preserve the gap for overlaps within six seconds.
    minimum = torch.where(current >= buffered, buffered, 0.0)
    minimum = torch.maximum(minimum, (current - 1.0).masked_fill(entry > 6.0, 0.0))
    return ~overlaps[..., None, None] | (clearance >= minimum[..., None, None])


def _discrete_layer_search(state, controls, active, feasible, *, members=None, margin=0.0, accelerated=False,
                           preserve_imminent=False, early=False, exact=False, joint=False, joint_reserve=0.0):
    if not isfinite(joint_reserve) or joint_reserve < 0 or (joint_reserve and not joint):
        raise ValueError("The reserve must be nonnegative and requires three-aircraft search")
    forecast = _forecast_command(state, controls, dense=False)
    position = state.position[..., :2]
    velocity = forecast.direction * forecast.ground_speed.unsqueeze(-1)
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    change = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    lo, hi, overlaps = _ball_interval(relative, change, SCALE.horizontal_radius, SCALE.horizon)
    unresolved = (active & ~feasible).any(dim=-1)
    overlaps = overlaps & _active_pairs(active)
    if not margin:
        if early:
            unresolved |= (overlaps & (lo <= 6.0)).any(dim=(-1, -2))
        overlaps &= unresolved[..., None, None]

    height = state.position[..., 2]
    selected = state.selected_altitude
    levels = (height - LAYER, height + LAYER)
    if margin:
        levels += (height - 2 * margin, height + 2 * margin)
    bounds = torch.stack(levels, dim=-1).clamp(0.0, 304.8)
    increment = ((bounds - height.unsqueeze(-1)) / LAYER).clamp(-1, 1)
    increment = torch.where(increment.abs() < 1e-5, 1e-3, increment)
    candidates = torch.cat((controls[..., 1:2], torch.zeros_like(height).unsqueeze(-1), increment), dim=-1)
    targets = torch.where(candidates != 0, height.unsqueeze(-1) + LAYER * candidates, selected.unsqueeze(-1))
    clearance = layer_clearance(height, state.vertical_speed, targets.clamp(0.0, 304.8), lo, hi,
                                accelerated=accelerated)
    if exact:
        clearance = exact_imminent_clearance(clearance, height, state.vertical_speed, selected, candidates,
                                             lo, hi, overlaps & (lo <= 6.0))
    independent = ~overlaps[..., None, None]
    preferred = ((clearance >= LAYER + margin) | independent) if margin else None
    preserve = None
    if preserve_imminent:
        baseline = clearance[..., 0, 0]
        imminent = overlaps & (lo <= 30.0) & (baseline < LAYER - 1e-4)
        # Do not trade away existing separation on a developing contact for a different pair.
        preserve = ~imminent[..., None, None] | (clearance >= baseline[..., None, None] - 1.0)
    compatible = (clearance >= LAYER - 1e-4) | independent
    choice = improve_layer_choices(compatible, preferred=preferred, preserve=preserve)
    if joint:
        reserved = _joint_clearance_reserve(clearance, overlaps, lo, choice, joint_reserve) if joint_reserve else None
        if preserve is not None:
            reserved = preserve if reserved is None else reserved & preserve
        choice = improve_three_aircraft_choices(compatible, choice, connected_components(overlaps),
                                               preferred=preferred, preserve=reserved)
    batch, aircraft, _, options, _ = compatible.shape
    other = choice[:, None, :, None, None].expand(batch, aircraft, aircraft, options, 1)
    own = choice[:, :, None, None].expand(batch, aircraft, aircraft, 1)
    chosen_compatible = compatible.gather(-1, other).squeeze(-1).gather(-1, own).squeeze(-1)
    layer_issues = (~chosen_compatible).any(dim=-1)
    scope = active.unsqueeze(-2) if members is None else members
    layer_feasible = active & ~(scope & layer_issues.unsqueeze(-2)).any(dim=-1)
    altitude = candidates.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    return torch.stack((controls[..., 0], altitude, controls[..., 2]), dim=-1), layer_feasible


def _margin_bands(distance, gap):
    return (SCALE.horizontal_radius - distance) / 600.0, (LAYER - gap - 6.0) / 8.0


def lookahead_risk(physical, active, action=None, *, dense=False, overlap=False):
    """Smooth per-aircraft loss on cylindrical CPA margins at 60 s and 120 s."""
    state = PhysicalState(*physical)
    if action is not None:
        state = _forecast_command(state, action, dense=dense)
    horizontal = state.position[..., :2]
    velocity = state.direction[..., :2] * state.ground_speed.unsqueeze(-1)
    rel = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    dv = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    pairs = _active_pairs(active)
    risk = []
    for horizon in (60.0, 120.0):
        distance, time, _ = closest_approach(rel, dv, horizon)
        gap = _vertical_difference(state, time).abs()
        if overlap:
            gap = _overlap_vertical_gap(state, rel, dv, horizon, gap)
        horizontal_band, vertical_band = _margin_bands(distance, gap)
        vertical_cost = vertical_band.sigmoid()
        maximum = (horizontal_band.sigmoid() * vertical_cost).masked_fill(~pairs, 0.0).amax(dim=-1)
        if dense:
            # Keep the original longest-range signal and add slope for up to three risks.
            penetration = softplus(horizontal_band) / (SCALE.horizontal_radius / 600.0)
            largest = (penetration * vertical_cost).masked_fill(~pairs, 0.0).topk(
                min(3, active.shape[-1]), dim=-1).values
            maximum = maximum + 0.35 * largest.sum(dim=-1)
        risk.append(maximum)
    return 0.55 * risk[0] + 0.45 * risk[1]


def expected_mode_risk(physical, active, probabilities, controls):
    """Evaluate each aircraft pair under the actor's four action modes."""
    state = PhysicalState(*physical)
    axes = torch.eye(controls.shape[-1], device=controls.device, dtype=controls.dtype)
    commands = torch.cat((torch.zeros_like(controls.unsqueeze(-2)), controls.unsqueeze(-1) * axes), dim=-2)
    # Add a mode axis before vector coordinates and after each scalar.
    expanded = PhysicalState(state.position.unsqueeze(-2), state.direction.unsqueeze(-2),
                             *(field.unsqueeze(-1) for field in state.tensors()[2:]))
    forecast = _forecast_command(expanded, commands, dense=False)
    velocity = forecast.direction * forecast.ground_speed.unsqueeze(-1)
    horizontal = state.position[..., :2]
    relative = (horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3))[..., None, None, :]
    change = velocity[:, :, None, :, None] - velocity[:, None, :, None, :]
    weights = probabilities[:, :, None, :, None] * probabilities[:, None, :, None, :]
    fields = forecast.position[..., 2], forecast.selected_altitude, forecast.vertical_speed
    own = tuple(field[:, :, None, :, None] for field in fields)
    other = tuple(field[:, None, :, None, :] for field in fields)
    pairs = _active_pairs(active)
    risks = []
    for horizon in (60.0, 120.0):
        distance, time, _ = closest_approach(relative, change, horizon)
        gap = (_altitude_at(*own, time) - _altitude_at(*other, time)).abs()
        horizontal_band, vertical_band = _margin_bands(distance, gap)
        risk = (horizontal_band.sigmoid() * vertical_band.sigmoid() * weights).sum(dim=(-1, -2))
        risks.append(risk.masked_fill(~pairs, 0.0).amax(dim=-1))
    return 0.55 * risks[0] + 0.45 * risks[1]
