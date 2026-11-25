"""Differentiable bounded speed consensus with continuous-time verification."""

from dataclasses import dataclass

import torch

from .conflict_graph import _ball_interval, closest_approach, connected_components, greedy_colors


@dataclass
class SpeedPlan:
    speed: torch.Tensor
    dual: torch.Tensor
    minimum_clearance: torch.Tensor
    primal_residual: torch.Tensor
    feasible: torch.Tensor
    conflicts: torch.Tensor


def _pair_columns(values):
    i, j = torch.broadcast_tensors(values.unsqueeze(-1), values.unsqueeze(-2))
    return torch.stack((i, j), dim=-1)


def _pairs(active):
    nodes = active.shape[-1]
    upper = torch.ones((nodes, nodes), dtype=torch.bool, device=active.device).triu(1)
    return active.unsqueeze(-1) & active.unsqueeze(-2) & upper


def _consensus_updates(reference, lower, upper, edge, project, iterations, rho, mobility, warm_dual,
                       extra_rounds=0, extra_nodes=None, *, exchange=None, colored_updates=False):
    penalty = torch.where(edge, torch.as_tensor(rho, dtype=reference.dtype, device=reference.device), 0.0)
    lower = torch.broadcast_to(torch.as_tensor(lower, dtype=reference.dtype, device=reference.device), reference.shape)
    upper = torch.broadcast_to(torch.as_tensor(upper, dtype=reference.dtype, device=reference.device), reference.shape)
    reference = reference.clamp(lower, upper)
    pair_lo, pair_hi = _pair_columns(lower), _pair_columns(upper)
    mobility = torch.ones_like(reference) if mobility is None else mobility
    mobility = _pair_columns(mobility.clamp_min(1e-3))
    responsibility = mobility / mobility.sum(dim=-1, keepdim=True)
    consensus = _pair_columns(reference)
    initial_dual = torch.zeros_like(consensus) if warm_dual is None else warm_dual
    dual = torch.where(edge.unsqueeze(-1), initial_dual, 0.0)
    delayed = getattr(exchange, "receive", None)
    if delayed is not None:
        consensus, dual = delayed(edge, consensus, dual)
    weight = 1 + penalty.sum(dim=-1) + penalty.sum(dim=-2)
    action = reference
    residual = torch.zeros_like(reference[..., 0])
    groups = (None,)
    if colored_updates:
        adjacent = edge | edge.transpose(-1, -2)
        colors = greedy_colors(adjacent)
        connected = adjacent.any(dim=-1)
        groups = tuple(connected & (colors == color) for color in range(int(colors.max()) + 1))
    for index in range(iterations + extra_rounds):
        extra = index >= iterations
        participating = edge
        if extra:
            local = extra_nodes.unsqueeze(-1) & extra_nodes.unsqueeze(-2)
            participating = edge & local
        for group in groups:
            weighted = penalty.unsqueeze(-1) * (consensus - dual)
            update = (reference + weighted[..., 0].sum(dim=-1) + weighted[..., 1].sum(dim=-2)) / weight
            update = update.clamp(lower, upper)
            selected = extra_nodes if extra else None
            requested = participating
            if group is not None:
                selected = group & extra_nodes if extra else group
                requested = participating & (selected.unsqueeze(-1) | selected.unsqueeze(-2))
            action = update if selected is None else torch.where(selected, update, action)
            shared = _pair_columns(action)
            proposed = project((shared + dual).clamp(pair_lo, pair_hi), responsibility, pair_lo, pair_hi)
            if delayed is not None:
                exchange.send(requested, proposed, dual + shared - proposed)
                continue
            if exchange is None and group is None:
                delivered = requested
                consensus = proposed
            else:
                delivered = requested if exchange is None else exchange(requested)
                consensus = torch.where(delivered.unsqueeze(-1), proposed, consensus)
            difference = torch.where(delivered.unsqueeze(-1), shared - consensus, 0.0)
            dual = dual + difference
        if colored_updates or exchange is not None:
            # Include a missed edge's residual against its last received state.
            difference = torch.where(participating.unsqueeze(-1), shared - consensus, 0.0)
        residual = difference.abs().amax(dim=(-3, -2, -1))
    return action, dual, residual


def speed_admm(position, direction, proposal, active, adjacency, *, minimum_speed, maximum_speed,
               horizon, radius, iterations=5, projections=3, rho=8.0, mobility=None, warm_dual=None,
               margin=1.0, vertical_radius=None, vertical_rate=None, exchange=None, colored_updates=False):
    """Unroll distributed speed updates on a spherical separation constraint.

    K=0 verifies the raw proposal. When speed-only is infeasible, callers must
    escalate to a different action axis; the returned proposal is audited.
    """
    pairs = _pairs(active)
    edge = pairs & (adjacency | adjacency.transpose(-1, -2))
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    direction_i = direction.unsqueeze(-2)
    direction_j = direction.unsqueeze(-3)
    first = torch.zeros_like(relative[..., 0])
    last = torch.full_like(first, horizon)
    if vertical_radius is not None:
        dz = vertical_rate.unsqueeze(-1) - vertical_rate.unsqueeze(-2)
        first, last, potential = _ball_interval(relative[..., 2:], dz.unsqueeze(-1), vertical_radius, horizon)
        pairs = pairs & potential
        edge = edge & potential
        relative = relative[..., :2]
        direction_i = direction_i[..., :2]
        direction_j = direction_j[..., :2]

    def clearance(edge_speed):
        velocity = direction_i * edge_speed[..., 0, None] - direction_j * edge_speed[..., 1, None]
        distance, tau, offset = closest_approach(relative + velocity * first.unsqueeze(-1), velocity, last - first)
        return distance, first + tau, offset

    def project(consensus, responsibility, lower, upper):
        for _ in range(projections):
            distance, time, offset = clearance(consensus)
            normal = offset / distance.clamp_min(1e-9).unsqueeze(-1)
            jacobian = time.unsqueeze(-1) * torch.stack(
                ((normal * direction_i).sum(dim=-1), -(normal * direction_j).sum(dim=-1)), dim=-1,
            )
            direction_step = responsibility * jacobian
            denominator = (jacobian * direction_step).sum(dim=-1).clamp_min(1e-9)
            shortfall = torch.where(edge, (radius + margin - distance).clamp_min(0), 0.0)
            step = (shortfall / denominator).unsqueeze(-1) * direction_step
            consensus = (consensus + step).clamp(lower, upper)
        return consensus

    speed, dual, residual = _consensus_updates(
        proposal, minimum_speed, maximum_speed, edge, project, iterations, rho, mobility, warm_dual,
        exchange=exchange, colored_updates=colored_updates,
    )
    distances, _, _ = clearance(_pair_columns(speed))
    clearance_margin = (distances - radius).masked_fill(~pairs, float("inf"))
    minimum = clearance_margin.amin(dim=(-1, -2))
    violations = clearance_margin < -1e-5
    return SpeedPlan(speed, dual, minimum, residual, minimum >= -1e-5, violations | violations.transpose(-1, -2))


@dataclass
class AltitudePlan:
    offset: torch.Tensor
    dual: torch.Tensor
    minimum_clearance: torch.Tensor
    primal_residual: torch.Tensor
    feasible: torch.Tensor
    conflicts: torch.Tensor
    adaptive_nodes: torch.Tensor


def altitude_admm(position, horizontal_velocity, proposal, active, *, minimum_altitude, maximum_altitude,
                  horizon=120.0, horizontal_radius=5000.0, vertical_radius=91.44, max_offset=91.44,
                  climb_time=15.0, iterations=5, rho=8.0, mobility=None, warm_dual=None, margin=3.0,
                  colored_ties=False, extra_rounds=0, exchange=None, colored_updates=False):
    """Resolve residual horizontal interactions using bounded piecewise-affine climbs.

    The climb starts immediately and reaches its commanded offset at climb_time.
    Every horizontal overlap is considered, including traffic in target layers.
    """
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    delta_velocity = horizontal_velocity.unsqueeze(-2) - horizontal_velocity.unsqueeze(-3)
    lo, hi, horizontal_hit = _ball_interval(relative[..., :2], delta_velocity, horizontal_radius, horizon)
    edge = _pairs(active) & horizontal_hit
    ramp_lo = (lo / climb_time).clamp(0, 1)
    ramp_hi = (hi / climb_time).clamp(0, 1)
    gap = relative[..., 2]
    extra_nodes = torch.zeros_like(active)
    if extra_rounds:
        # A clique of size at least three exists exactly when there is a triangle.
        reachable = edge & (gap.abs() < vertical_radius + 2 * max_offset)
        links = reachable | reachable.transpose(-1, -2)
        shared = links.to(position.dtype) @ links.to(position.dtype)
        triangle_nodes = (links & (shared > 0)).any(dim=-1)
        # Continue the full connected constraint component of each triangle.
        labels = connected_components(edge | edge.transpose(-1, -2))
        members = labels.unsqueeze(-1) == labels.unsqueeze(-2)
        extra_nodes = (members & triangle_nodes.unsqueeze(-2)).any(dim=-1)
    ordering = torch.where(gap > 0, 1.0, -1.0)
    if colored_ties:
        ties = edge & (gap.abs() < 1e-3)
        colors = greedy_colors(ties | ties.transpose(-1, -2))
        above = colors.unsqueeze(-1) > colors.unsqueeze(-2)
        ordering = torch.where(ties, torch.where(above, 1.0, -1.0), ordering)
    lower = (minimum_altitude - position[..., 2]).clamp_min(-max_offset)
    upper = (maximum_altitude - position[..., 2]).clamp_max(max_offset)

    def offsets(consensus):
        difference = consensus[..., 0] - consensus[..., 1]
        return gap + difference * ramp_lo, gap + difference * ramp_hi

    def project(consensus, responsibility, lower, upper):
        early, late = offsets(consensus)
        early, late = early * ordering, late * ordering
        worst = torch.minimum(early, late)
        ramp = torch.where(early <= late, ramp_lo, ramp_hi)
        derivative = ordering * ramp
        gradient = torch.stack((derivative, -derivative), dim=-1)
        step_direction = responsibility * gradient
        scale = (gradient * step_direction).sum(dim=-1).clamp_min(1e-9)
        shortfall = torch.where(edge, (vertical_radius + margin - worst).clamp_min(0), 0.0)
        return (consensus + (shortfall / scale).unsqueeze(-1) * step_direction).clamp(lower, upper)

    altitude, dual, residual = _consensus_updates(
        proposal, lower, upper, edge, project, iterations, rho, mobility, warm_dual, extra_rounds, extra_nodes,
        exchange=exchange, colored_updates=colored_updates,
    )
    early, late = offsets(_pair_columns(altitude))
    separation = torch.where(early * late <= 0, 0.0, torch.minimum(early.abs(), late.abs()))
    clearance_margin = (separation - vertical_radius).masked_fill(~edge, float("inf"))
    minimum = clearance_margin.amin(dim=(-1, -2))
    violations = clearance_margin < -1e-5
    return AltitudePlan(altitude, dual, minimum, residual, minimum >= -1e-5,
                        violations | violations.transpose(-1, -2), extra_nodes)
