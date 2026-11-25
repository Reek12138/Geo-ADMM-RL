"""Continuous-time relative geometry. Distances are in metres, times in seconds."""

from dataclasses import dataclass

import torch


def closest_approach(relative, velocity, duration):
    """Minimum along a synchronized relative trajectory, including both endpoints."""
    squared_speed = velocity.square().sum(dim=-1)
    crossing = -(relative * velocity).sum(dim=-1) / squared_speed.clamp_min(1e-12)
    crossing = torch.where(squared_speed > 1e-12, crossing, 0.0)
    duration = torch.as_tensor(duration, device=relative.device, dtype=relative.dtype)
    time = torch.minimum(crossing.clamp_min(0), duration.clamp_min(0))
    offset = relative + velocity * time.unsqueeze(-1)
    return torch.linalg.vector_norm(offset, dim=-1), time, offset


def _pair_clearance_torch(params, vi, vj, radius):
    """Tensor counterpart to Geometry-UAV-CDR's six-case _pair_clearance.

    With two endpoint orders, the clipped minimizer represents the interior,
    entry and exit minimizers in each order.
    """
    start_i, start_j = params["t1_s"], params["t2_s"]
    end_i = start_i + params["D"] / vi.clamp_min(1e-4)
    end_j = start_j + params["E"] / vj.clamp_min(1e-4)
    lo = torch.maximum(start_i, start_j)
    hi = torch.minimum(end_i, end_j)
    vel_i = params["n1"] * vi.unsqueeze(-1)
    vel_j = params["n2"] * vj.unsqueeze(-1)
    relative = (
        params["p_io"] + vel_i * (lo - start_i).unsqueeze(-1)
        - params["p_jo"] - vel_j * (lo - start_j).unsqueeze(-1)
    )
    minimum, _, _ = closest_approach(relative, vel_i - vel_j, hi - lo)
    disjoint = (lo - hi) * torch.maximum(vi, vj) + 0.05
    return torch.where(hi > lo, minimum - radius, disjoint)


def _ball_interval(relative, velocity, radius, horizon):
    """Interval inside a moving-reference ball (a vertical band in 1D)."""
    a = velocity.square().sum(dim=-1)
    b = (relative * velocity).sum(dim=-1)
    c = relative.square().sum(dim=-1) - radius**2
    moving = a > 1e-12
    discriminant = b.square() - a * c
    root = discriminant.clamp_min(torch.finfo(a.dtype).tiny).sqrt()
    safe_a = a.clamp_min(1e-12)
    lo = torch.where(moving, (-b - root) / safe_a, 0.0).clamp_min(0)
    hi = torch.where(moving, (-b + root) / safe_a, horizon).clamp_max(horizon)
    has_interval = torch.where(moving, discriminant > 0, c < 0) & (hi > lo)
    return lo, hi, has_interval


def cylinder_interval(relative, velocity, horizontal_radius, vertical_radius, horizon):
    horizontal = _ball_interval(relative[..., :2], velocity[..., :2], horizontal_radius, horizon)
    vertical = _ball_interval(relative[..., 2:], velocity[..., 2:], vertical_radius, horizon)
    lo = torch.maximum(horizontal[0], vertical[0])
    hi = torch.minimum(horizontal[1], vertical[1])
    valid = horizontal[2] & vertical[2] & (hi > lo)
    return torch.where(valid, lo, 0.0), torch.where(valid, hi, 0.0), valid


def native_vertical_radius(altitude_bound, radius):
    """Match the native detector's floating-point tolerance at the layer boundary."""
    scale = torch.maximum(altitude_bound.unsqueeze(-1), altitude_bound.unsqueeze(-2))
    return radius - 4.0 * torch.finfo(altitude_bound.dtype).eps * scale


@dataclass
class ConflictGraph:
    adjacency: torch.Tensor
    existing: torch.Tensor
    potential: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    closest_horizontal: torch.Tensor
    cross_layer: torch.Tensor


def conflict_graph(position, velocity, active, *, horizon=120.0, detection=20_000.0,
                   horizontal_radius=5000.0, vertical_radius=91.44, speed_slack=25.0):
    """Build observed plus conservatively reachable cylindrical conflicts.

    The potential mask bounds either aircraft's horizontal speed deviation.
    Cross-layer edges include one +/-300 ft target change by either aircraft.
    """
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    change = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    nodes = active.shape[-1]
    pairs = active.unsqueeze(-1) & active.unsqueeze(-2)
    pairs = pairs & ~torch.eye(nodes, device=active.device, dtype=torch.bool)
    nearby = pairs & (torch.linalg.vector_norm(relative[..., :2], dim=-1) <= detection)
    lo, hi, hit = cylinder_interval(relative, change, horizontal_radius, vertical_radius, horizon)
    closest, _, _ = closest_approach(relative[..., :2], change[..., :2], horizon)
    end_vertical = (relative[..., 2] + change[..., 2] * horizon).abs()
    start_vertical = relative[..., 2].abs()
    passes_altitude = relative[..., 2] * (relative[..., 2] + change[..., 2] * horizon) <= 0
    minimum_vertical = torch.where(passes_altitude, 0.0, torch.minimum(start_vertical, end_vertical))
    reachable = closest < horizontal_radius + 2 * speed_slack * horizon
    existing = nearby & hit
    potential = nearby & reachable & (minimum_vertical < vertical_radius + 91.44) & ~existing
    cross_layer = potential & (minimum_vertical >= vertical_radius)
    return ConflictGraph(existing | potential, existing, potential, torch.where(existing, lo, 0.0),
                         torch.where(existing, hi, 0.0), closest, cross_layer)


def connected_components(adjacency):
    nodes = adjacency.shape[-1]
    labels = torch.arange(nodes, device=adjacency.device).expand(adjacency.shape[:-1])
    for _ in range(nodes - 1):
        neighbors = torch.where(adjacency, labels.unsqueeze(-2), nodes)
        labels = torch.minimum(labels, neighbors.amin(dim=-1))
    return labels


def greedy_colors(adjacency):
    nodes = adjacency.shape[-1]
    colors = torch.full(adjacency.shape[:-1], -1, device=adjacency.device, dtype=torch.long)
    candidate = torch.arange(nodes, device=adjacency.device).unsqueeze(-1)
    for i in range(nodes):
        used = (adjacency[..., i, :].unsqueeze(-2) & (colors.unsqueeze(-2) == candidate)).any(dim=-1)
        colors[..., i] = (~used).to(torch.int64).argmax(dim=-1)
    return colors
