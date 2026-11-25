"""Ego-frame observations for the first, constant-heading speed curriculum."""

from dataclasses import dataclass, replace

import torch

from core.conflict_graph import conflict_graph, greedy_colors
from core.rstd import speed_forbidden_interval

SPEED_MIN = 5.0
SPEED_MAX = 20.0
SPEED_UNIT = (SPEED_MAX - SPEED_MIN) / 2
HORIZON = 80.0
RADIUS = 50.0
LAYER = 91.44


@dataclass(frozen=True)
class ObservationScale:
    speed_min: float = SPEED_MIN
    speed_max: float = SPEED_MAX
    nominal_speed: float = 12.0
    horizon: float = HORIZON
    horizontal_radius: float = RADIUS
    detection: float = 2 * SPEED_MAX * HORIZON

    @property
    def speed_unit(self):
        return (self.speed_max - self.speed_min) / 2

    @property
    def distance_unit(self):
        return self.speed_max * self.horizon


def _ego_frame(vectors, axis):
    x, y = axis[..., 0:1], axis[..., 1:2]
    along = x * vectors[..., 0] + y * vectors[..., 1]
    cross = x * vectors[..., 1] - y * vectors[..., 0]
    return torch.stack((along, cross, vectors[..., 2]), dim=-1)


def observe(position, direction, speed, active, *, reference_altitude=6000.0, scale=ObservationScale(),
            vertical_speed=None, include_potential=True):
    vertical_speed = torch.zeros_like(speed) if vertical_speed is None else vertical_speed
    velocity = torch.cat((direction[..., :2] * speed.unsqueeze(-1), vertical_speed.unsqueeze(-1)), dim=-1)
    graph = conflict_graph(position, velocity, active, horizon=scale.horizon, detection=scale.detection,
                           horizontal_radius=scale.horizontal_radius, speed_slack=scale.speed_unit)
    if not include_potential:
        excluded = torch.zeros_like(graph.potential)
        graph = replace(graph, adjacency=graph.existing, potential=excluded, cross_layer=excluded)
    relative_position = position.unsqueeze(-3) - position.unsqueeze(-2)
    relative_velocity = velocity.unsqueeze(-3) - velocity.unsqueeze(-2)
    local_position = _ego_frame(relative_position, direction)
    local_velocity = _ego_frame(relative_velocity, direction)
    altitude_delta = local_position[..., 2]
    layer = torch.isclose(altitude_delta.abs(), torch.full_like(altitude_delta, LAYER), atol=0.01)
    same_position_window = relative_position[..., :2].norm(dim=-1) < scale.detection
    neighbor_layer = layer & same_position_window & active.unsqueeze(-2)
    degree = graph.adjacency.sum(dim=-1)
    shared_neighbors = graph.adjacency.float() @ graph.adjacency.float()
    triangles = (shared_neighbors * graph.adjacency).sum(dim=-1)
    density = triangles / (degree * (degree - 1)).clamp_min(1)
    channels = [
        speed / scale.speed_max, (speed - scale.nominal_speed) / scale.speed_unit,
        (position[..., 2] - reference_altitude) / LAYER,
        torch.zeros_like(speed), torch.zeros_like(speed), torch.zeros_like(speed), torch.ones_like(speed),
        torch.ones_like(speed), torch.ones_like(speed), torch.zeros_like(speed),
        (speed - scale.speed_min) / scale.speed_unit, (scale.speed_max - speed) / scale.speed_unit,
        (neighbor_layer & (altitude_delta > 0)).any(dim=-1), (neighbor_layer & (altitude_delta < 0)).any(dim=-1),
        torch.ones_like(speed), *(torch.zeros_like(speed) for _ in range(6)),
        degree / (active.sum(dim=-1, keepdim=True) - 1).clamp_min(1),
        density, greedy_colors(graph.adjacency) / max(speed.shape[-1] - 1, 1),
    ]
    node = torch.stack(channels, dim=-1).masked_fill(~active.unsqueeze(-1), 0.0)
    zero_edge = torch.zeros_like(altitude_delta)
    forbidden_low, forbidden_high, forbidden = speed_forbidden_interval(
        -relative_position, direction[..., None, :2], velocity[..., None, :, :2], -relative_velocity[..., 2],
        lower=scale.speed_min, upper=scale.speed_max, horizontal_radius=scale.horizontal_radius,
        vertical_radius=LAYER, horizon=scale.horizon,
    )
    speed_reference = speed.unsqueeze(-1)
    forbidden_low = torch.where(forbidden, (forbidden_low - speed_reference) / scale.speed_unit, 0.0)
    forbidden_high = torch.where(forbidden, (forbidden_high - speed_reference) / scale.speed_unit, 0.0)
    edge = torch.cat((
        local_position / position.new_tensor([scale.distance_unit, scale.distance_unit, LAYER]),
        local_velocity / scale.speed_unit,
        local_position[..., :2].norm(dim=-1, keepdim=True) / scale.distance_unit,
        altitude_delta.abs().unsqueeze(-1) / LAYER,
        torch.stack(((graph.closest_horizontal - scale.horizontal_radius) / scale.distance_unit,
                     graph.start / scale.horizon, graph.end / scale.horizon, graph.existing, graph.potential, zero_edge,
                     forbidden_low, forbidden_high, graph.cross_layer, zero_edge), dim=-1),
    ), dim=-1)
    return node, edge, graph
