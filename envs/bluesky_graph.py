"""Map either native or Torch BlueSky traffic to the Geo-ADMM observation."""

from dataclasses import dataclass, replace
from math import cos, isfinite, pi

import torch

from core.conflict_graph import closest_approach
from envs.observations import LAYER, ObservationScale, observe

RADIANS = pi / 180.0
EARTH_RADIUS_KM = 6371.0
METRES_PER_DEGREE = 1000.0 * EARTH_RADIUS_KM * RADIANS
SCALE = ObservationScale(speed_min=145.0, speed_max=220.0, nominal_speed=183.0, horizon=120.0,
                         horizontal_radius=5000.0, detection=20000.0)
ALTITUDE_REFERENCE = 20000.0 * 0.3048
FAR_CONFLICT_FEATURES = slice(15, 21)
SELECTED_ALTITUDE_NODE_FEATURE = 4
SELECTED_ALTITUDE_EDGE_FEATURES = (13, 17)


@dataclass(frozen=True)
class ObservationProfile:
    detection_metres: float = SCALE.detection
    geodesic_navigation: bool = False
    cross_track_observation: bool = False
    far_conflict_observation: bool = False
    selected_altitude_observation: bool = False

    def __post_init__(self):
        if not isfinite(self.detection_metres) or self.detection_metres <= 0:
            raise ValueError("Observation distance must be positive and finite")
        if not isinstance(self.geodesic_navigation, bool):
            raise ValueError("Geodesic navigation must be a boolean")
        if not isinstance(self.cross_track_observation, bool):
            raise ValueError("Cross-track observation must be a boolean")
        if not isinstance(self.far_conflict_observation, bool):
            raise ValueError("Far conflict observation must be a boolean")
        if not isinstance(self.selected_altitude_observation, bool):
            raise ValueError("Selected altitude observation must be a boolean")


def _bearing(latitude, target_latitude, longitude_delta):
    east = longitude_delta.sin() * target_latitude.cos()
    north = latitude.cos() * target_latitude.sin() - latitude.sin() * target_latitude.cos() * longitude_delta.cos()
    return torch.atan2(east, north)


def cross_track_km(latitude, longitude, goal_latitude, goal_longitude, origin_latitude, origin_longitude):
    start, current, goal = (value * RADIANS for value in (origin_latitude, latitude, goal_latitude))
    travelled_longitude = (longitude - origin_longitude) * RADIANS
    goal_longitude = (goal_longitude - origin_longitude) * RADIANS
    route = _bearing(start, goal, goal_longitude)
    travel = _bearing(start, current, travelled_longitude)
    arc_half = ((current - start).mul(0.5).sin().square()
                + start.cos() * current.cos() * travelled_longitude.mul(0.5).sin().square())
    arc = 2 * arc_half.clamp(0.0, 1.0).sqrt().asin()
    return (arc.sin() * (travel - route).sin()).clamp(-1.0, 1.0).asin() * EARTH_RADIUS_KM


def _far_horizontal_candidates(position, direction, speed, active, scale):
    horizontal = position[..., :2]
    velocity = direction[..., :2] * speed.unsqueeze(-1)
    displacement = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    relative_velocity = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    distance = displacement.norm(dim=-1)
    closest, time, _ = closest_approach(displacement, relative_velocity, scale.horizon)
    candidates = ((distance > scale.detection) & (distance <= 50_000.0)
                  & (closest < scale.horizontal_radius) & active.unsqueeze(-1) & active.unsqueeze(-2))
    return candidates, time


def with_far_horizontal_observation(graph, physical, detection_metres):
    """Add an early attention edge to the first predicted horizontal overlap for each aircraft."""
    nodes, edges, adjacency, active = graph
    position, direction, speed, *_ = physical
    candidates, time = _far_horizontal_candidates(position, direction, speed, active,
                                                   replace(SCALE, detection=detection_metres))
    earliest = time.masked_fill(~candidates, torch.inf).argmin(dim=-1, keepdim=True)
    selected = torch.zeros_like(candidates).scatter_(-1, earliest, candidates.gather(-1, earliest))
    return nodes, edges, adjacency | selected | selected.transpose(-1, -2), active


def _far_conflict_features(edge, position, direction, speed, active, vertical, target, scale):
    candidates, time = _far_horizontal_candidates(position, direction, speed, active, scale)
    height = position[..., 2]
    low, high = torch.minimum(height, target), torch.maximum(height, target)
    own = (height.unsqueeze(-1) + vertical.unsqueeze(-1) * time).clamp(low.unsqueeze(-1), high.unsqueeze(-1))
    other = (height.unsqueeze(-2) + vertical.unsqueeze(-2) * time).clamp(low.unsqueeze(-2), high.unsqueeze(-2))
    gap = other - own
    candidates = candidates & (gap.abs() < 2 * LAYER)
    index = time.masked_fill(~candidates, torch.inf).argmin(dim=-1, keepdim=True)
    geometry = torch.cat((edge[..., :2], edge[..., 3:5] / 4, (time / scale.horizon).unsqueeze(-1),
                          (gap / (2 * LAYER)).unsqueeze(-1)), dim=-1)
    selected = geometry.gather(-2, index.unsqueeze(-1).expand(*index.shape, geometry.shape[-1])).squeeze(-2)
    return selected.masked_fill(~candidates.any(dim=-1, keepdim=True), 0.0)


def traffic_graph(*, latitude, longitude, altitude, heading, east, north, vertical, goal_latitude, goal_longitude,
                  planned_altitude, lnav, active, center_latitude, center_longitude, airspace_half_km,
                  cas=None, tas=None, selected_altitude=None, scale=SCALE, geodesic_navigation=False, route_origin=None,
                  far_conflict_observation=False, selected_altitude_observation=False, include_potential=True):
    meter_lon = METRES_PER_DEGREE * cos(center_latitude * RADIANS)
    relative_longitude = torch.remainder(longitude - center_longitude + 180.0, 360.0) - 180.0
    position = torch.stack((relative_longitude * meter_lon, (latitude - center_latitude) * METRES_PER_DEGREE,
                            altitude - ALTITUDE_REFERENCE), dim=-1).float()
    course = torch.stack((east, north), dim=-1).float()
    speed = torch.linalg.vector_norm(course, dim=-1)
    heading = heading.float() * RADIANS
    heading_vector = torch.stack((heading.sin(), heading.cos()), dim=-1)
    direction = torch.where((speed > 1e-5).unsqueeze(-1), course / speed.clamp_min(1e-5).unsqueeze(-1), heading_vector)
    node, edge, graph = observe(position, direction, speed, active, reference_altitude=0.0,
                                scale=scale, vertical_speed=vertical.float(), include_potential=include_potential)
    longitude_delta = torch.remainder(goal_longitude - longitude + 180.0, 360.0) - 180.0
    goal_x = longitude_delta * meter_lon
    goal_y = (goal_latitude - latitude) * METRES_PER_DEGREE
    goal_distance = torch.sqrt(goal_x.square() + goal_y.square())
    if geodesic_navigation:
        source_lat = latitude * RADIANS
        target_lat = goal_latitude * RADIANS
        longitude_delta = longitude_delta * RADIANS
        heading_delta = _bearing(source_lat, target_lat, longitude_delta) - heading
    else:
        heading_delta = torch.atan2(goal_x, goal_y) - heading
    node[..., 3] = (altitude - planned_altitude) / LAYER
    node[..., 5] = heading_delta.sin()
    node[..., 6] = heading_delta.cos()
    node[..., 7] = goal_distance / (2_000.0 * airspace_half_km)
    node[..., 8] = lnav
    if route_origin is not None:
        node[..., 9] = cross_track_km(latitude, longitude, goal_latitude, goal_longitude, *route_origin) / 20.0
    if far_conflict_observation:
        target = planned_altitude if selected_altitude is None else selected_altitude
        node[..., FAR_CONFLICT_FEATURES] = _far_conflict_features(edge, position, direction, speed, active,
                                                                 vertical.float(),
                                                                 (target - ALTITUDE_REFERENCE).float(), scale)
    if selected_altitude_observation:
        if selected_altitude is None:
            raise ValueError("Selected altitude observation requires the autopilot target")
        node[..., SELECTED_ALTITUDE_NODE_FEATURE] = (selected_altitude - altitude) / LAYER
        relative_target = (selected_altitude.unsqueeze(-2) - selected_altitude.unsqueeze(-1)) / LAYER
        target_channel, rate_channel = SELECTED_ALTITUDE_EDGE_FEATURES
        edge[..., target_channel] = relative_target.masked_fill(~graph.adjacency, 0)
        edge[..., rate_channel] = (vertical.unsqueeze(-2) / 7.62).masked_fill(~graph.adjacency, 0)
    node = node.masked_fill(~active.unsqueeze(-1), 0.0)
    result = (node, edge, graph.adjacency, active)
    if cas is None:
        return result
    physical = (position, direction, speed, vertical.float(), cas.float(), tas.float(),
                (selected_altitude - ALTITUDE_REFERENCE).float())
    return result, physical
