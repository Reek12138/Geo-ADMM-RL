"""Independent Geo-ADMM route families shared by both simulation adapters."""

from dataclasses import dataclass
from math import cos, pi

import torch

from envs.bluesky_graph import METRES_PER_DEGREE, RADIANS


FAMILIES = ("upstream", "mixed", "full_mixture", "crossing3", "crossing4", "crossing6", "head_on",
            "yaw_plus15", "yaw_minus15")
MIXTURE_NAMES = ("crossing", "head_on", "yaw")
HELD_TOPOLOGIES = ("crossing13", "crossing15", "crossing17")
_FIXED_ROUTE_COUNTS = {
    "crossing3": 3, "crossing4": 4, "crossing6": 6, "crossing13": 13, "crossing15": 15,
    "crossing17": 17, "head_on": 2, "yaw_plus15": 6, "yaw_minus15": 6,
}


@dataclass
class Routes:
    latitude: torch.Tensor
    longitude: torch.Tensor
    goal_latitude: torch.Tensor
    goal_longitude: torch.Tensor
    altitude_ft: torch.Tensor
    cas_kts: torch.Tensor
    yaw_deg: torch.Tensor
    active: torch.Tensor
    category: torch.Tensor
    departure_seconds: torch.Tensor
    route_count: torch.Tensor


def sample_routes(family, envs, aircraft, center_lat, center_lon, half_km, rng, device, dtype=torch.float32,
                  *, timed_departures=False, correlated_yaw=False, training_yaw_degrees=None):
    if timed_departures and family != "full_mixture":
        raise ValueError("Timed departures are enabled only for the full training mixture")
    if correlated_yaw and family != "full_mixture":
        raise ValueError("Correlated yaw is enabled only for the full training mixture")
    slots = torch.arange(aircraft, device=device)
    mixed = family in ("mixed", "full_mixture")
    category = torch.zeros((envs, 1), device=device, dtype=torch.long)
    if family == "full_mixture":
        draw = torch.rand((envs, 1), device=device, generator=rng)
        category = (draw >= 0.4).long() + (draw >= 0.7).long()
        fleet = torch.randint(min(8, aircraft), aircraft + 1, (envs, 1), device=device, generator=rng)
        choices = slots.new_tensor([3, 4, 6, 8])
        routes = choices[torch.randint(4, (envs, 1), device=device, generator=rng)]
        routes = torch.where(category == 1, 2, routes).clamp_max(aircraft)
    elif mixed:
        fleet = torch.randint(min(6, aircraft), aircraft + 1, (envs, 1), device=device, generator=rng)
        routes = torch.randint(min(2, aircraft), min(12, aircraft) + 1, (envs, 1), device=device, generator=rng)
        routes = torch.minimum(routes, fleet)
    else:
        count = _FIXED_ROUTE_COUNTS[family]
        if family in HELD_TOPOLOGIES and aircraft < count:
            raise ValueError(f"{family} requires at least {count} aircraft to cover every route")
        routes = torch.full((envs, 1), min(count, aircraft), device=device)
        fleet = torch.full((envs, 1), aircraft, device=device)

    def uniform(shape, low, high):
        return low + (high - low) * torch.rand(shape, device=device, generator=rng, dtype=dtype)

    azimuth = uniform((envs, 1), 0.0, 2 * pi) + 2 * pi * (slots % routes) / routes
    shape = (envs, aircraft)
    opposing = (category == 1) | (family == "head_on")
    if family != "head_on":
        jitter = 0.15 if family == "full_mixture" else 0.075
        azimuth = azimuth + uniform(shape, -jitter, jitter) * torch.where(opposing, 0.28, 1.0)
    along = torch.stack((azimuth.sin(), azimuth.cos()), dim=-1)
    normal = torch.stack((along[..., 1], -along[..., 0]), dim=-1)
    radius = 1_000.0 * half_km
    departure_seconds = torch.zeros(shape, device=device, dtype=dtype)
    if timed_departures:
        lane = slots % routes
        earlier = (lane.unsqueeze(-1) == lane.unsqueeze(-2)) & (slots.unsqueeze(-1) >= slots)
        intervals = uniform(shape, 48.0, 80.0).masked_fill(slots < routes, 0.0)
        departure_seconds = (earlier * intervals.unsqueeze(-2)).sum(dim=-1)
        start_radius = radius * uniform((envs, 1), 0.54, 0.62)
    else:
        spacing = uniform((envs, 1), 9_000.0, 15_000.0) if mixed else 11_000.0
        start_radius = radius * uniform((envs, 1), 0.54, 0.62) + (slots // routes) * spacing
    start_radius = start_radius + uniform(shape, -1_000.0, 1_000.0)
    goal_radius = radius * uniform(shape, 0.70, 0.80)
    sway = uniform(shape, -1_250.0, 1_250.0) if family == "head_on" else uniform(shape, -7_500.0, 7_500.0)
    if family == "full_mixture":
        sway = torch.where(opposing, sway / 6.0, sway)
    entrance = start_radius.unsqueeze(-1) * along + sway.unsqueeze(-1) * normal
    exit = -goal_radius.unsqueeze(-1) * along + sway.unsqueeze(-1) * normal
    metres_lon = METRES_PER_DEGREE * cos(center_lat * RADIANS)
    lat = center_lat + entrance[..., 1] / METRES_PER_DEGREE
    lon = center_lon + entrance[..., 0] / metres_lon
    goal_lat = center_lat + exit[..., 1] / METRES_PER_DEGREE
    goal_lon = center_lon + exit[..., 0] / metres_lon
    layer = torch.randint(0, 4, shape, device=device, generator=rng)
    altitude = (20000 + 300 * layer).to(dtype=dtype)
    altitude = torch.where(opposing, 20600.0, altitude)
    cas = uniform(shape, 235.0, 265.0)
    yaw = torch.full_like(cas, {"yaw_plus15": 15.0, "yaw_minus15": -15.0}.get(family, 0.0))
    if family == "mixed":
        disturbed = uniform(shape, 0.0, 1.0) < 0.12
        yaw = torch.where(disturbed, uniform(shape, -17.5, 17.5), yaw)
        scene_disturbance = uniform((envs, 1), 0.0, 1.0) < 0.18
        shared_sign = torch.where(uniform((envs, 1), 0.0, 1.0) < 0.5, -1.0, 1.0)
        yaw = torch.where(scene_disturbance, shared_sign * uniform(shape, 8.0, 19.0), yaw)
    elif family == "full_mixture":
        magnitude = (uniform(shape, 1.0, 20.0) if training_yaw_degrees is None
                     else torch.full_like(cas, training_yaw_degrees))
        draw = uniform(shape, 0.0, 1.0)
        positive = draw >= 0.5
        if correlated_yaw:
            # The first draw supplies independent sign and group bits without shifting other traffic draws.
            first = draw[..., :1]
            shared = (first < 0.25) | (first >= 0.75)
            positive = torch.where(shared, positive[..., :1], positive)
        yaw = torch.where(category == 2, torch.where(positive, magnitude, -magnitude), 0.0)
    return Routes(lat, lon, goal_lat, goal_lon, altitude, cas, yaw, slots < fleet, category.squeeze(-1),
                  departure_seconds, routes.squeeze(-1))
