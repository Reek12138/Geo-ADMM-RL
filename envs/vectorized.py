"""Device-resident independent airspaces for rollout and sim-to-sim calibration."""

from dataclasses import dataclass
from math import pi

import torch

from core.conflict_graph import cylinder_interval
from envs.observations import LAYER, RADIUS, SPEED_MAX, SPEED_MIN, observe


@dataclass(frozen=True)
class Dynamics:
    integration_seconds: float = 0.1
    substeps: int = 5
    max_decisions: int = 320
    acceleration: float = 2.0
    climb_rate: float = 6.096
    turn_rate: float = pi / 60
    boundary: float = 1150.0


@dataclass
class Transition:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    collisions: torch.Tensor
    arrivals: torch.Tensor
    exits: torch.Tensor


def wrap_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


class VectorAirspace:
    def __init__(self, count, aircraft, device, seed, *, dynamics=Dynamics(), transfer=False):
        if count < 1 or aircraft < 2 or dynamics.integration_seconds <= 0 or dynamics.substeps < 1:
            raise ValueError("An airspace needs two slots and a positive integration step")
        self.count = count
        self.aircraft = aircraft
        self.device = torch.device(device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.dynamics = dynamics
        self.transfer = transfer
        self.slots = torch.arange(aircraft, device=device)
        self.distinct = ~torch.eye(aircraft, device=device, dtype=torch.bool)
        self.reset()

    def _uniform(self, shape, low, high):
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self.rng)

    def reset(self, done=None):
        shape = (self.count, self.aircraft)
        traffic = torch.randint(2, self.aircraft + 1, (self.count, 1), device=self.device, generator=self.rng)
        routes = torch.randint(2, min(self.aircraft, 8) + 1, (self.count, 1), device=self.device, generator=self.rng)
        angle = self._uniform((self.count, 1), -pi, pi)
        spread = self._uniform((self.count, 1), 0.6, 1.0)
        if self.transfer:
            spread = self._uniform((self.count, 1), 0.16, 0.42)
        corridor = self.slots % routes
        heading = angle + corridor * (2 * pi * spread / routes) + self._uniform(shape, -0.12, 0.12)
        axis = torch.stack((heading.cos(), heading.sin()), dim=-1)
        perpendicular = torch.stack((-axis[..., 1], axis[..., 0]), dim=-1)
        queue = self.slots // routes
        lane = self._uniform(shape, -40, 40) + 13 * corridor
        origin = lane.unsqueeze(-1) * perpendicular
        start = origin - (self._uniform(shape, 180, 255) + 85 * queue).unsqueeze(-1) * axis
        finish = origin + self._uniform(shape, 260, 440).unsqueeze(-1) * axis
        altitude = LAYER * (self._uniform(shape, 0.0, 1.0) >= 0.72)
        position = torch.cat((start, altitude.unsqueeze(-1)), dim=-1)
        speed = self._uniform(shape, 10, 14)
        wind = self._uniform((self.count, 1, 2), -1.0, 1.0)
        if self.transfer:
            wind = wind * 3
        state = {"position": position, "destinations": finish, "heading": heading, "course": heading,
                 "speed": speed, "nominal_speed": speed, "nominal_altitude": altitude, "wind": wind,
                 "active": self.slots < traffic, "decisions": torch.zeros(self.count, device=self.device, dtype=torch.long),
                 "width": self._uniform((self.count, 1), 0.75, 1.25)}
        for field, value in state.items():
            if done is not None:
                condition = done.reshape(self.count, *(1 for _ in value.shape[1:]))
                value = torch.where(condition, value, getattr(self, field))
            setattr(self, field, value)

    def observation(self):
        heading = self.heading
        axis = torch.stack((heading.cos(), heading.sin(), torch.zeros_like(heading)), dim=-1)
        node, edge, graph = observe(self.position, axis, self.speed, self.active, reference_altitude=0)
        to_goal = self.destinations - self.position[..., :2]
        target_delta = wrap_angle(torch.atan2(to_goal[..., 1], to_goal[..., 0]) - self.heading)
        node[..., 1] = (self.speed - self.nominal_speed) / (SPEED_MAX - SPEED_MIN)
        node[..., 3] = (self.position[..., 2] - self.nominal_altitude) / LAYER
        node[..., 5] = target_delta.sin()
        node[..., 6] = target_delta.cos()
        node[..., 7] = to_goal.norm(dim=-1) / self.dynamics.boundary
        node = node.masked_fill(~self.active.unsqueeze(-1), 0.0)
        return node, edge, graph.adjacency, self.active

    def _integrate(self, target_speed, target_altitude, target_heading):
        dt = self.dynamics.integration_seconds
        speed_step = self.dynamics.acceleration * self.width * dt
        next_speed = self.speed + (target_speed - self.speed).clamp(-speed_step, speed_step)
        turn_step = self.dynamics.turn_rate * self.width * dt
        next_heading = self.heading + wrap_angle(target_heading - self.heading).clamp(-turn_step, turn_step)
        advance_angle = (self.heading + next_heading) * 0.5
        advance_speed = (self.speed + next_speed) * 0.5
        axis = torch.stack((advance_angle.cos(), advance_angle.sin()), dim=-1)
        dx = (axis * advance_speed.unsqueeze(-1) + self.wind) * dt
        climb_step = self.dynamics.climb_rate * self.width * dt
        dz = (target_altitude - self.position[..., 2]).clamp(-climb_step, climb_step)
        delta = torch.cat((dx, dz.unsqueeze(-1)), dim=-1).masked_fill(~self.active.unsqueeze(-1), 0.0)
        relative = self.position.unsqueeze(-2) - self.position.unsqueeze(-3)
        change = (delta.unsqueeze(-2) - delta.unsqueeze(-3)) / dt
        _, _, conflict = cylinder_interval(relative, change, RADIUS, LAYER, dt)
        contact = (conflict & self.active.unsqueeze(-1) & self.active.unsqueeze(-2) & self.distinct).any(dim=-1)
        self.position = self.position + delta
        self.speed = torch.where(self.active, next_speed, self.speed)
        self.heading = torch.where(self.active, next_heading, self.heading)
        return contact

    def step(self, action):
        """Apply normalized speed, altitude and heading commands for one decision."""
        action = action.clamp(-1.0, 1.0)
        before = (self.destinations - self.position[..., :2]).norm(dim=-1)
        target_speed = (self.nominal_speed + 7.5 * action[..., 0]).clamp(SPEED_MIN, SPEED_MAX)
        target_altitude = self.nominal_altitude + LAYER * action[..., 1]
        target_heading = self.course + (pi / 6) * action[..., 2]
        collisions = torch.zeros_like(self.active)
        for _ in range(self.dynamics.substeps):
            collisions = collisions | self._integrate(target_speed, target_altitude, target_heading)
        distance = (self.destinations - self.position[..., :2]).norm(dim=-1)
        arrivals = self.active & ~collisions & (distance <= 18.0)
        exits = self.active & ~collisions & ~arrivals & (self.position[..., :2].norm(dim=-1) > self.dynamics.boundary)
        cost = (action.square() * action.new_tensor([0.015, 0.06, 0.12])).sum(dim=-1)
        progress = (before - distance) / (SPEED_MAX * self.dynamics.integration_seconds * self.dynamics.substeps)
        reward = (progress - cost + 20 * arrivals - 30 * collisions - 20 * exits).masked_fill(~self.active, 0.0)
        terminated = collisions | arrivals | exits
        self.decisions = self.decisions + 1
        self.active = self.active & ~terminated
        truncated = self.decisions >= self.dynamics.max_decisions
        return Transition(reward, terminated, truncated, collisions, arrivals, exits)
