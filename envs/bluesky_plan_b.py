"""Accelerator-resident Geo-ADMM rollout adapter for TorchVecAirspaceEnv."""

from dataclasses import fields, replace
from math import cos

import torch
from torch.nn.functional import one_hot

from bluesky.rl_gpu import TorchAirspaceState, TorchVecAirspaceEnv
from bluesky.rl_gpu.aero_torch import FT, KTS, vcas2tas_torch
from bluesky.rl_gpu.physics_torch import bearing_deg_torch, haversine_km_torch
from core.conflict_graph import cylinder_interval, native_vertical_radius
from envs.bluesky_graph import METRES_PER_DEGREE, RADIANS, SCALE, traffic_graph
from envs.bluesky_traffic import sample_routes


class GeoTorchBlueSky(TorchVecAirspaceEnv):
    def __init__(self, config, *, family="upstream", perf=None, audit_physics=False, timed_departures=False,
                 correlated_yaw=False,
                 detection_metres=SCALE.detection, geodesic_navigation=False, cross_track_observation=False,
                 native_contact_tolerance=False, event_checks_per_decision=1, far_conflict_observation=False,
                 selected_altitude_observation=False):
        super().__init__(config, perf=perf)
        self.route_count = torch.zeros_like(self.step_counts, dtype=torch.long)
        if event_checks_per_decision < 1 or self.substeps % event_checks_per_decision:
            raise ValueError("Event checks per decision must divide the physics substeps")
        self.family = family
        self.audit_physics = audit_physics
        self.event_checks_per_decision = self.substeps if audit_physics else event_checks_per_decision
        self.timed_departures = timed_departures
        self.correlated_yaw = correlated_yaw
        self.observation_scale = replace(SCALE, detection=detection_metres)
        self.geodesic_navigation = geodesic_navigation
        self.cross_track_observation = cross_track_observation
        self.far_conflict_observation = far_conflict_observation
        self.selected_altitude_observation = selected_altitude_observation
        self.native_contact_tolerance = native_contact_tolerance

    def _sample_envs(self, num_envs):
        if self.family == "upstream":
            state = super()._sample_envs(num_envs)
            self.fresh_category = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self.fresh_route_count = self.fresh_category
            return state
        cfg = self.cfg
        route = sample_routes(self.family, num_envs, cfg.max_aircraft, cfg.center_lat, cfg.center_lon,
                              cfg.airspace_half_km, self.gen, self.device, self.dtype,
                              timed_departures=self.timed_departures, correlated_yaw=self.correlated_yaw)
        alt, cas = route.altitude_ft * FT, route.cas_kts * KTS
        tas = vcas2tas_torch(cas, alt)
        heading = bearing_deg_torch(route.latitude, route.longitude, route.goal_latitude, route.goal_longitude)
        heading = (heading + route.yaw_deg) % 360.0
        angle = torch.deg2rad(heading)
        zero = torch.zeros_like(cas)
        self.fresh_category = route.category
        self.fresh_route_count = route.route_count
        self.fresh_departures = route.departure_seconds
        self.fresh_pending = route.active & (route.departure_seconds > 0)
        active = route.active & ~self.fresh_pending
        lnav = route.active & (route.yaw_deg == 0)
        return TorchAirspaceState(route.latitude, route.longitude, alt, tas, cas, heading, zero,
                                  tas * angle.cos(), tas * angle.sin(), tas, cas, alt, heading, zero,
                                  lnav, route.goal_latitude, route.goal_longitude, active)

    def reset(self):
        self.state = self._sample_envs(self.cfg.num_envs)
        if self.selected_altitude_observation:
            self.route_altitude = self.state.alt.clone()
        self.step_counts.zero_()
        self.origins = torch.stack((self.state.lat, self.state.lon), dim=-1)
        self.category = self.fresh_category
        self.route_count = self.fresh_route_count
        if self.timed_departures:
            self.pending = self.fresh_pending
            self.departures = self.fresh_departures
            self.launched_aircraft = self.state.active.sum()
        self.traffic_draws = one_hot(self.category, 3).sum(dim=0)
        return self.get_observations()

    def _reset_masked_envs(self, done_mask):
        # The upstream bool(done_mask.any()) synchronizes the device on every step.
        fresh = self._sample_envs(self.cfg.num_envs)
        mask = done_mask.unsqueeze(-1)
        self.state = TorchAirspaceState(**{
            field.name: torch.where(mask, getattr(fresh, field.name), getattr(self.state, field.name))
            for field in fields(TorchAirspaceState)
        })
        origins = torch.stack((fresh.lat, fresh.lon), dim=-1)
        self.origins = torch.where(mask.unsqueeze(-1), origins, self.origins)
        if self.selected_altitude_observation:
            self.route_altitude = torch.where(mask, fresh.alt, self.route_altitude)
        self.category = torch.where(done_mask, self.fresh_category, self.category)
        self.route_count = torch.where(done_mask, self.fresh_route_count, self.route_count)
        if self.timed_departures:
            self.pending = torch.where(mask, self.fresh_pending, self.pending)
            self.departures = torch.where(mask, self.fresh_departures, self.departures)
            self.launched_aircraft += (fresh.active & mask).sum()
        self.traffic_draws += (one_hot(self.fresh_category, 3) * mask).sum(dim=0)
        self.step_counts = torch.where(done_mask, 0, self.step_counts)

    def _commands(self, state, actions):
        cfg = self.cfg
        cas, alt, heading = actions.masked_fill(~state.active.unsqueeze(-1), 0.0).unbind(dim=-1)
        cas_target = (state.cas / KTS + cas * cfg.max_speed_delta_kts).clamp(*cfg.cas_bounds_kts) * KTS
        alt_target = (state.alt / FT + alt * cfg.max_alt_delta_ft).clamp(*cfg.alt_bounds_ft) * FT
        offset = heading * cfg.max_hdg_delta_deg
        turn = offset.abs() > cfg.hdg_deadband_deg
        return replace(state, sel_cas=torch.where(cas != 0, cas_target, state.sel_cas),
                       sel_alt=torch.where(alt != 0, alt_target, state.sel_alt),
                       sel_hdg=torch.where(turn, (state.hdg + offset).remainder(360), state.sel_hdg),
                       swlnav=state.swlnav & ~turn)

    def _events(self, start, end, seconds, route=None):
        cfg = self.cfg
        metre_lon = METRES_PER_DEGREE * cos(cfg.center_lat * RADIANS)

        def position(state):
            return torch.stack(((state.lon - cfg.center_lon) * metre_lon,
                                (state.lat - cfg.center_lat) * METRES_PER_DEGREE, state.alt), dim=-1)

        origin, dest = position(start), position(end)
        delta = (dest - origin) / seconds
        rel = origin.unsqueeze(-2) - origin.unsqueeze(-3)
        velocity = delta.unsqueeze(-2) - delta.unsqueeze(-3)
        vertical_radius = cfg.collision_v_ft * FT - 1e-4
        if self.native_contact_tolerance:
            altitude_bound = torch.maximum(start.alt.abs(), end.alt.abs())
            vertical_radius = native_vertical_radius(altitude_bound, cfg.collision_v_ft * FT)
        _, _, contact = cylinder_interval(rel, velocity, cfg.collision_h_km * 1000, vertical_radius, seconds)
        pairs = start.active.unsqueeze(-1) & start.active.unsqueeze(-2) & self.distinct
        collision = (contact & pairs).any(dim=-1)
        arrival = start.active & ~collision & (
            haversine_km_torch(end.lat, end.lon, end.goal_lat, end.goal_lon) < cfg.arrival_dist_km
        )
        lat0, lon0 = self.origins.unbind(dim=-1)
        if route is None:
            route = bearing_deg_torch(lat0, lon0, end.goal_lat, end.goal_lon)
        travel = bearing_deg_torch(lat0, lon0, end.lat, end.lon)
        arc = haversine_km_torch(lat0, lon0, end.lat, end.lon) / 6371.0
        cross_track = (arc.sin() * ((travel - route) * RADIANS).sin()).clamp(-1, 1).asin().abs() * 6371.0
        exit_route = start.active & ~collision & ~arrival & (cross_track > 20.0)
        return collision, arrival, exit_route

    def step(self, actions):
        # Keep the original kernel path for historical throughput comparisons.
        groups = self.event_checks_per_decision
        if self.family != "full_mixture" and groups == 1 and not (self.audit_physics or self.native_contact_tolerance):
            return super().step(actions)
        self.distinct = ~torch.eye(actions.shape[-2], dtype=torch.bool, device=self.device)
        state = self._commands(self.state, actions.to(self.dtype))
        # Goals and initial positions hold throughout the physics groups of one decision.
        route = bearing_deg_torch(*self.origins.unbind(dim=-1), state.goal_lat, state.goal_lon)
        collided, arrived, exits = (torch.zeros_like(state.active) for _ in range(3))
        ticks = self.substeps // groups
        for group in range(groups):
            next_state = self.core.step_physics(state, simdt=self.cfg.sim_dt, n_substeps=ticks)
            new_collision, new_arrival, new_exit = self._events(state, next_state, ticks * self.cfg.sim_dt, route)
            collided |= new_collision
            arrived |= new_arrival
            exits |= new_exit
            state = replace(next_state, active=state.active & ~(new_collision | new_arrival | new_exit))
            # Changing contact sampling should not also shift the scheduled traffic.
            if self.timed_departures and (self.audit_physics or group + 1 == groups):
                elapsed = (self.step_counts * self.substeps + (group + 1) * ticks) * self.cfg.sim_dt
                launched = self.pending & (self.departures <= elapsed.unsqueeze(-1))
                self.pending = self.pending & ~launched
                self.launched_aircraft += launched.sum()
                angle = state.hdg * RADIANS
                state = replace(state, active=state.active | launched,
                                gs=torch.where(launched, state.tas, state.gs),
                                gsnorth=torch.where(launched, state.tas * angle.cos(), state.gsnorth),
                                gseast=torch.where(launched, state.tas * angle.sin(), state.gseast))
        self.state = state
        self.step_counts += 1
        remaining = state.active | self.pending if self.timed_departures else state.active
        timeout = (self.step_counts >= self.cfg.max_steps).unsqueeze(-1) & remaining
        done = (self.step_counts >= self.cfg.max_steps) | ~remaining.any(dim=-1)
        reward = (-0.1 * actions.abs().sum(dim=-1) + 200.0 * arrived - 50.0 * (collided | exits)
                  - 20.0 * timeout)
        info = {"arrivals": arrived.sum(dim=-1), "collisions": collided.sum(dim=-1), "exits": exits.sum(dim=-1),
                "collision_aircraft": collided,
                "timeouts": timeout.sum(dim=-1), "category": self.category, "route_count": self.route_count}
        self._reset_masked_envs(done)
        return self.get_observations(), reward, done, info

    def get_observations(self):
        s = self.state
        graph, self.physical = traffic_graph(
            latitude=s.lat, longitude=s.lon, altitude=s.alt, heading=s.hdg, east=s.gseast,
            north=s.gsnorth, vertical=s.vs, goal_latitude=s.goal_lat, goal_longitude=s.goal_lon,
            planned_altitude=self.route_altitude if self.selected_altitude_observation else s.sel_alt,
            lnav=s.swlnav, active=s.active,
            center_latitude=self.cfg.center_lat, center_longitude=self.cfg.center_lon,
            airspace_half_km=self.cfg.airspace_half_km, cas=s.cas, tas=s.tas, selected_altitude=s.sel_alt,
            scale=self.observation_scale,
            geodesic_navigation=self.geodesic_navigation,
            route_origin=self.origins.unbind(dim=-1) if self.cross_track_observation else None,
            far_conflict_observation=self.far_conflict_observation,
            selected_altitude_observation=self.selected_altitude_observation,
        )
        return graph
