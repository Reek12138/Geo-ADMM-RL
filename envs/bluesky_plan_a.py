"""Native BlueSky batched rollout adapter. One BlueSky simulator per worker process."""

from dataclasses import replace

import bluesky as bs
from bluesky.rl import BatchedBlueSkyEnv, set_commands_direct
from bluesky.rl.batched_env import _bearing_deg, _haversine_km
from bluesky.tools.aero import Rearth, ft, kts
import numpy as np
import torch

from envs.bluesky_graph import SCALE, traffic_graph
from envs.bluesky_traffic import sample_routes


class GeoNativeBlueSky(BatchedBlueSkyEnv):
    def __init__(self, config, *, workdir, family="upstream", timed_departures=False, correlated_yaw=False,
                 training_yaw_degrees=None,
                 detection_metres=SCALE.detection, geodesic_navigation=False, cross_track_observation=False,
                 far_conflict_observation=False, selected_altitude_observation=False, include_potential=True):
        bs.init(mode="sim", detached=True, workdir=str(workdir))
        super().__init__(config, init_bluesky=False)
        self.family = family
        self.timed_departures = timed_departures
        self.correlated_yaw = correlated_yaw
        self.training_yaw_degrees = training_yaw_degrees
        self.observation_scale = replace(SCALE, detection=detection_metres)
        self.geodesic_navigation = geodesic_navigation
        self.cross_track_observation = cross_track_observation
        self.far_conflict_observation = far_conflict_observation
        self.selected_altitude_observation = selected_altitude_observation
        self.include_potential = include_potential
        self.pending_departures = {}
        self.launched_aircraft = 0
        self.scene_category = np.zeros(config.num_envs, dtype=np.int64)
        self.scene_route_count = np.zeros_like(self.scene_category)
        self.layout_generator = torch.Generator().manual_seed(config.seed)
        self.traffic_draws = np.zeros((1, 3), dtype=np.int64)

    def _sample_geo_routes(self):
        cfg = self.cfg
        return sample_routes(self.family, 1, cfg.aircraft_per_env, cfg.center_lat, cfg.center_lon, cfg.airspace_half_km,
                             self.layout_generator, "cpu", timed_departures=self.timed_departures,
                             correlated_yaw=self.correlated_yaw, training_yaw_degrees=self.training_yaw_degrees)

    def _spawn_env_aircraft(self, env_id):
        if self.family == "upstream":
            return super()._spawn_env_aircraft(env_id)
        routes = self._sample_geo_routes()
        self.traffic_draws[0, int(routes.category[0])] += 1
        self.scene_category[env_id] = int(routes.category[0])
        self.scene_route_count[env_id] = int(routes.route_count[0])
        active = routes.active[0].numpy()
        slots = np.flatnonzero(active).astype(np.int32)
        lat, lon, goal_lat, goal_lon, altitude, speed = (
            value[0].numpy()[active].astype(np.float64)
            for value in (routes.latitude, routes.longitude, routes.goal_latitude, routes.goal_longitude,
                          routes.altitude_ft, routes.cas_kts)
        )
        yaw = routes.yaw_deg[0].numpy()[active]
        data = (slots, lat, lon, goal_lat, goal_lon, altitude, speed, yaw)
        release = routes.departure_seconds[0].numpy()[active]
        waiting = release > 0.0
        if waiting.any():
            self.pending_departures[env_id] = (bs.sim.simt + release[waiting],
                                               tuple(values[waiting] for values in data))
        else:
            self.pending_departures.pop(env_id, None)
        self._create_aircraft(env_id, *(values[~waiting] for values in data))
        self.env_step_counts[env_id] = 0

    def _release_due(self):
        for env_id, (schedule, data) in tuple(self.pending_departures.items()):
            due = schedule <= bs.sim.simt + 1e-9
            if not due.any():
                continue
            self._create_aircraft(env_id, *(values[due] for values in data))
            if due.all():
                del self.pending_departures[env_id]
            else:
                self.pending_departures[env_id] = schedule[~due], tuple(values[~due] for values in data)

    def _create_aircraft(self, env_id, slots, lat, lon, goal_lat, goal_lon, altitude, speed, yaw, *,
                         initial_heading=None, callsigns=None):
        cfg, traf = self.cfg, bs.traf
        n = len(slots)
        if not n:
            return
        heading = ((_bearing_deg(lat, lon, goal_lat, goal_lon) + yaw) % 360.0
                   if initial_heading is None else initial_heading)
        start_idx = traf.ntraf
        acids = (callsigns if callsigns is not None else
                 [f"E{env_id}A{index}" for index in range(self._uid_counter + 1, self._uid_counter + n + 1)])
        self._uid_counter += n
        self.launched_aircraft += n
        traf.cre(acids, actype=cfg.actype, aclat=lat, aclon=lon, achdg=heading, acalt=altitude * ft, acspd=speed * kts)
        for slot in range(n):
            idx = start_idx + slot
            route = traf.ap.route[idx]
            waypoint = route.addwpt_simple(idx, f"G{env_id}_{slots[slot]}", 0, goal_lat[slot], goal_lon[slot],
                                           altitude[slot] * ft, speed[slot] * kts)
            route.direct(idx, route.wpname[waypoint])
            if yaw[slot] != 0:
                traf.ap.selhdgcmd(idx, heading[slot])
                traf.swlnav[idx] = False
        columns = {
            "ac_env_id": np.full(n, env_id, dtype=np.int32), "ac_slot": slots,
            "ac_orig_lat": lat, "ac_orig_lon": lon, "ac_goal_lat": goal_lat, "ac_goal_lon": goal_lon,
            "ac_route_hdg": heading, "ac_target_alt_ft": altitude,
        }
        for name, values in columns.items():
            setattr(self, name, np.concatenate((getattr(self, name), values)))

    def _build_observations(self):
        traf = bs.traf
        coords = (self.ac_env_id, self.ac_slot)
        shape = (self.cfg.num_envs, self.cfg.aircraft_per_env)
        columns = {
            "latitude": traf.lat, "longitude": traf.lon, "altitude": traf.alt, "heading": traf.hdg,
            "east": traf.gseast, "north": traf.gsnorth, "vertical": traf.vs, "goal_latitude": self.ac_goal_lat,
            "goal_longitude": self.ac_goal_lon, "planned_altitude": self.ac_target_alt_ft * ft,
            "lnav": traf.swlnav, "active": np.ones(traf.ntraf, dtype=np.bool_),
            "cas": traf.cas, "tas": traf.tas, "selected_altitude": traf.selalt,
        }
        if self.cross_track_observation:
            columns.update(origin_latitude=self.ac_orig_lat, origin_longitude=self.ac_orig_lon)
        tensors = {}
        for name, values in columns.items():
            padded = np.zeros(shape, dtype=values.dtype)
            padded[coords] = values
            tensors[name] = torch.from_numpy(padded)
        origin = (tensors.pop("origin_latitude"), tensors.pop("origin_longitude")) if self.cross_track_observation else None
        graph, self.physical = traffic_graph(
            **tensors, center_latitude=self.cfg.center_lat, center_longitude=self.cfg.center_lon,
            airspace_half_km=self.cfg.airspace_half_km, scale=self.observation_scale,
            geodesic_navigation=self.geodesic_navigation, route_origin=origin,
            far_conflict_observation=self.far_conflict_observation,
            selected_altitude_observation=self.selected_altitude_observation,
            include_potential=self.include_potential,
        )
        return graph

    def step_native_audited(self, actions, *, fast=False, track_separation=True, track_contacts=False):
        """Check independent traffic at every native physics frame."""
        cfg, traf = self.cfg, bs.traf
        shape = (cfg.num_envs, cfg.aircraft_per_env)
        reward = -0.1 * np.linalg.norm(actions, axis=-1).astype(np.float32)
        if traf.ntraf:
            commands = actions[self.ac_env_id, self.ac_slot]
            set_commands_direct(np.arange(traf.ntraf),
                                d_cas_kts=commands[:, 0] * cfg.max_speed_delta_kts,
                                d_alt_ft=commands[:, 1] * cfg.max_alt_delta_ft,
                                d_hdg_deg=commands[:, 2] * cfg.max_hdg_delta_deg,
                                hdg_deadband_deg=cfg.hdg_deadband_deg)
        events = {name: np.zeros(cfg.num_envs, dtype=np.int32) for name in ("collisions", "arrivals", "exits")}
        collision_aircraft = np.zeros(shape, dtype=np.bool_)
        if track_separation:
            min_horizontal = np.full(shape, np.inf)
            min_3d = np.full(shape, np.inf)
        if track_contacts:
            initial_altitude = np.zeros(shape)
            initial_altitude[self.ac_env_id, self.ac_slot] = traf.alt
            contacts = []
        step = bs.sim.step_fast if fast else bs.sim.step
        for frame in range(self.substeps):
            step()
            if self.pending_departures:
                self._release_due()
            if not traf.ntraf:
                continue
            lat, lon, alt = traf.lat, traf.lon, traf.alt
            distance = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
            vertical = np.abs(alt[:, None] - alt[None, :])
            # Recover exact 300-ft layer equality after subtracting two altitudes.
            rounding = 4.0 * np.finfo(alt.dtype).eps * np.maximum(np.abs(alt[:, None]), np.abs(alt[None, :]))
            inside_vertical = vertical < cfg.collision_v_ft * ft - rounding
            same = self.ac_env_id[:, None] == self.ac_env_id[None, :]
            np.fill_diagonal(same, False)
            if track_separation:
                slots = (self.ac_env_id, self.ac_slot)
                closest_h = np.where(same, distance, np.inf).min(axis=-1)
                closest_3d = np.where(same, np.hypot(distance, vertical / 1000.0), np.inf).min(axis=-1)
                min_horizontal[slots] = np.minimum(min_horizontal[slots], closest_h)
                min_3d[slots] = np.minimum(min_3d[slots], closest_3d)
            close_pair = same & (distance < cfg.collision_h_km) & inside_vertical
            collide = close_pair.any(axis=1)
            collision_aircraft[self.ac_env_id[collide], self.ac_slot[collide]] = True
            if track_contacts:
                for left, right in np.argwhere(np.triu(close_pair, 1)):
                    env = int(self.ac_env_id[left])
                    i, j = int(self.ac_slot[left]), int(self.ac_slot[right])
                    contacts.append({
                        "env": env, "pair": [i, j], "frame": frame + 1,
                        "initial_vertical_m": abs(float(initial_altitude[env, i] - initial_altitude[env, j])),
                        "vertical_m": float(vertical[left, right]), "horizontal_km": float(distance[left, right]),
                        "vertical_rates_mps": [float(traf.vs[left]), float(traf.vs[right])],
                    })
            remaining = ~collide
            arrive = remaining & (_haversine_km(lat, lon, self.ac_goal_lat, self.ac_goal_lon) < cfg.arrival_dist_km)
            initial_bearing = _bearing_deg(self.ac_orig_lat, self.ac_orig_lon, self.ac_goal_lat, self.ac_goal_lon)
            actual_bearing = _bearing_deg(self.ac_orig_lat, self.ac_orig_lon, lat, lon)
            arc = 1000.0 * _haversine_km(self.ac_orig_lat, self.ac_orig_lon, lat, lon) / Rearth
            cross_track = np.abs(np.arcsin(np.clip(np.sin(arc) * np.sin(np.deg2rad(actual_bearing - initial_bearing)),
                                                   -1.0, 1.0))) * (Rearth / 1000.0)
            exit_route = remaining & ~arrive & (cross_track > cfg.max_xtk_km)
            for name, selected, penalty in (("collisions", collide, -50), ("arrivals", arrive, 200),
                                             ("exits", exit_route, -50)):
                np.add.at(events[name], self.ac_env_id[selected], 1)
                reward[self.ac_env_id[selected], self.ac_slot[selected]] = penalty
            lost = np.flatnonzero(collide | arrive | exit_route)
            self._delete_aircraft_indices(lost)

        self.env_step_counts += 1
        remaining = np.bincount(self.ac_env_id, minlength=shape[0])
        for env_id, (schedule, _) in self.pending_departures.items():
            remaining[env_id] += len(schedule)
        timeout = self.env_step_counts >= cfg.max_steps
        terminal = (remaining == 0) | timeout
        events["timeouts"] = np.where(timeout, remaining, 0)
        events["collision_aircraft"] = collision_aircraft
        graph = self.reset(env_ids=np.flatnonzero(terminal))
        if track_separation:
            events["min_horizontal_km"] = min_horizontal
            events["min_3d_km"] = min_3d
        if track_contacts:
            events["contacts"] = contacts
        return graph, reward, terminal, events
