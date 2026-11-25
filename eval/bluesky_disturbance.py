"""Evaluate reproducible disturbances in the native 0.05 s simulator."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
from bluesky.tools.aero import ft, kts
import numpy as np
import torch

from envs.bluesky_plan_a import GeoNativeBlueSky
from envs.bluesky_safety import protected_command
from envs.bluesky_traffic import sample_routes
from eval.admm_gap_mpc import plan_gap_maneuver
from eval.bluesky_pair import _checksum, _communication_interval, _load
from eval.graph_statistics import GraphStatistics
from eval.stage_acceptance import MPC_COMMUNICATION
from train.td3 import encode_action


DISTURBANCES = ("popup", "noncooperative", "speed-noise", "wind-field")
FAMILIES = ("crossing3", "crossing4", "crossing6", "head_on", "yaw_plus15", "yaw_minus15")
OUTCOMES = ("arrivals", "collisions", "exits", "timeouts")


def _rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


class DisturbedBlueSky(GeoNativeBlueSky):
    def __init__(self, config, *, disturbances, **kwargs):
        self.disturbances = frozenset(disturbances)
        self.base_aircraft = config.aircraft_per_env - 1
        popup_rng = _rng(config.seed, 1)
        self.popup_slot = self.base_aircraft
        self.popup_lane_slot = int(popup_rng.integers(0, self.base_aircraft))
        self.popup_release = int(popup_rng.integers(10, 31)) * 6
        self.popup_fraction = float(popup_rng.uniform(.25, .45))
        self.popup_actual_seconds = None
        if "popup" in disturbances and self.popup_release > config.max_steps * config.control_dt:
            raise ValueError("Evaluation horizon ends before the scheduled insertion")
        slots = _rng(config.seed, 2).choice(self.base_aircraft, max(1, round(self.base_aircraft / 10)), replace=False)
        self.noncooperative = np.sort(slots).tolist() if "noncooperative" in disturbances else []
        self.noncooperative_targets = {}
        self.noise = _rng(config.seed, 3).uniform(-.05, .05, (config.max_steps, config.aircraft_per_env))
        self.base_routes_sha256 = None
        super().__init__(config, **kwargs)
        enable_rl_fast_mode(False, simdt=config.sim_dt)

    def _spawn_env_aircraft(self, env_id):
        if self.base_routes_sha256 is not None:
            self.pending_departures.clear()
            return
        super()._spawn_env_aircraft(env_id)

    def _sample_geo_routes(self):
        cfg = self.cfg
        routes = sample_routes(self.family, 1, self.base_aircraft, cfg.center_lat, cfg.center_lon,
                               cfg.airspace_half_km, self.layout_generator, "cpu")
        attributes = ("latitude", "longitude", "goal_latitude", "goal_longitude", "altitude_ft", "cas_kts", "yaw_deg")
        self.base_routes_sha256 = sha256(b"".join(getattr(routes, k).numpy().tobytes() for k in attributes)).hexdigest()
        index = self.popup_lane_slot
        data = {k: getattr(routes, k)[:, index:index+1].clone() for k in attributes}
        for origin, goal in (("latitude", "goal_latitude"), ("longitude", "goal_longitude")):
            data[origin].lerp_(data[goal], self.popup_fraction)
        data["yaw_deg"].zero_()
        data["active"] = torch.tensor([["popup" in self.disturbances]])
        data["departure_seconds"] = torch.tensor([[self.popup_release]], dtype=routes.departure_seconds.dtype)
        return replace(routes, **{k: torch.cat((getattr(routes, k), v), dim=-1) for k, v in data.items()})

    def _create_aircraft(self, env_id, slots, *args, **kwargs):
        first = bs.traf.ntraf
        super()._create_aircraft(env_id, slots, *args, **kwargs)
        for index, slot in enumerate(slots, first):
            if slot == self.popup_slot:
                self.popup_actual_seconds = float(bs.sim.simt)
            if slot in self.noncooperative:
                bs.traf.ap.selhdgcmd(index, bs.traf.hdg[index])
                bs.traf.swlnav[index] = False
                bs.traf.swvnav[index] = False
                bs.traf.swvnavspd[index] = False
                self.noncooperative_targets[int(slot)] = self._targets(index)

    @staticmethod
    def _targets(index):
        return (float(bs.traf.ap.trk[index]), float(bs.traf.selspd[index]), float(bs.traf.selalt[index]))

    def disturb(self, commands, decision):
        result = commands.clone()
        if "speed-noise" in self.disturbances:
            result[..., 0] = (result[..., 0] * (1 + result.new_tensor(self.noise[decision]))).clamp(-1, 1)
        if self.noncooperative:
            result[:, self.noncooperative] = 0
        return result

    def verify_refusal(self):
        for index, slot in enumerate(self.ac_slot):
            if slot in self.noncooperative:
                if bs.traf.swlnav[index]:
                    raise ValueError("A noncooperative aircraft resumed LNAV")
                np.testing.assert_allclose(self._targets(index), self.noncooperative_targets[int(slot)],
                                           atol=1e-8, rtol=0)


def _mean_finite(values):
    selected = values[np.isfinite(values)]
    return float(selected.mean()) if selected.size else None


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    mpc = getattr(args, "mpc_gap", False)
    if mpc and args.without_projection:
        raise ValueError("MPC gap control requires projection")
    layer_options = dict(early_layer_search=getattr(args, "early_layer_search", False),
                         exact_layer_forecast=getattr(args, "exact_layer_forecast", False),
                         batched_layer_margin=getattr(args, "batched_layer_margin", 0.))
    args.workdir.mkdir()
    if args.navdata:
        cache = args.workdir / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(args.navdata)
    actor = _load(args.checkpoint, torch.device("cpu"))
    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=args.aircraft+1, sim_dt=.05, control_dt=6,
                                 hdg_deadband_deg=.25, max_steps=args.max_steps, seed=args.seed)
    interval = _communication_interval(MPC_COMMUNICATION["admm_communication_period"] if mpc else None,
                                       config.control_dt)
    env = DisturbedBlueSky(config, disturbances=args.disturbances, workdir=args.workdir, family=args.family,
                           **asdict(actor.observation_profile))
    wind_profiles = []
    if "wind-field" in args.disturbances:
        for offset, headings, speeds in ((-.35, [270., 240.], [18., 42.]),
                                         (.35, [220., 190.], [22., 36.])):
            longitude = config.center_lon + offset
            levels = np.array([19000., 22000.])
            bs.traf.wind.addpoint(config.center_lat, longitude, np.asarray(headings), np.asarray(speeds) * kts,
                                  levels * ft)
            wind_profiles.append(dict(latitude=config.center_lat, longitude=longitude, altitude_ft=levels.tolist(),
                                      wind_from_deg=headings, wind_kts=speeds))
    graph = env.reset()
    totals = Counter(dict.fromkeys(OUTCOMES, 0))
    mins = {k: np.full(config.aircraft_per_env, np.inf) for k in ("min_horizontal_km", "min_3d_km")}
    statistics = GraphStatistics()
    contacts, elapsed_ms = [], []
    intervention_count = disturbed_count = 0
    gap_ticks = gap_adjusted = 0
    control_cost = 0.
    wind_samples = 0
    wind_max_mps = wind_velocity_error_mps = 0.
    for decision in range(config.max_steps):
        active = graph[-1]
        started = perf_counter()
        policy = actor(*graph)
        action = encode_action(policy)
        planned = action[..., 4:]
        if not args.without_projection:
            if decision % interval:
                planned, _, _ = plan_gap_maneuver(env.physical, active, planned)
                gap_ticks += 1
                gap_adjusted += int(planned.abs().amax(dim=-1).gt(1e-5).sum())
            else:
                tuning = policy.consensus_inputs
                planned, _ = protected_command(action, env.physical, graph[2], active, tuning, 2, 0,
                                              directional_heading=True, extra_rounds=2, batched_layer_search=True,
                                              joint_layer_search=True, joint_layer_reserve=10., **layer_options)
        elapsed_ms.append((perf_counter()-started)*1000)
        planned = planned.masked_fill(~active.unsqueeze(-1), 0)
        commands = env.disturb(planned, decision)
        intervention_count += int(((planned-action[..., 4:]).abs().amax(-1).gt(1e-5) & active).sum())
        disturbed_count += int(((commands-planned).abs().amax(-1).gt(1e-5) & active).sum())
        control_cost += float(commands.abs().sum())
        statistics.add(graph[2][0].numpy(), active[0].numpy())
        graph, _, done, info = env.step_native_audited(commands.numpy(), track_contacts=True)
        env.verify_refusal()
        if bs.traf.ntraf:
            heading = np.deg2rad(bs.traf.hdg)
            north, east = bs.traf.windnorth, bs.traf.windeast
            wind_samples += bs.traf.ntraf
            wind_max_mps = max(wind_max_mps, float(np.hypot(north, east).max()))
            error = np.hypot(bs.traf.gsnorth - bs.traf.tas * np.cos(heading) - north,
                             bs.traf.gseast - bs.traf.tas * np.sin(heading) - east)
            wind_velocity_error_mps = max(wind_velocity_error_mps, float(error.max()))
        totals.update({k: int(info[k][0]) for k in OUTCOMES})
        contacts.extend(dict(decision=decision+1, **row) for row in info["contacts"])
        for k in mins:
            np.minimum(mins[k], info[k][0], out=mins[k])
        if done[0]:
            break
    expected = args.aircraft + (1 if "popup" in args.disturbances else 0)
    if sum(totals.values()) != expected or env.launched_aircraft != expected:
        raise ValueError("Disturbance evaluation did not launch and account for every aircraft")
    if "popup" in args.disturbances:
        np.testing.assert_allclose(env.popup_actual_seconds, env.popup_release, rtol=0, atol=1e-7)
    if wind_profiles and (wind_max_mps <= 0 or wind_velocity_error_mps > 1e-7):
        raise ValueError("Native wind was not applied to the measured ground velocity")
    output = dict(seed=args.seed, family=args.family, sim_dt=.05, control_dt=6, aircraft=expected,
                  checkpoint_sha256=_checksum(args.checkpoint), projection=not args.without_projection,
                  layer_safety_options=layer_options, disturbances=args.disturbances, base_routes_sha256=env.base_routes_sha256,
                  popup=dict(origin_route_slot=env.popup_lane_slot, route_fraction=env.popup_fraction,
                             scheduled_seconds=env.popup_release, actual_seconds=env.popup_actual_seconds),
                  noncooperative_slots=env.noncooperative, noncooperative_targets=env.noncooperative_targets,
                  speed_increment_noise=dict(draws_sha256=sha256(env.noise.tobytes()).hexdigest(), relative_max=.05),
                  wind=dict(profiles=wind_profiles, aircraft_samples=wind_samples, max_mps=wind_max_mps,
                            max_ground_velocity_error_mps=wind_velocity_error_mps),
                  outcomes=dict(totals), contacts=contacts, mean_minimum={k: _mean_finite(v) for k,v in mins.items()},
                  graph=statistics.report(), admm_interventions=intervention_count,
                  disturbed_commands=disturbed_count, normalized_cost_per_aircraft=control_cost/expected,
                  decision_ms=dict(mean=float(np.mean(elapsed_ms)), p95=float(np.percentile(elapsed_ms,95))))
    if mpc:
        output["communication"] = MPC_COMMUNICATION | dict(gap_ticks=gap_ticks, adjusted_aircraft=gap_adjusted)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(args.family, args.seed, args.disturbances, dict(totals), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "workdir", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--navdata", type=Path)
    p.add_argument("--early-layer-search", action="store_true")
    p.add_argument("--exact-layer-forecast", action="store_true")
    p.add_argument("--batched-layer-margin", type=float, default=0.)
    p.add_argument("--mpc-gap", action="store_true")
    p.add_argument("--family", choices=FAMILIES, default="crossing6")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--aircraft", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--disturbances", choices=DISTURBANCES, nargs="+", default=[])
    p.add_argument("--without-projection", action="store_true")
    options = p.parse_args()
    if options.aircraft < 2 or options.max_steps < 1:
        p.error("Use at least two aircraft and a positive horizon")
    if len(set(options.disturbances)) != len(options.disturbances):
        p.error("Disturbances must be distinct")
    run(options)
