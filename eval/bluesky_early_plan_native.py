"""Replay an early speed/altitude plan in native BlueSky with the exact Torch layout."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
from bluesky.rl.batched_env import _bearing_deg
from bluesky.tools.aero import ft, kts
import numpy as np
import torch

from envs.bluesky_plan_a import GeoNativeBlueSky
from envs.bluesky_safety import single_axis_commands
from envs.bluesky_traffic import Routes, sample_routes
from eval.bluesky_pair import _checksum


class FixedRoutesBlueSky(GeoNativeBlueSky):
    def __init__(self, config, *, workdir, family, routes):
        self.routes = routes
        super().__init__(config, workdir=workdir, family=family)

    def _sample_geo_routes(self):
        return self.routes


def run(args):
    torch.set_num_threads(1)
    source = json.loads(args.plan.read_text())
    family, = source['families']
    report = source['families'][family]
    selected = np.array(report['selected_profiles'][args.world])
    cfg = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=source['aircraft'], sim_dt=.05, control_dt=6,
                              hdg_deadband_deg=.25, max_steps=400)
    route = sample_routes(family, len(report['search']), cfg.aircraft_per_env, cfg.center_lat, cfg.center_lon,
                          cfg.airspace_half_km, torch.Generator().manual_seed(report['seed']), 'cpu', dtype=torch.float64)
    fixed = Routes(**{field.name: getattr(route, field.name)[args.world:args.world + 1] for field in fields(Routes)})
    args.output.mkdir(parents=True)
    env = FixedRoutesBlueSky(cfg, workdir=args.output, family=family, routes=fixed)
    env.reset()
    enable_rl_fast_mode(False, simdt=cfg.sim_dt)
    limits = source.get('limits', {'cas_kts': (200, 300), 'altitude_ft': (20000, 21000)})
    target_cas = np.clip(fixed.cas_kts[0].numpy() + selected[:, 0], *limits['cas_kts'])
    target_alt = np.clip(fixed.altitude_ft[0].numpy() + selected[:, 1] * cfg.max_alt_delta_ft, *limits['altitude_ft'])
    prepare = report.get('start_controls')
    start_controls = np.zeros(cfg.aircraft_per_env) if prepare is None else np.asarray(prepare[args.world])
    fields_to_count = ('arrivals', 'collisions', 'exits', 'timeouts')
    outcomes = np.zeros(len(fields_to_count), dtype=np.int64)
    for decision in range(cfg.max_steps):
        slots = env.ac_slot
        control = np.zeros((1, cfg.aircraft_per_env, 3))
        heading = _bearing_deg(bs.traf.lat, bs.traf.lon, env.ac_goal_lat, env.ac_goal_lon)
        heading_error = (heading - bs.traf.hdg + 180) % 360 - 180
        control[0, slots] = np.clip(np.stack((
            (target_cas[slots] - bs.traf.cas / kts) / cfg.max_speed_delta_kts,
            (target_alt[slots] - bs.traf.alt / ft) / cfg.max_alt_delta_ft,
            heading_error * ~bs.traf.swlnav / cfg.max_hdg_delta_deg,
        ), axis=-1), -1, 1)
        control[0, slots, :2] *= (decision >= start_controls[slots]).reshape(-1, 1)
        if source.get('single_axis', False):
            control = single_axis_commands(torch.from_numpy(control)).numpy()
        _, _, terminal, info = env.step_native_audited(control, track_separation=False)
        outcomes += np.array([info[key][0] for key in fields_to_count])
        if terminal[0]:
            break
    result = {'family': family, 'torch_layout_seed': report['seed'], 'world': args.world,
              'event_sampling_seconds': cfg.sim_dt, 'aircraft': cfg.aircraft_per_env, 'heading_recovery': 'geodesic',
              'plan_sha256': _checksum(args.plan),
              'outcomes': dict(zip(fields_to_count, outcomes.tolist(), strict=True))}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
