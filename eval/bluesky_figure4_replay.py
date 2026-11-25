"""Replay a fixed Figure 4 flight manifest in the native 0.05 s simulator."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
from bluesky.tools.aero import ft, kts
import numpy as np
import torch

from envs.bluesky_plan_a import GeoNativeBlueSky
from eval.bluesky_figure4_flow import Flight, sha256


class Figure4Replay(GeoNativeBlueSky):
    """A one-episode native environment with policy-independent release frames."""

    def __init__(self, config, *, workdir, manifest, observation_profile=None):
        if config.num_envs != 1 or config.sim_dt != .05 or config.control_dt != 6:
            raise ValueError("Figure 4 replay requires one airspace, 0.05 s physics, and six-second decisions")
        if (manifest["sim_dt_seconds"], manifest["control_dt_seconds"]) != (config.sim_dt, config.control_dt):
            raise ValueError("Manifest and simulator clocks differ")
        self.flights = [Flight(**row) for row in manifest["flights"]]
        if not self.flights or len(self.flights) > config.aircraft_per_env:
            raise ValueError("The aircraft slot limit must cover the whole manifest")
        frames = [flight.release_frame for flight in self.flights]
        if min(frames) < 0 or frames != sorted(frames) or len({f.callsign for f in self.flights}) != len(frames):
            raise ValueError("Expected unique, time-ordered Figure 4 departures")
        if frames[-1] > round(config.max_steps * config.control_dt / config.sim_dt):
            raise ValueError("The native episode ends before the last scheduled departure")
        self.cursor = 0
        self.start_time = None
        self.departures = []
        profile = asdict(observation_profile) if observation_profile is not None else {}
        super().__init__(config, workdir=workdir, family="figure4", **profile)
        enable_rl_fast_mode(False, simdt=config.sim_dt)

    def _spawn_env_aircraft(self, env_id):
        if self.start_time is not None:
            self.pending_departures.clear()
            return
        self.start_time = bs.sim.simt
        self.pending_departures[env_id] = (np.arange(len(self.flights)), ())
        # Native BlueSky otherwise waits for initial traffic or a scenario command.
        bs.sim.op()
        self._release_due()

    def _release_due(self):
        frame = round((bs.sim.simt - self.start_time) / self.cfg.sim_dt)
        stop = self.cursor
        while stop < len(self.flights) and self.flights[stop].release_frame <= frame:
            stop += 1
        if stop == self.cursor:
            return
        batch = self.flights[self.cursor:stop]
        if any(flight.release_frame != frame for flight in batch):
            raise ValueError(f"Missed scheduled Figure 4 release before frame {frame}")

        attributes = ("latitude", "longitude", "goal_latitude", "goal_longitude", "altitude_ft", "cas_kts", "heading")
        values = [np.asarray([getattr(flight, attr) for flight in batch]) for attr in attributes]
        slots = np.arange(self.cursor, stop, dtype=np.int32)
        index = slice(bs.traf.ntraf, bs.traf.ntraf + len(batch))
        callsigns = [flight.callsign for flight in batch]
        self._create_aircraft(0, slots, *values[:6], np.zeros(len(batch)), initial_heading=values[6],
                              callsigns=callsigns)
        if list(bs.traf.id[index]) != callsigns:
            raise ValueError("The simulator changed the Figure 4 callsigns")
        self.departures.extend(dict(callsign=flight.callsign, slot=int(slot), frame=frame)
                               for flight, slot in zip(batch, slots, strict=True))
        measured = (bs.traf.lat[index], bs.traf.lon[index], self.ac_goal_lat[index], self.ac_goal_lon[index],
                    bs.traf.alt[index] / ft, bs.traf.cas[index] / kts, bs.traf.hdg[index])
        for expected, actual in zip(values, measured, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        self.cursor = stop
        if stop == len(self.flights):
            self.pending_departures.pop(0)
        else:
            self.pending_departures[0] = (np.arange(stop, len(self.flights)), ())


def run(args):
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    args.workdir.mkdir()
    if args.navdata:
        cache = args.workdir / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(args.navdata)
    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=len(manifest["flights"]), sim_dt=.05, control_dt=6,
                                 max_steps=2000, seed=manifest["seed"])
    env = Figure4Replay(config, workdir=args.workdir, manifest=manifest)
    env.reset()
    for _ in range(args.steps):
        _, _, done, _ = env.step_native_audited(np.zeros((1, config.aircraft_per_env, 3), dtype=np.float32),
                                               track_separation=False)
        if done[0]:
            raise ValueError("The replay ended during the input probe")
    cutoff = round(args.steps * config.control_dt / config.sim_dt)
    expected = [f["callsign"] for f in manifest["flights"] if f["release_frame"] <= cutoff]
    observed = [row["callsign"] for row in env.departures]
    actual_frame = round((bs.sim.simt - env.start_time) / config.sim_dt)
    if actual_frame != cutoff or expected != observed:
        raise ValueError(f"At frame {actual_frame}, expected {expected} by frame {cutoff}, observed {observed}")
    result = dict(case=manifest["case"], seed=manifest["seed"], sim_dt=config.sim_dt, control_dt=config.control_dt,
                  manifest_sha256=sha256(args.manifest), verified_through_frame=cutoff,
                  departures=env.departures)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["case"], result["seed"], len(env.departures), cutoff, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "workdir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("Replay must cover at least one control step")
    run(args)
