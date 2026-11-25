"""One native BlueSky instance per subprocess, with packed airspaces in each."""

from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import traceback

import numpy as np
import torch

from envs.bluesky_graph import SCALE

SEPARATION_FIELDS = ("min_horizontal_km", "min_3d_km")


@dataclass(frozen=True)
class WorkerConfig:
    envs: int
    aircraft: int
    dt: float
    control: float
    max_steps: int
    seed: int
    native: bool
    family: str = "upstream"
    audit_frames: bool = False
    timed_departures: bool = False
    detection_metres: float = SCALE.detection
    geodesic_navigation: bool = False
    correlated_yaw: bool = False
    training_yaw_degrees: float | None = None
    cross_track_observation: bool = False
    far_conflict_observation: bool = False
    selected_altitude_observation: bool = False
    navdata: Path | None = None
    include_potential: bool = True


def _worker(connection, cfg, workdir):
    try:
        import bluesky as bs
        from bluesky.rl import BatchedBlueSkyConfig
        from envs.bluesky_plan_a import GeoNativeBlueSky

        torch.set_num_threads(1)
        workdir.mkdir()
        if cfg.navdata is not None:
            cache = workdir / "cache"
            cache.mkdir()
            (cache / "navdata.p").hardlink_to(cfg.navdata)
        env = GeoNativeBlueSky(BatchedBlueSkyConfig(
            num_envs=cfg.envs, aircraft_per_env=cfg.aircraft, sim_dt=cfg.dt, control_dt=cfg.control,
            max_steps=cfg.max_steps, hdg_deadband_deg=0.25, seed=cfg.seed,
        ), workdir=workdir, family=cfg.family, timed_departures=cfg.timed_departures,
           detection_metres=cfg.detection_metres, geodesic_navigation=cfg.geodesic_navigation,
           correlated_yaw=cfg.correlated_yaw, training_yaw_degrees=cfg.training_yaw_degrees,
           cross_track_observation=cfg.cross_track_observation,
           far_conflict_observation=cfg.far_conflict_observation,
           selected_altitude_observation=cfg.selected_altitude_observation,
           include_potential=cfg.include_potential)
        if cfg.native:
            from bluesky.rl import enable_rl_fast_mode
            enable_rl_fast_mode(False, simdt=cfg.dt)

            def native_step(*, n_steps, dt):
                for _ in range(n_steps):
                    bs.sim.step()

            bs.sim.step_fast = native_step
        graph = env.reset()
        connection.send(("ok", (tuple(value.numpy() for value in graph), tuple(v.numpy() for v in env.physical),
                                (env.traffic_draws, np.array([env.launched_aircraft])))))
        step = partial(env.step_native_audited, fast=not cfg.native, track_separation=cfg.native) if cfg.audit_frames else env.step
        while (action := connection.recv()) is not None:
            category = env.scene_category.copy()
            route_count = env.scene_route_count.copy()
            graph, reward, done, extras = step(action)
            basic = tuple(value.numpy() for value in graph)
            physical = tuple(value.numpy() for value in env.physical)
            exits = extras.get("exits", np.zeros_like(extras["arrivals"]))
            timeouts = extras.get("timeouts", np.zeros_like(extras["arrivals"]))
            separation = tuple(extras[name] for name in SEPARATION_FIELDS if name in extras)
            collision_labels = (extras["collision_aircraft"],) if cfg.audit_frames else ()
            connection.send(("ok", (basic, physical, reward, done, extras["arrivals"], extras["collisions"], exits,
                                    timeouts, env.traffic_draws, np.array([env.launched_aircraft]), category, route_count,
                                    *collision_labels, *separation)))
    except (EOFError, KeyboardInterrupt):
        return
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class NativeWorkers:
    def __init__(self, workers, config, workdir, device, *, seed_stride=10_007):
        self.device = device
        self.audit_frames = config.audit_frames
        self.paths = []
        ctx = get_context("spawn")
        for index in range(workers):
            parent, child = ctx.Pipe()
            worker = replace(config, seed=config.seed + seed_stride * index)
            process = ctx.Process(target=_worker, args=(child, worker, workdir / f"worker-{index:02d}"))
            process.start()
            child.close()
            self.paths.append((parent, process))

    def _receive(self):
        messages = []
        for endpoint, process in self.paths:
            if not endpoint.poll(120):
                raise TimeoutError(f"BlueSky worker {process.pid} exceeded 120 s")
            status, payload = endpoint.recv()
            if status != "ok":
                raise RuntimeError(payload)
            messages.append(payload)
        return messages

    def _join(self, columns):
        return tuple(torch.from_numpy(np.concatenate(parts)).to(self.device) for parts in zip(*columns, strict=True))

    def reset(self):
        observations, states, draws = zip(*self._receive(), strict=True)
        self.physical = self._join(states)
        categories, launched = self._join(draws)
        self.traffic_draws = categories.sum(dim=0)
        self.launched_aircraft = launched.sum()
        return self._join(observations)

    def start_step(self, action):
        action = action.detach().cpu().numpy()
        for (endpoint, _), chunk in zip(self.paths, np.array_split(action, len(self.paths)), strict=True):
            endpoint.send(chunk)

    def finish_step(self):
        messages = self._receive()
        graphs = tuple(message[0] for message in messages)
        self.physical = self._join(tuple(message[1] for message in messages))
        rewards, terminated, arrived, collided, exits, timeouts, draws, launched, category, route_count, *audit = self._join(
            tuple(m[2:] for m in messages))
        self.traffic_draws = draws.sum(dim=0)
        self.launched_aircraft = launched.sum()
        events = {"arrivals": arrived, "collisions": collided, "exits": exits, "timeouts": timeouts,
                  "category": category, "route_count": route_count}
        if self.audit_frames:
            collision_aircraft, *separation = audit
            events["collision_aircraft"] = collision_aircraft
            if separation:
                events.update(zip(SEPARATION_FIELDS, separation, strict=True))
        return self._join(graphs), rewards, terminated, events

    def step(self, action):
        self.start_step(action)
        return self.finish_step()

    def close(self):
        for endpoint, process in self.paths:
            if process.is_alive():
                try:
                    endpoint.send(None)
                except (OSError, EOFError):
                    pass
            endpoint.close()
        for _, process in self.paths:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        return all(not process.is_alive() for _, process in self.paths)
