"""Compare native, fast, and packed frames using the same commands and traffic."""

import argparse
from pathlib import Path

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, BatchedBlueSkyEnv, enable_rl_fast_mode, set_commands_direct
from bluesky.tools.aero import ft, kts
import numpy as np


def trace(mode, workdir, output):
    packed = 8 if mode == "packed" else 1
    bs.init(mode="sim", detached=True, workdir=str(workdir))
    cfg = BatchedBlueSkyConfig(num_envs=packed, aircraft_per_env=12, seed=52, sim_dt=0.05)
    env = BatchedBlueSkyEnv(cfg, init_bluesky=False)
    env.reset()
    native = mode == "native"
    enable_rl_fast_mode(not native, simdt=cfg.sim_dt)
    ids = np.flatnonzero(env.ac_env_id == 0)
    frames = []
    for frame in range(600):
        if frame in (0, 160, 360):
            direction = -1 if frame == 160 else 1
            speed, vertical, yaw = 3.0 * direction, 180.0 * direction, 5.0 * direction
            if native:
                ap = bs.traf.ap
                ap.selspdcmd(ids[0], bs.traf.cas[ids[0]] + speed * kts)
                ap.selaltcmd(ids[1], bs.traf.alt[ids[1]] + vertical * ft)
                ap.selhdgcmd(ids[2], (bs.traf.hdg[ids[2]] + yaw) % 360)
            else:
                speed_commands = np.zeros(len(ids))
                alt_commands = np.zeros(len(ids))
                heading_commands = np.zeros(len(ids))
                speed_commands[0], alt_commands[1], heading_commands[2] = speed, vertical, yaw
                heading_commands[3] = 0.2
                set_commands_direct(ids, speed_commands, alt_commands, heading_commands, hdg_deadband_deg=0.25)
        if native:
            bs.sim.step()
        else:
            bs.sim.step_fast()
        state = (bs.traf.lat, bs.traf.lon, bs.traf.alt, bs.traf.tas, bs.traf.hdg, bs.traf.swlnav,
                 bs.traf.perf.mass, bs.traf.perf.phase)
        frames.append(np.column_stack([column[ids] for column in state]))
    assert np.all(np.asarray(frames)[:, 3:, 5]), "Hold and near-zero headings must preserve LNAV"
    np.save(output, np.stack(frames))


def compare(output):
    fields = ("latitude_deg", "longitude_deg", "altitude_m", "tas_mps", "heading_deg", "lnav",
              "aircraft_mass_kg", "performance_phase")
    native = np.load(output / "native.npy")
    for candidate in ("fast", "packed"):
        error = np.load(output / f"{candidate}.npy") - native
        error[..., 4] = (error[..., 4] + 180.0) % 360.0 - 180.0
        maximum = dict(zip(fields, np.abs(error).max(axis=(0, 1)), strict=True))
        print(candidate, maximum)
        np.testing.assert_array_less(list(maximum.values()), [1e-5, 1e-5, 0.05, 0.05, 0.05, 0.1, 1e-3, 0.1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "fast", "packed", "compare"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        compare(args.output)
    else:
        workdir = args.output / args.mode
        workdir.mkdir(exist_ok=True)
        trace(args.mode, workdir, args.output / f"{args.mode}.npy")
