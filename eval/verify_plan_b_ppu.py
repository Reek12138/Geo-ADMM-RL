"""Run the upstream parity scenario with float64 physics resident on the PPU."""

import importlib.util
import math

import torch

from bluesky.rl_gpu.verify_parity import verify_parity


def main():
    if importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    if torch.cuda.device_count() != 1 or "PPU" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The extended parity test requires exactly one PPU")
    print(f"Physics on {torch.cuda.get_device_name(0)}, float64")
    with torch.device("cuda:0"):
        error = verify_parity(sim_dt=0.05, steps=600)
    allowed = {"lat_deg": 1e-5, "lon_deg": 1e-5, "alt_m": 0.05, "hdg_deg": 0.05, "tas_mps": 0.05}
    failure = {name: value for name, value in error.items() if not math.isfinite(value) or value > allowed[name]}
    if failure:
        raise RuntimeError(f"PPU parity over tolerance: {failure}")


if __name__ == "__main__":
    main()
