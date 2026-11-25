"""Run an upstream module with an isolated BlueSky configuration and cache."""

from functools import partial
import importlib.util
from pathlib import Path
import runpy
import sys

import bluesky


def main():
    workdir, target, *arguments = sys.argv[1:]
    Path(workdir).mkdir(exist_ok=True)
    bluesky.init = partial(bluesky.init, workdir=workdir)
    if target == "bluesky.rl_gpu.benchmark_plan_b":
        import torch
        if importlib.util.find_spec("torch_ppu") is not None:
            __import__("torch_ppu")
        if torch.cuda.device_count() != 1 or "PPU" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("Plan B throughput requires exactly one PPU")
    sys.argv = [target, *arguments]
    runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
