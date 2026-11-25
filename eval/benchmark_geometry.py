"""Time and audit two-stage ADMM on 30 simultaneously active crossing aircraft."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

import torch

from core.admm_layer import altitude_admm, speed_admm
from core.conflict_graph import _ball_interval, connected_components, cylinder_interval, greedy_colors


def dense_crossing(batch, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    gaps = 64 + 8 * torch.rand(batch, 2, 15, device=device, generator=rng)
    lanes = gaps.cumsum(dim=-1) - gaps[..., :1]
    x, y = lanes.unbind(dim=-2)
    left = -240 - 25 * torch.rand(batch, 15, device=device, generator=rng)
    bottom = -270 - 25 * torch.rand(batch, 15, device=device, generator=rng)
    horizontal = torch.stack((torch.cat((left, x), dim=-1), torch.cat((y, bottom), dim=-1)), dim=-1)
    angles = 2 * torch.pi * torch.rand(batch, 1, device=device, generator=rng)
    rot_x = torch.cat((angles.cos(), -angles.sin()), dim=-1)
    rot_y = torch.cat((angles.sin(), angles.cos()), dim=-1)
    rotation = torch.stack((rot_x, rot_y), dim=-2)
    direction = torch.cat((rot_x[:, None, :].expand(-1, 15, -1), rot_y[:, None, :].expand(-1, 15, -1)), dim=1)
    direction = torch.cat((direction, torch.zeros_like(direction[..., :1])), dim=-1)
    position = torch.cat((horizontal @ rotation, torch.full_like(horizontal[..., :1], 6000)), dim=-1)
    speed = 11.5 + torch.rand(batch, 30, device=device, generator=rng)
    active = torch.ones_like(speed, dtype=torch.bool)
    return position, direction, speed, active


def predict(position, direction, speed, active):
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    velocity = direction * speed.unsqueeze(-1)
    delta = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    _, _, encounter = cylinder_interval(relative, delta, 50, 91.44, 120)
    edges = encounter & active.unsqueeze(-1) & active.unsqueeze(-2)
    edges = edges & ~torch.eye(active.shape[-1], dtype=torch.bool, device=active.device)
    return edges


def benchmark(device, batch, repeat, seed):
    torch.set_num_threads(1)
    position, direction, speed, active = dense_crossing(batch, device, seed)
    adjacency = predict(position, direction, speed, active)
    colors = greedy_colors(adjacency)
    same_color = colors.unsqueeze(-1) == colors.unsqueeze(-2)
    assert not (same_color & adjacency).any()
    components = connected_components(adjacency)
    component_sizes = (components.unsqueeze(-1) == components.unsqueeze(-2)).sum(dim=-1)

    def run():
        first = speed_admm(position, direction, speed, active, adjacency, minimum_speed=5,
                           maximum_speed=20, radius=50, horizon=120, iterations=5)
        second = altitude_admm(position, (first.speed.unsqueeze(-1) * direction)[..., :2],
                               torch.zeros_like(speed), active, minimum_altitude=5900,
                               maximum_altitude=6100, horizontal_radius=50, iterations=5)
        offsets = torch.where(first.feasible.unsqueeze(-1), 0.0, second.offset)
        return first, second, offsets

    for _ in range(4):
        run()
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = perf_counter()
        result = run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((perf_counter() - begin) * 1000)

    first, second, offsets = result
    relative = position.unsqueeze(-2) - position.unsqueeze(-3)
    horizontal = first.speed.unsqueeze(-1) * direction[..., :2]
    delta = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    lo, hi, horizontal_hit = _ball_interval(relative[..., :2], delta, 50, 120)
    vertical_delta = offsets.unsqueeze(-1) - offsets.unsqueeze(-2)
    z_lo = relative[..., 2] + vertical_delta * (lo / 15).clamp(0, 1)
    z_hi = relative[..., 2] + vertical_delta * (hi / 15).clamp(0, 1)
    minimum = torch.where(z_lo * z_hi <= 0, 0.0, torch.minimum(z_lo.abs(), z_hi.abs()))
    upper = torch.ones_like(adjacency).triu(1)
    collisions = (horizontal_hit & upper & (minimum < 91.44)).sum(dim=(-2, -1))
    return {
        "device": str(device), "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed, "scenes": batch, "aircraft_per_scene": 30,
        "initial_collision_edges": (adjacency & upper).sum(dim=(-2, -1)).tolist(),
        "remaining_collision_edges": collisions.tolist(),
        "max_component_sizes": component_sizes.amax(dim=-1).tolist(),
        "speed_only_feasible": int(first.feasible.sum()), "altitude_feasible": int(second.feasible.sum()),
        "collision_free_scenes": int((collisions == 0).sum()),
        "median_batch_ms": median(times), "p95_batch_ms": quantiles(times, n=20)[18],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "ppu"), default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.repeat < 2:
        parser.error("Positive batch and at least two repetitions required")
    if args.device == "ppu":
        if importlib.util.find_spec("torch_ppu") is not None:
            __import__("torch_ppu")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("The benchmark needs exactly one allocated accelerator")
        if "PPU" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("The selected accelerator is not a PPU")
    device = torch.device("cuda:0" if args.device == "ppu" else "cpu")
    metrics = benchmark(device, args.batch, args.repeat, args.seed)
    rendered = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
