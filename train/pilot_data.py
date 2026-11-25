"""Generate certified single-segment expert data from the reference NLP solver."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

from envs.observations import HORIZON, LAYER, RADIUS, SPEED_MAX, SPEED_MIN


CLIMB_TIME = 15.0


def scene(seed, size, ood, mixed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, size + 1))
    origin = rng.uniform(-120, 120, size=2)
    rotation = rng.uniform(-np.pi, np.pi)
    isolated_opposing = mixed and not ood and seed % 5 == 1
    if ood:
        # Paired opposing corridors held out of training.
        angles = rotation + rng.uniform(-0.13, 0.13, count) + np.pi * (np.arange(count) % 2)
        crossing_points = origin + rng.normal(0, 35, size=(count, 2))
    elif isolated_opposing:
        angles = rotation + np.pi * (np.arange(count) % 2)
        lane = np.arange(count) // 2 * rng.uniform(145, 210)
        lateral = np.array([-np.sin(rotation), np.cos(rotation)])
        crossing_points = origin + lane[:, None] * lateral + rng.normal(0, 2, size=(count, 2))
    elif mixed and seed % 3 == 0:
        routes = int(rng.integers(2, min(5, count) + 1))
        angles = rotation + (np.arange(count) % routes) * 2 * np.pi / routes + rng.uniform(-0.35, 0.35, count)
        crossing_points = origin + rng.normal(0, 30, size=(count, 2))
    else:
        angles = rotation + rng.uniform(-1.25, 1.25, count)
        crossing_points = origin + rng.normal(0, 22, size=(count, 2))
    direction = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count)))
    speed = rng.uniform(10, 14, size=count)
    position = np.column_stack((crossing_points, rng.choice([6000, 6000 + LAYER], p=[0.8, 0.2], size=count)))
    if isolated_opposing:
        position[:, 2] = 6000
    position -= direction * (speed * rng.uniform(17, 35, size=count))[:, None]
    return position, direction, speed


def _audit(position, direction, speed):
    left, right = np.triu_indices(len(speed), k=1)
    relative = position[left] - position[right]
    velocity = direction * speed[:, None]
    change = velocity[left] - velocity[right]
    time = np.clip(-np.sum(relative * change, axis=-1) / np.maximum(np.sum(change * change, axis=-1), 1e-12),
                   0, HORIZON)
    minimum = np.linalg.norm(relative + change * time[:, None], axis=-1)
    return bool(np.all(minimum >= RADIUS - 1e-5))


def _altitude_expert(position, direction, speed):
    count = len(speed)
    left, right = np.triu_indices(count, k=1)
    r = position[left] - position[right]
    horizontal = direction[:, :2] * speed[:, None]
    v = horizontal[left] - horizontal[right]
    a, b = np.sum(v * v, axis=-1), np.sum(r[:, :2] * v, axis=-1)
    c = np.sum(r[:, :2] ** 2, axis=-1) - RADIUS**2
    discriminant = b * b - a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    moving = a > 1e-12
    lo = np.where(moving, (-b - root) / np.maximum(a, 1e-12), 0).clip(min=0)
    hi = np.where(moving, (-b + root) / np.maximum(a, 1e-12), HORIZON).clip(max=HORIZON)
    encounters = np.where(moving, discriminant > 0, c < 0) & (hi > lo)
    i, j = left[encounters], right[encounters]
    gap = r[encounters, 2]
    sign = np.where(gap > 0, 1.0, -1.0)
    ramp = np.concatenate((lo[encounters], hi[encounters])).clip(0, CLIMB_TIME) / CLIMB_TIME
    rows = np.arange(len(ramp))
    pair = np.zeros((len(ramp), count))
    pair[rows, np.tile(i, 2)] = -ramp * np.tile(sign, 2)
    pair[rows, np.tile(j, 2)] = ramp * np.tile(sign, 2)
    bound = np.tile(np.abs(gap) - LAYER, 2) - ramp
    identity = np.eye(count)
    constraints = np.block([[pair, np.zeros_like(pair)], [identity, -identity], [-identity, -identity]])
    result = linprog(np.concatenate((np.zeros(count), np.ones(count))), A_ub=constraints,
                     b_ub=np.concatenate((bound, np.zeros(2 * count))),
                     bounds=[(-LAYER, LAYER)] * count + [(0, LAYER)] * count, method="highs")
    if not result.success:
        return np.zeros(count), False
    altitude = result.x[:count]
    at_lo = gap + (altitude[i] - altitude[j]) * ramp[:len(i)]
    at_hi = gap + (altitude[i] - altitude[j]) * ramp[len(i):]
    minimum = np.where(at_lo * at_hi <= 0, 0.0, np.minimum(np.abs(at_lo), np.abs(at_hi)))
    return altitude, bool(np.all(minimum >= LAYER - 1e-5))


def _sample(task):
    from speed_adjustment.solver import resolve_speeds_single_layer

    seed, size, ood, with_altitude = task
    position, direction, speed = scene(seed, size, ood, with_altitude)
    if with_altitude:
        position[:, 2] -= 6000
    nominal_feasible = _audit(position, direction, speed)
    # All aircraft remain on this segment for the whole certification horizon.
    route_length = SPEED_MAX * HORIZON
    waypoints = np.stack((position, position + route_length * direction), axis=1)
    times = np.column_stack((np.zeros(len(speed)), route_length / speed))
    solution = resolve_speeds_single_layer(waypoints, times, speed[:, None], 0,
                                           rs=RADIUS, v_min=SPEED_MIN, v_max=SPEED_MAX, n_restarts=1, seed=seed)
    expert = solution["v_opt"]
    speed_certified = _audit(position, direction, expert)
    altitude = np.zeros(size, dtype=np.float32)
    altitude_certified = False
    if with_altitude and not speed_certified:
        expert = speed
        altitude[:len(speed)], altitude_certified = _altitude_expert(position, direction, speed)
    padded = np.zeros((size, 3, 3), dtype=np.float32)
    count = len(speed)
    padded[:count, 0] = position
    padded[:count, 1] = direction
    padded[:count, 2, 0] = speed
    padded[:count, 2, 1] = expert
    padded[:count, 2, 2] = 1.0
    return padded, altitude, (nominal_feasible, speed_certified, altitude_certified)


def generate(output, reference, counts, size, workers, with_altitude):
    sys.path.insert(0, str(reference))
    splits = {"train": (117_001, counts[0]), "validation": (221_001, counts[1]), "ood": (312_001, counts[2])}
    arrays = {}
    report = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, (start_seed, count) in splits.items():
            tasks = ((start_seed + index, size, split == "ood", with_altitude) for index in range(count))
            samples, altitude, flags = zip(*pool.map(_sample, tasks, chunksize=4))
            nominal, speed_certified, altitude_certified = np.asarray(flags).T
            certified = speed_certified | altitude_certified
            arrays[f"{split}_scenes"] = np.stack(samples)
            arrays[f"{split}_certified"] = certified
            arrays[f"{split}_nominal"] = nominal
            if with_altitude:
                arrays[f"{split}_altitude"] = np.stack(altitude)
                arrays[f"{split}_altitude_certified"] = altitude_certified
            report[split] = {"first_seed": start_seed, "scenes": count, "expert_certified": int(certified.sum()),
                             "speed_certified": int(speed_certified.sum()), "altitude_certified": int(altitude_certified.sum()),
                             "nominal_conflicts": int((~nominal).sum()),
                             "expert_resolved": int((certified & ~nominal).sum())}
            print(split, report[split], flush=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=int, default=768)
    parser.add_argument("--validation", type=int, default=160)
    parser.add_argument("--ood", type=int, default=160)
    parser.add_argument("--aircraft", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--with-altitude", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.reference, (args.train, args.validation, args.ood), args.aircraft, args.workers,
             args.with_altitude)
