"""Train and evaluate the speed-only behavior cloning pilot."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from core.admm_layer import altitude_admm, speed_admm
from envs.observations import HORIZON, LAYER, RADIUS, SPEED_MAX, SPEED_MIN, SPEED_UNIT, observe
from models.graphformer import Actor


def _load(path, device):
    splits = {}
    with np.load(path) as data:
        for name in ("train", "validation", "ood"):
            records = torch.as_tensor(data[f"{name}_scenes"], device=device)
            position, direction, other = records.unbind(dim=-2)
            speed, expert, active = other.unbind(dim=-1)
            active = active.bool()
            altitude_supported = f"{name}_altitude" in data
            origin = 0.0 if altitude_supported else 6000.0
            node, edge, graph = observe(position, direction, speed, active, reference_altitude=origin)
            splits[name] = {
                "position": position, "direction": direction, "speed": speed, "expert": expert, "active": active,
                "node": node, "edge": edge, "adjacency": graph.adjacency,
                "certified": torch.as_tensor(data[f"{name}_certified"], device=device),
                "nominal": torch.as_tensor(data[f"{name}_nominal"], device=device),
            }
            if altitude_supported:
                splits[name]["altitude"] = torch.as_tensor(data[f"{name}_altitude"], device=device)
                splits[name]["altitude_certified"] = torch.as_tensor(data[f"{name}_altitude_certified"], device=device)
    return splits


def _select(split, indices):
    return {key: values[indices] for key, values in split.items()}


def _proposal(policy, speed):
    first_segment = policy.mode[..., 1] * policy.speed[..., 0]
    return (speed + SPEED_UNIT * first_segment).clamp(SPEED_MIN, SPEED_MAX)


def _plan(batch, speed, iterations, policy=None):
    tuning = {}
    if policy is not None:
        warm, rho, mobility, _, _ = policy.consensus_inputs
        tuning = {"rho": rho, "mobility": mobility, "warm_dual": warm}
    return speed_admm(batch["position"], batch["direction"], speed, batch["active"], batch["adjacency"],
                      minimum_speed=SPEED_MIN, maximum_speed=SPEED_MAX, horizon=HORIZON, radius=RADIUS,
                      iterations=iterations, **tuning)


def _altitude_plan(batch, speed, iterations, policy=None):
    proposal = torch.zeros_like(speed)
    tuning = {}
    if policy is not None:
        proposal = LAYER * policy.mode[..., 2] * policy.altitude
        tuning = {"rho": policy.altitude_rho, "mobility": policy.mobility, "warm_dual": policy.altitude_dual}
    velocity = speed.unsqueeze(-1) * batch["direction"][..., :2]
    return altitude_admm(batch["position"], velocity, proposal, batch["active"], horizon=HORIZON,
                         horizontal_radius=RADIUS, minimum_altitude=-2 * LAYER, maximum_altitude=3 * LAYER,
                         iterations=iterations, **tuning)


def _clear(batch, first, iterations, policy=None):
    if "altitude" not in batch:
        return first.feasible
    second = _altitude_plan(batch, first.speed, iterations, policy)
    return first.feasible | second.feasible


def _policy(model, batch, temperature=1.0):
    return model(batch["node"], batch["edge"], batch["adjacency"], batch["active"], temperature=temperature)


def _loss(model, batch, temperature):
    prediction = _policy(model, batch, temperature)
    active, speed, expert = batch["active"], batch["speed"], batch["expert"]
    target = (expert - speed) / SPEED_UNIT
    intent = ((expert - speed).abs() > 0.05).long()
    altitude_loss = 0.0
    if "altitude" in batch:
        target_altitude = batch["altitude"] / LAYER
        intent = torch.where(target_altitude.abs() > 0.01, 2, intent)
        routed = prediction.mode[..., 2] * prediction.altitude
        altitude_loss = (routed[active] - target_altitude[active]).square().mean()
        altitude_loss = altitude_loss + 0.5 * (prediction.altitude[active] - target_altitude[active]).square().mean()
    route = F.cross_entropy(prediction.logits[active], intent[active], weight=speed.new_tensor([1, 2, 3, 1]))
    proposal = _proposal(prediction, speed)
    deployed = _plan(batch, proposal, 2, prediction)
    normalized_action = (proposal - speed) / SPEED_UNIT
    per_segment = (prediction.speed - target.unsqueeze(-1)).square().mean(dim=-1)
    imitation = (normalized_action[active] - target[active]).square().mean() + per_segment[active].mean()
    safety = ((-deployed.minimum_clearance).clamp_min(0) / RADIUS).square().mean()
    if "altitude" in batch:
        second = _altitude_plan(batch, deployed.speed, 2, prediction)
        speed_violation = (-deployed.minimum_clearance).clamp_min(0) / RADIUS
        altitude_violation = (-second.minimum_clearance).clamp_min(0) / LAYER
        safety = torch.where(batch["altitude_certified"], altitude_violation, speed_violation).square().mean()
    intervention = ((deployed.speed[active] - expert[active]) / SPEED_UNIT).square().mean()
    return imitation + altitude_loss + 0.25 * route + 0.6 * safety + 0.25 * intervention


@torch.no_grad()
def evaluate(model, split, batch_size):
    model.eval()
    outcomes = {key: [] for key in ("nominal", "expert", "raw", "cold_k2", "cold_k10", "warm_k2")}
    mse_num, mse_den = 0.0, 0
    for index in range(0, len(split["speed"]), batch_size):
        batch = _select(split, slice(index, index + batch_size))
        prediction = _policy(model, batch)
        proposal = _proposal(prediction, batch["speed"])
        mask = batch["active"] & batch["certified"].unsqueeze(-1)
        errors = (proposal - batch["expert"]) / SPEED_UNIT
        mse_num += float(errors[mask].square().sum())
        mse_den += int(mask.sum())
        outcomes["nominal"].append(_plan(batch, batch["speed"], 0).feasible)
        outcomes["expert"].append(batch["certified"])
        outcomes["raw"].append(_clear(batch, _plan(batch, proposal, 0), 0, prediction))
        outcomes["cold_k2"].append(_clear(batch, _plan(batch, batch["speed"], 2), 2))
        outcomes["cold_k10"].append(_clear(batch, _plan(batch, batch["speed"], 10), 10))
        outcomes["warm_k2"].append(_clear(batch, _plan(batch, proposal, 2, prediction), 2, prediction))
    groups = {"all": torch.ones_like(split["certified"]), "expert_certified": split["certified"],
              "nominal_conflicts": ~split["nominal"],
              "expert_resolvable_conflicts": ~split["nominal"] & split["certified"]}
    if "altitude_certified" in split:
        groups["expert_altitude"] = split["altitude_certified"]
    metrics = {"normalized_action_mse_certified": mse_num / mse_den if mse_den else None}
    for name, group in groups.items():
        metrics[name] = {"scenes": int(group.sum()), **{
            key: int((torch.cat(values) & group).sum()) for key, values in outcomes.items()
        }}
    return metrics


def _save_json(report, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def train(args):
    if args.device == "ppu" and importlib.util.find_spec("torch_ppu") is not None:
        __import__("torch_ppu")
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    if args.device == "ppu" and (torch.cuda.device_count() != 1 or "PPU" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Run PPU training within a single PPU allocation")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    data = _load(args.data, device)
    train_set = _select(data["train"], data["train"]["certified"])
    if len(train_set["speed"]) < 2 or not data["validation"]["certified"].any():
        raise ValueError("The pilot needs certified training and validation scenes")
    model = Actor(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    report = {
        "seed": args.seed, "torch": torch.__version__, "device": str(device),
        "accelerator": torch.cuda.get_device_name(0) if args.device == "ppu" else None,
        "actor_parameters": sum(p.numel() for p in model.parameters()),
        "width": args.width, "batch": args.batch, "epochs": [], "unique_certified_train": len(train_set["speed"]),
    }
    best = None
    checkpoint = args.output / "actor.pt"
    for epoch in range(args.epochs):
        begin = perf_counter()
        model.train()
        order = torch.randperm(len(train_set["speed"]), device=device)
        losses = []
        temperature = max(0.1, 1.0 - epoch / args.epochs)
        for indices in order.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, _select(train_set, indices), temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Nonfinite gradients in epoch {epoch + 1}")
            optimizer.step()
            losses.append(loss.detach())
        validation = evaluate(model, data["validation"], args.batch)
        score = (validation["expert_resolvable_conflicts"]["warm_k2"],
                 -validation["normalized_action_mse_certified"])
        if best is None or score > best:
            best = score
            report["selected_epoch"] = epoch + 1
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({"actor": model.state_dict(), "width": args.width, "epoch": epoch + 1}, tmp)
            tmp.replace(checkpoint)
        epoch_report = {"epoch": epoch + 1, "loss": float(torch.stack(losses).mean()),
                        "seconds": perf_counter() - begin, "validation": validation}
        report["epochs"].append(epoch_report)
        _save_json(report, args.output / "training.json")
        print(json.dumps(epoch_report), flush=True)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["actor"])
    report["validation"] = evaluate(model, data["validation"], args.batch)
    report["ood"] = evaluate(model, data["ood"], args.batch)
    if args.device == "ppu":
        report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    _save_json(report, args.output / "training.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "ppu"), default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=21701)
    train(parser.parse_args())
