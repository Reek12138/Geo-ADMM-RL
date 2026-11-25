"""Warm-start the policy with audited, single-axis BlueSky teacher commands."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from eval.bluesky_pair import _checksum
from envs.bluesky_graph import ObservationProfile
from models.graphformer import Actor
from train.td3 import _anchor_loss, _teacher_loss, load_teacher_data


def train(args):
    if args.updates <= 0 or args.batch <= 0:
        raise ValueError("Teacher training needs a positive update and batch count")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.source, map_location=device, weights_only=True)
    actor = Actor(width=state["width"]).to(device)
    actor.load_state_dict(state["actor"])
    anchor = deepcopy(actor).eval().requires_grad_(False)
    profile = ObservationProfile(**state.get("observation_profile", {}))
    demos = load_teacher_data(args.data, device, profile)
    graph = tuple(torch.cat(columns) for columns in zip(*(item[0] for item in demos), strict=True))
    commands = torch.cat([item[1] for item in demos])
    mask = torch.cat([item[2] for item in demos])

    args.output.mkdir(parents=True)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    diagnose = torch.arange(min(len(commands), 64), device=device)

    @torch.no_grad()
    def training_sample_loss():
        sample = tuple(value[diagnose] for value in graph)
        return float(_teacher_loss(actor.eval()(*sample), commands[diagnose], mask[diagnose]))

    initial = training_sample_loss()
    actor.train()
    for _ in range(args.updates):
        indices = torch.randint(len(commands), (args.batch,), generator=generator, device=device)
        batch = tuple(value[indices] for value in graph)
        policy = actor(*batch)
        with torch.no_grad():
            reference = anchor(*batch)
        loss = _teacher_loss(policy, commands[indices], mask[indices]) + 0.5 * _anchor_loss(policy, reference, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()

    final = training_sample_loss()
    target = args.output / "actor.pt"
    torch.save({"actor": actor.state_dict(), "width": state["width"], "observation_profile": asdict(profile)}, target)
    (args.output / "training.json").write_text(json.dumps({
        "source": str(args.source), "source_sha256": _checksum(args.source), "output_sha256": _checksum(target),
        "data": [str(path) for path in args.data], "seed": args.seed, "updates": args.updates,
        "teacher_rows": len(commands), "initial_training_sample_loss": initial, "final_training_sample_loss": final,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--data", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--updates", type=int, default=128)
    cli.add_argument("--batch", type=int, default=16)
    cli.add_argument("--seed", type=int, default=67393)
    cli.add_argument("--learning-rate", type=float, default=4e-5)
    cli.add_argument("--device", default="cpu")
    train(cli.parse_args())
