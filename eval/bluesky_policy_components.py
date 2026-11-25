"""Replay native, unprojected counterfactuals with selected policy components."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from bluesky.rl import BatchedBlueSkyConfig, enable_rl_fast_mode
import torch

from envs.bluesky_plan_a import GeoNativeBlueSky
from eval.bluesky_pair import FAMILIES, _checksum, _load
from train.td3 import encode_action


AXES = ("speed", "altitude", "heading")
OUTCOMES = ("arrivals", "collisions", "exits", "timeouts")


def mixed_commands(candidate, reference, axes, block):
    if block == "router":
        return encode_action(replace(candidate, mode=reference.mode))[..., 4:]
    if block == "magnitudes":
        return encode_action(replace(reference, mode=candidate.mode))[..., 4:]
    primary = encode_action(candidate)[..., 4:]
    alternate = encode_action(reference)[..., 4:]
    return torch.where(axes, alternate, primary)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    candidate, reference = (_load(path, device) for path in (args.candidate, args.reference))
    if candidate.observation_profile != reference.observation_profile:
        raise ValueError("Component replay requires identical observation profiles")
    args.workdir.mkdir()
    if args.navdata:
        cache = args.workdir / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(args.navdata)

    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=12, sim_dt=.05, control_dt=6,
                                 hdg_deadband_deg=.25, seed=args.seed)
    env = GeoNativeBlueSky(config, workdir=args.workdir, family=args.family,
                           **asdict(candidate.observation_profile))
    graph = env.reset()
    enable_rl_fast_mode(False, simdt=.05)
    axes = torch.tensor([name in args.reference_axes for name in AXES])
    totals = Counter(dict.fromkeys(OUTCOMES, 0))
    actions, simultaneous, contacts = 0, 0, []
    swapping = bool(args.reference_axes or args.reference_block)
    reference_visits = 0
    normalized_cost = 0.0
    for decision in range(config.max_steps):
        primary = candidate(*graph)
        commands = mixed_commands(primary, reference(*graph), axes, args.reference_block)
        active = graph[-1]
        selected = active
        if args.reference_scope != "all":
            anchor_eligible = graph[2].any(dim=-1) | (graph[0][..., 8] > .5)
            selected = active & (anchor_eligible if args.reference_scope == "anchor-eligible" else ~anchor_eligible)
            commands = torch.where(selected.unsqueeze(-1), commands, encode_action(primary)[..., 4:])
        commands = commands.masked_fill(~active.unsqueeze(-1), 0)
        actions += int(active.sum())
        if swapping:
            reference_visits += int(selected.sum())
        simultaneous += int((commands.abs().gt(1e-5).sum(dim=-1).gt(1) & active).sum())
        normalized_cost += float(commands.abs().sum())
        graph, _, done, info = env.step_native_audited(commands.numpy(), track_separation=False, track_contacts=True)
        totals.update({key: int(info[key][0]) for key in OUTCOMES})
        contacts.extend(dict(decision=decision + 1, **row) for row in info["contacts"])
        if done[0]:
            break
    if sum(totals.values()) != config.aircraft_per_env:
        raise ValueError("Native replay did not account for all aircraft")
    output = dict(seed=args.seed, family=args.family, sim_dt=config.sim_dt, control_dt=config.control_dt,
                  candidate_sha256=_checksum(args.candidate), reference_sha256=_checksum(args.reference),
                  reference_axes=args.reference_axes, reference_block=args.reference_block,
                  reference_scope=args.reference_scope, reference_eligible_visits=reference_visits,
                  projection=False, outcomes=dict(totals), contacts=contacts, decisions=decision + 1,
                  active_aircraft_decisions=actions, simultaneous_axis_commands=simultaneous,
                  normalized_cost_per_aircraft=normalized_cost / config.aircraft_per_env)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(args.family, args.seed, args.reference_block or args.reference_axes, dict(totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "workdir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--navdata", type=Path)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    mixing = parser.add_mutually_exclusive_group()
    mixing.add_argument("--reference-axes", nargs="+", choices=AXES, default=[])
    mixing.add_argument("--reference-block", choices=("router", "magnitudes"))
    parser.add_argument("--reference-scope", choices=("all", "anchor-eligible", "free-off-lnav"), default="all")
    args = parser.parse_args()
    if args.reference_scope != "all" and not (args.reference_axes or args.reference_block):
        parser.error("A restricted scope requires a reference component")
    run(args)
