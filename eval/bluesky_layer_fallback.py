"""CPU development projection for discrete altitude choices after an unresolved ADMM step."""

from math import radians

import torch

from core.conflict_graph import _ball_interval, connected_components
from envs.bluesky_graph import SCALE
from envs.bluesky_safety import CAS_DELTA, LAYER, PhysicalState, protected_command
from eval.bluesky_collision_trace import _component_plan, _native_vertical_clearance


def _partial_layer_choices(allowed, members):
    pairs = [(first, second) for index, first in enumerate(members) for second in members[index + 1:]]
    choices = dict.fromkeys(members, 0)

    def unsafe():
        return {(first, second) for first, second in pairs if not allowed[first][second][choices[first]][choices[second]]}

    initial = remaining = unsafe()
    order = sorted(members, key=lambda aircraft: -sum(aircraft in pair for pair in initial))
    for _ in range(3):
        previous = remaining
        for aircraft in order:
            chosen = choices[aircraft]
            for candidate in range(4):
                choices[aircraft] = candidate
                proposed = unsafe()
                if proposed < remaining:
                    chosen = candidate
                    remaining = proposed
            choices[aircraft] = chosen
        if remaining == previous:
            break
    return choices if remaining < initial else None


def _adjust_layers(controls, physical, active, feasible):
    affected = (active & ~feasible).any(dim=-1).nonzero().flatten()
    if affected.numel() == 0:
        return

    state = PhysicalState(*(value[affected] for value in physical))
    proposed = controls[affected]
    participating = active[affected]
    east, north = state.direction[..., :2].unbind(-1)
    turn = radians(5.0) * proposed[..., 2]
    direction = torch.stack((east * turn.cos() + north * turn.sin(), north * turn.cos() - east * turn.sin()), dim=-1)
    ground_delta = CAS_DELTA * (state.tas / state.cas.clamp_min(1)).clamp(0.8, 2.0)
    speed = state.ground_speed + ground_delta * proposed[..., 0]
    horizontal = state.position[..., :2]
    velocity = direction * speed.unsqueeze(-1)
    relative = horizontal.unsqueeze(-2) - horizontal.unsqueeze(-3)
    change = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
    lo, hi, overlaps = _ball_interval(relative, change, SCALE.horizontal_radius, SCALE.horizon)
    overlaps &= participating.unsqueeze(-1) & participating.unsqueeze(-2)
    overlaps &= ~torch.eye(participating.shape[-1], dtype=torch.bool, device=participating.device)
    if not overlaps.any():
        return

    height = state.position[..., 2]
    selected = state.selected_altitude
    requested = torch.where(proposed[..., 1] != 0, height + LAYER * proposed[..., 1], selected)
    targets = torch.stack((requested, selected, height - LAYER, height + LAYER), dim=-1).clamp(0.0, 304.8)
    clear = _native_vertical_clearance(height, targets, state.vertical_speed, lo, hi)
    allowed = (clear | ~overlaps[..., None, None]).tolist()
    labels = connected_components(overlaps)

    for index, world in enumerate(affected.tolist()):
        components = labels[index, overlaps[index].any(dim=-1)].unique().tolist()
        for component in components:
            members = (labels[index] == component).nonzero().flatten().tolist()
            compatibility = allowed[index]
            if all(compatibility[i][j][0][0] for i in members for j in members):
                continue
            domains = {aircraft: [0, 1, 2, 3] for aircraft in members}
            choices, _ = _component_plan(compatibility, members, choices=domains)
            if choices is None:
                choices = _partial_layer_choices(compatibility, members)
                if choices is None:
                    continue
            else:
                feasible[world, members] = True
            for aircraft, choice in choices.items():
                if choice == 0:
                    continue
                target = targets[index, aircraft, choice]
                if choice == 1 or target == selected[index, aircraft]:
                    controls[world, aircraft, 1] = 0
                else:
                    offset = ((target - height[index, aircraft]) / LAYER).clamp(-1.0, 1.0)
                    controls[world, aircraft, 1] = 1e-3 if offset.abs() < 1e-5 else offset


def protected_with_discrete_layers(action, physical, adjacency, active, consensus, iterations, fallback, **options):
    if action.device.type != "cpu":
        raise ValueError("Discrete layer development evaluation requires CPU")
    controls, feasible = protected_command(action, physical, adjacency, active, consensus, iterations, fallback,
                                           **options)
    # Bound the temporary pair-by-time clearance tensors on larger development panels.
    for start in range(0, len(action), 8):
        group = slice(start, start + 8)
        _adjust_layers(controls[group], tuple(value[group] for value in physical), active[group], feasible[group])
    return controls, feasible
