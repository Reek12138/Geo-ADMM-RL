"""Replan one bounded maneuver at a time between communication ticks."""

from dataclasses import fields

import torch

from core.conflict_graph import _ball_interval
from envs.bluesky_safety import PhysicalState, _active_pairs, _altitude_at, _forecast_command
from envs.bluesky_graph import SCALE
from envs.observations import LAYER


def _predicted_clearance(own, other, active):
    position = own.position[..., :2].unsqueeze(-2) - other.position[..., :2].unsqueeze(-3)
    a = own.direction * own.ground_speed.unsqueeze(-1)
    b = other.direction * other.ground_speed.unsqueeze(-1)
    velocity = a.unsqueeze(-2) - b.unsqueeze(-3)
    lo, hi, across = _ball_interval(position, velocity, SCALE.horizontal_radius, SCALE.horizon)
    a_z = own.position[..., 2].unsqueeze(-1)
    b_z = other.position[..., 2].unsqueeze(-2)
    a_goal = own.selected_altitude.unsqueeze(-1)
    b_goal = other.selected_altitude.unsqueeze(-2)
    a_rate = own.vertical_speed.unsqueeze(-1)
    b_rate = other.vertical_speed.unsqueeze(-2)
    a_capture = ((a_goal-a_z) / torch.where(a_rate != 0, a_rate, 1.)).clamp(lo, hi)
    b_capture = ((b_goal-b_z) / torch.where(b_rate != 0, b_rate, 1.)).clamp(lo, hi)
    times = torch.stack((lo, hi, a_capture, b_capture))
    gaps = _altitude_at(a_z, a_goal, a_rate, times) - _altitude_at(b_z, b_goal, b_rate, times)
    clearance = torch.maximum(gaps.amin(0), -gaps.amax(0)).clamp_min(0.)
    overlap = across & _active_pairs(active).unsqueeze(1)
    shortfall = (LAYER - clearance).masked_fill(~overlap, 0.).clamp_min(0.)
    return shortfall > 1e-4, shortfall


def plan_gap_maneuver(physical, active, proposal):
    """Choose a first action using sensed tracks and a 120 s hold-after-move forecast.

    Each aircraft evaluates its own candidates against neighbors holding their
    targets. A joint forecast may reject a combination that introduces a new
    predicted contact. This evaluator uses the current physical observations
    without an ADMM exchange on the gap tick.
    """
    axes = torch.eye(3, dtype=proposal.dtype, device=proposal.device)
    moves = torch.cat((axes, -axes))[None, :, None, :].expand(proposal.shape[0], -1, proposal.shape[1], -1)
    hold = torch.zeros_like(proposal)
    candidates = torch.cat((hold.unsqueeze(1), proposal.unsqueeze(1), moves), dim=1)
    candidates = candidates.masked_fill(~active[:, None, :, None], 0.)
    state = PhysicalState(*(value.unsqueeze(1) for value in physical))
    nominal = _forecast_command(state, hold.unsqueeze(1), dense=False)
    alternatives = _forecast_command(state, candidates, dense=False)
    predicted, below = _predicted_clearance(alternatives, nominal, active)
    per_aircraft = predicted.sum(dim=-1)
    permitted = per_aircraft == per_aircraft.amin(dim=1, keepdim=True)
    penetration = below.sum(dim=-1).masked_fill(~permitted, float("inf"))
    permitted &= penetration == penetration.amin(dim=1, keepdim=True)
    cost = (candidates.abs() * proposal.new_tensor([1., 2., 4.])).sum(dim=-1)
    choice = cost.masked_fill(~permitted, float("inf")).argmin(dim=1)
    commands = candidates.gather(1, choice[:, None, :, None].expand(-1, -1, -1, 3)).squeeze(1)

    baseline, baseline_shortfall = _predicted_clearance(nominal, nominal, active)
    while True:
        forecast = _forecast_command(state, commands.unsqueeze(1), dense=False)
        contacts, shortfall = _predicted_clearance(forecast, forecast, active)
        extra = (contacts & ~baseline).squeeze(1)
        remove = extra.any(dim=-1) | extra.any(dim=-2)
        if not remove.any():
            break
        commands = commands.masked_fill(remove.unsqueeze(-1), 0.)
    new_count = contacts.sum(dim=(-1, -2, -3))
    old_count = baseline.sum(dim=(-1, -2, -3))
    new_depth = shortfall.sum(dim=(-1, -2, -3))
    old_depth = baseline_shortfall.sum(dim=(-1, -2, -3))
    improves = (new_count < old_count) | ((new_count == old_count) & (new_depth < old_depth - 1e-4))
    commands = commands.masked_fill(~improves[:, None, None], 0.)
    return commands, old_count // 2, torch.where(improves, new_count, old_count) // 2
