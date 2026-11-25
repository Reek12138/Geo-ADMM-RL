"""Decentralized gap-tick MPC using passive ADS-B tracks and accelerated vertical kinematics."""

import torch

from core.conflict_graph import _ball_interval
from envs.bluesky_graph import SCALE
from envs.bluesky_safety import PhysicalState, _active_pairs, _forecast_command
from envs.observations import LAYER

_VERTICAL_RATE = 7.62
_VERTICAL_ACCEL = 1.524


def _vertical_profile(height, vertical_speed, target):
    displacement = target - height
    sign = displacement.sign()
    speed = (sign * vertical_speed).clamp(-_VERTICAL_RATE, _VERTICAL_RATE)
    reversal = torch.where(displacement == 0, 0.0, (-speed / _VERTICAL_ACCEL).clamp_min(0.0))
    accel_time = (_VERTICAL_RATE - speed) / _VERTICAL_ACCEL
    accel_distance = (_VERTICAL_RATE + speed) * accel_time * 0.5
    distance = displacement.abs()
    early_capture = ((speed.square() + 2.0 * _VERTICAL_ACCEL * distance).sqrt() - speed) / _VERTICAL_ACCEL
    capture = torch.where(distance <= accel_distance, early_capture,
                          accel_time + (distance - accel_distance) / _VERTICAL_RATE)
    capture = torch.where(displacement == 0, 0.0, capture)
    accel_until = torch.minimum(accel_time, capture)
    return height, target, sign, speed, reversal, capture, accel_until


def _altitude_at_profile(profile, time):
    height, target, sign, speed, _, capture, accel_until = profile
    capped = torch.minimum(time, capture)
    ramp = torch.minimum(capped, accel_until)
    distance = speed * ramp + 0.5 * _VERTICAL_ACCEL * ramp.square() + _VERTICAL_RATE * (capped - ramp)
    return torch.where(time >= capture, target, height + sign * distance)


def _accelerated_vertical_clearance(z_a, vs_a, target_a, z_b, vs_b, target_b, lo, hi):
    """Exact minimum vertical separation under bounded rate and acceleration over [lo, hi]."""
    prof_a = tuple(t.unsqueeze(-1) for t in _vertical_profile(z_a, vs_a, target_a))
    prof_b = tuple(t.unsqueeze(-2) for t in _vertical_profile(z_b, vs_b, target_b))
    _, _, sign_a, speed_a, rev_a, cap_a, acc_a = prof_a
    _, _, sign_b, speed_b, rev_b, cap_b, acc_b = prof_b

    first_acc, second_acc = sign_a * _VERTICAL_ACCEL, sign_b * _VERTICAL_ACCEL
    first_vs, second_vs = sign_a * speed_a, sign_b * speed_b
    first_cruise, second_cruise = sign_a * _VERTICAL_RATE, sign_b * _VERTICAL_RATE

    def root(delta_speed, delta_acc):
        return delta_speed / torch.where(delta_acc != 0, delta_acc, 1.0)

    def gap(time):
        return _altitude_at_profile(prof_a, time) - _altitude_at_profile(prof_b, time)

    start, end = gap(lo), gap(hi)
    minimum, maximum = torch.minimum(start, end), torch.maximum(start, end)
    critical = (
        rev_a, rev_b, cap_a, cap_b, acc_a, acc_b,
        root(second_vs - first_vs, first_acc - second_acc),
        root(second_cruise - first_vs, first_acc),
        root(first_cruise - second_vs, second_acc),
    )
    for time in critical:
        delta = gap(time.clamp(lo, hi))
        minimum = torch.minimum(minimum, delta)
        maximum = torch.maximum(maximum, delta)
    return torch.maximum(minimum, -maximum).clamp_min(0.0)


def _predicted_clearance(own, own_vs, other, other_vs, active, *, margin=0.0):
    position = own.position[..., :2].unsqueeze(-2) - other.position[..., :2].unsqueeze(-3)
    a = own.direction[..., :2] * own.ground_speed.unsqueeze(-1)
    b = other.direction[..., :2] * other.ground_speed.unsqueeze(-1)
    velocity = a.unsqueeze(-2) - b.unsqueeze(-3)
    lo, hi, across = _ball_interval(position, velocity, SCALE.horizontal_radius, SCALE.horizon)
    clearance = _accelerated_vertical_clearance(
        own.position[..., 2], own_vs, own.selected_altitude,
        other.position[..., 2], other_vs, other.selected_altitude,
        lo, hi,
    )
    overlap = across & _active_pairs(active).unsqueeze(1)
    raw_shortfall = (LAYER - clearance).masked_fill(~overlap, 0.0).clamp_min(0.0)
    buf_shortfall = (LAYER + margin - clearance).masked_fill(~overlap, 0.0).clamp_min(0.0)
    return raw_shortfall > 1e-4, buf_shortfall > 1e-4, buf_shortfall


def plan_gap_maneuver(physical, active, proposal, *, margin=6.0):
    """Choose a decentralized gap-tick action from passive ADS-B tracks alone.

    Each aircraft relies exclusively on locally sensed tracks (no inter-aircraft
    communication or centralized joint rollback):
      - Evaluates speed/altitude candidates with exact B737 accelerated vertical
        kinematics and a configurable vertical clearance margin.
      - Breaks vertical symmetry deterministically from relative altitude (tie-
        broken by slot index) and verifies every candidate against both nominal
        neighbors and neighbors executing their deterministic vertical sense.
      - Preserves safe conflict-free heading-recovery proposals so off-route
        aircraft do not drift past the lateral corridor during long comm gaps.
    """
    batch, aircraft, _ = proposal.shape
    device, dtype = proposal.device, proposal.dtype
    axes = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=dtype, device=device,
    )
    moves = axes[None, :, None, :].expand(batch, -1, aircraft, -1)
    hold = torch.zeros_like(proposal)
    candidates = torch.cat((hold.unsqueeze(1), proposal.unsqueeze(1), moves), dim=1)
    candidates = candidates.masked_fill(~active[:, None, :, None], 0.0)

    state = PhysicalState(*(value.unsqueeze(1) for value in physical))
    nominal = _forecast_command(state, hold.unsqueeze(1), dense=False)
    true_vs = state.vertical_speed

    baseline_raw, baseline_buf, _ = _predicted_clearance(nominal, true_vs, nominal, true_vs, active, margin=margin)
    nominal_conflict = baseline_buf.squeeze(1)
    in_conflict = nominal_conflict.any(dim=-1)

    # Deterministic ADS-B vertical sense (higher climbs, lower descends; slot tie-break).
    altitude = physical[0][..., 2]
    slots = torch.arange(aircraft, device=device)
    tied = (altitude.unsqueeze(-1) - altitude.unsqueeze(-2)).abs() <= 1e-3
    higher = (altitude.unsqueeze(-1) > altitude.unsqueeze(-2) + 1e-3) | (tied & (slots.unsqueeze(-1) < slots.unsqueeze(-2)))
    want_up = (nominal_conflict & higher).any(dim=-1)
    want_down = (nominal_conflict & ~higher).any(dim=-1)
    sense = want_up.to(dtype) - want_down.to(dtype)

    # Predict neighbors' deterministic vertical sense from ADS-B tracks (zero comm).
    sense_cmd = torch.zeros_like(proposal)
    sense_cmd[..., 1] = sense
    sense_cmd = sense_cmd.masked_fill(~active.unsqueeze(-1), 0.0)
    sense_forecast = _forecast_command(state, sense_cmd.unsqueeze(1), dense=False)

    alternatives = _forecast_command(state, candidates, dense=False)
    alt_vs = true_vs.expand(-1, candidates.shape[1], -1)
    raw_nom, buf_nom, below_nom = _predicted_clearance(alternatives, alt_vs, nominal, true_vs, active, margin=margin)
    raw_sen, buf_sen, below_sen = _predicted_clearance(alternatives, alt_vs, sense_forecast, true_vs, active, margin=margin)

    raw_contacts = raw_nom | raw_sen
    buf_contacts = buf_nom | buf_sen
    shortfall = torch.maximum(below_nom, below_sen)

    # Enforce deterministic vertical sense and prohibit introducing contacts with clear neighbors.
    cand_alt = candidates[..., 1]
    sense_ok = torch.ones_like(cand_alt, dtype=torch.bool)
    sense_ok &= ~(want_up.unsqueeze(1) & ~want_down.unsqueeze(1) & (cand_alt < -1e-5))
    sense_ok &= ~(want_down.unsqueeze(1) & ~want_up.unsqueeze(1) & (cand_alt > 1e-5))
    sense_ok &= ~(want_up.unsqueeze(1) & want_down.unsqueeze(1) & (cand_alt.abs() > 1e-5))
    clear_neighbors = ~nominal_conflict.unsqueeze(1)
    creates_new_contact = (raw_contacts & clear_neighbors).any(dim=-1)
    eligible = sense_ok & ~creates_new_contact
    eligible[:, 0, :] = True  # Hold is always an eligible fallback.

    # Score conflicting aircraft lexicographically: raw contacts -> buffered contacts -> shortfall -> cost.
    raw_count = raw_contacts.sum(dim=-1)
    buf_count = buf_contacts.sum(dim=-1)
    penetration = shortfall.sum(dim=-1)
    cost = (candidates.abs() * proposal.new_tensor([1.0, 2.0, 4.0])).sum(dim=-1)

    big_int = aircraft + 1
    masked_raw = raw_count.masked_fill(~eligible, big_int)
    permitted = eligible & (raw_count == masked_raw.amin(dim=1, keepdim=True))
    masked_buf = buf_count.masked_fill(~permitted, big_int)
    permitted &= buf_count == masked_buf.amin(dim=1, keepdim=True)
    masked_pen = penetration.masked_fill(~permitted, float("inf"))
    permitted &= (penetration <= masked_pen.amin(dim=1, keepdim=True) + 1e-4)
    choice = cost.masked_fill(~permitted, float("inf")).argmin(dim=1)

    # For conflict-free aircraft, preserve safe heading-recovery proposals; otherwise hold.
    pure_heading = (proposal[..., 2].abs() > 0.05) & (proposal[..., :2].abs().amax(dim=-1) <= 1e-5)
    safe_heading = pure_heading & (buf_count[:, 1, :] == 0) & (raw_count[:, 1, :] == 0)
    clear_choice = torch.where(safe_heading, torch.ones_like(choice), torch.zeros_like(choice))
    choice = torch.where(in_conflict, choice, clear_choice)

    commands = candidates.gather(1, choice[:, None, :, None].expand(-1, -1, -1, 3)).squeeze(1)
    commands = commands.masked_fill(~active.unsqueeze(-1), 0.0)

    # Read-only telemetry for auditing (never mutates the decentralized commands).
    final_forecast = _forecast_command(state, commands.unsqueeze(1), dense=False)
    final_contacts, _, _ = _predicted_clearance(final_forecast, true_vs, final_forecast, true_vs, active, margin=0.0)
    hold_count = baseline_raw.sum(dim=(-1, -2, -3)) // 2
    planned_count = final_contacts.sum(dim=(-1, -2, -3)) // 2
    return commands, hold_count, planned_count
