"""Continuous-time forbidden speed interval for one aircraft against fixed traffic."""

import torch

from .conflict_graph import _ball_interval


def _quadratic_roots(a, b, c):
    """Real root candidates, with NaNs for degenerate or non-real candidates."""
    quadratic = a.abs() > 1e-12
    linear = (~quadratic) & (b.abs() > 1e-12)
    delta = b.square() - 4 * a * c
    root = delta.clamp_min(0).sqrt()
    sign = torch.where(b >= 0, 1.0, -1.0)
    q = -0.5 * (b + sign * root)
    one = q / torch.where(quadratic, a, 1.0)
    two = torch.where(q != 0, c / q, one)
    first = torch.where(quadratic & (delta >= 0), one, torch.nan)
    second = torch.where(quadratic & (delta >= 0), two, torch.nan)
    first = torch.where(linear, -c / torch.where(linear, b, 1.0), first)
    return first, second


def speed_forbidden_interval(relative, ego_axis, other_velocity, vertical_rate, *, lower, upper,
                             horizontal_radius, vertical_radius, horizon):
    """Project the collision cone onto ego speed while holding neighbor velocity fixed.

    Relative position is ego minus neighbor. The horizontal ego heading is unit
    length. Vertical relative motion is supplied independently of the ego speed.
    Returned endpoints describe the closure of the forbidden interval.
    """
    perpendicular = torch.stack((-ego_axis[..., 1], ego_axis[..., 0]), dim=-1)
    r = relative[..., :2]
    r_along = (r * ego_axis).sum(dim=-1)
    r_cross = (r * perpendicular).sum(dim=-1)
    other_along = (other_velocity * ego_axis).sum(dim=-1)
    other_cross = (other_velocity * perpendicular).sum(dim=-1)
    enter, leave, vertical = _ball_interval(relative[..., 2:], vertical_rate.unsqueeze(-1), vertical_radius, horizon)
    d2 = r.square().sum(dim=-1)
    radius2 = horizontal_radius**2
    max_closure = max(abs(lower), abs(upper)) + other_velocity.norm(dim=-1)
    earliest = (d2.sqrt() - horizontal_radius) / max_closure.clamp_min(1e-10)
    first_time = torch.maximum(enter, earliest)
    inv_start = leave.clamp_min(1e-10).reciprocal()
    inv_end = first_time.clamp_min(1e-10).reciprocal()
    a = radius2 - r_cross.square()
    b = 2 * r_cross * other_cross
    c = -other_cross.square()
    extremum = a - r_along.square()
    vertex = -b / torch.where(a != 0, 2 * a, 1.0)
    candidates = torch.stack((
        inv_start, inv_end, vertex, *_quadratic_roots(a, b, c),
        *_quadratic_roots(a * extremum, b * extremum, b.square() / 4 - r_along.square() * c),
    ), dim=-1)
    finite = torch.isfinite(candidates) & (candidates >= inv_start.unsqueeze(-1)) & (
        candidates <= inv_end.unsqueeze(-1)
    )
    x = torch.where(finite, candidates, 0.0)
    q = (a.unsqueeze(-1) * x + b.unsqueeze(-1)) * x + c.unsqueeze(-1)
    # Contact endpoints can have a small negative radicand from roundoff.
    scale = a.abs().unsqueeze(-1) * x.square() + b.abs().unsqueeze(-1) * x.abs() + c.abs().unsqueeze(-1)
    tolerance = 16 * torch.finfo(relative.dtype).eps * (scale + 1)
    valid = finite & (q >= -tolerance)
    center = other_along.unsqueeze(-1) - r_along.unsqueeze(-1) * x
    half_width = q.clamp_min(0).sqrt()
    lo = (center - half_width).masked_fill(~valid, torch.inf).amin(dim=-1).clamp_min(lower)
    hi = (center + half_width).masked_fill(~valid, -torch.inf).amax(dim=-1).clamp_max(upper)
    inside_now = (enter == 0) & (d2 < radius2)
    lo = torch.where(inside_now, lower, lo)
    hi = torch.where(inside_now, upper, hi)
    tangent_now = (enter == 0) & (d2 == radius2)
    radial = (r * other_velocity).sum(dim=-1)
    tangent_speed = radial / torch.where(r_along != 0, r_along, 1.0)
    tangent_lo = torch.where(r_along < 0, tangent_speed.clamp_min(lower), lower)
    tangent_hi = torch.where(r_along > 0, tangent_speed.clamp_max(upper), upper)
    tangent_hi = torch.where((r_along == 0) & (radial <= 0), lower, tangent_hi)
    lo = torch.where(tangent_now, tangent_lo, lo)
    hi = torch.where(tangent_now, tangent_hi, hi)
    open_horizontal = (finite & (q > 0)).any(dim=-1)
    exists = vertical & (((leave > first_time) & open_horizontal) | inside_now | tangent_now) & (hi > lo)
    return torch.where(exists, lo, 0.0), torch.where(exists, hi, 0.0), exists
