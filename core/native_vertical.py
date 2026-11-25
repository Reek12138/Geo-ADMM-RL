"""BlueSky B737 vertical clearance on its native 0.05 s physics grid."""

import torch


FT = 0.3048
DT = 0.05
RATE = 1500 * FT / 60
ACCEL = 300 * FT / 60
ALTITUDE_BASE_FT = 20_000


def altitude_paths(height, vertical_speed, selected, commands, frames):
    """Return absolute altitudes, including the initial frame and target capture."""
    height = height.double().unsqueeze(-1) + ALTITUDE_BASE_FT * FT
    speed = vertical_speed.double().unsqueeze(-1)
    delta_ft = (commands.float() * 300).double()
    target = torch.where(delta_ft != 0, (height / FT + delta_ft).clamp(20_000, 21_000) * FT,
                         selected.double().unsqueeze(-1) + ALTITUDE_BASE_FT * FT)
    initial = height.expand_as(target).unsqueeze(-1)
    if frames == 0:
        return initial

    steps = torch.arange(1, frames + 1, device=height.device, dtype=torch.float64)
    desired = (target - height).sign() * RATE
    error = desired - speed
    acceleration_step = ACCEL * DT
    # BlueSky snaps to the desired rate once the remaining error is at most ACCEL.
    accelerating = error.abs().unsqueeze(-1) - (steps - 1) * acceleration_step > ACCEL + 1e-12
    rates = torch.where(accelerating, speed.unsqueeze(-1) + error.sign().unsqueeze(-1) * steps * acceleration_step,
                        desired.unsqueeze(-1))
    increment = rates * DT
    first = ((height + increment[..., 0]) * 1e6).round().unsqueeze(-1)
    remaining = (increment[..., 1:] * 1e6).round().cumsum(dim=-1)
    free = torch.cat((first, first + remaining), dim=-1) / 1e6
    previous = torch.cat((initial, free[..., :-1]), dim=-1)
    previous_rate = torch.cat((speed.expand_as(target).unsqueeze(-1), rates[..., :-1]), dim=-1)
    capture = (target.unsqueeze(-1) - previous).abs() <= 1.05 * DT * previous_rate.abs().clamp_min(RATE)
    captured = capture.cumsum(dim=-1, dtype=torch.int32) > 0
    return torch.cat((initial, torch.where(captured, target.unsqueeze(-1), free)), dim=-1)


def exact_imminent_clearance(clearance, height, vertical_speed, selected, commands, entry, exit, imminent):
    """Refine only imminent pairs, bounding the option-by-frame working set."""
    world, left, right = torch.triu(imminent, diagonal=1).nonzero(as_tuple=True)
    if not world.numel():
        return clearance
    first = (entry[world, left, right] / DT).ceil().long()
    last = (exit[world, left, right] / DT).floor().long()
    valid = first <= last
    world, left, right, first, last = (value[valid] for value in (world, left, right, first, last))
    if not world.numel():
        return clearance

    frames = int(last.max())
    # All ordinary B737 layer targets settle within this prefix. Check the tail
    # for atypical starting states before treating either path as constant.
    prefix = min(frames, 1200)
    aircraft = height.shape[-1]
    slots, inverse = torch.cat((world * aircraft + left, world * aircraft + right)).unique(return_inverse=True)
    inputs = (height.flatten()[slots], vertical_speed.flatten()[slots], selected.flatten()[slots],
              commands.flatten(0, 1)[slots])
    paths = altitude_paths(*inputs, prefix)
    if prefix < frames and not torch.equal(paths[..., -1], paths[..., -2]):
        paths = altitude_paths(*inputs, frames)
        prefix = frames

    ticks = torch.arange(prefix + 1, device=height.device)
    result = clearance.clone()
    indices = torch.arange(world.numel(), device=height.device)
    left_path, right_path = inverse.chunk(2)
    for tile in indices.split(64):
        one, two = paths[left_path[tile]], paths[right_path[tile]]
        inside = (ticks >= first[tile, None].clamp_max(prefix)) & (ticks <= last[tile, None])
        gap = (one.unsqueeze(-2) - two.unsqueeze(-3)).abs()
        refined = gap.masked_fill(~inside[:, None, None, :], torch.inf).amin(dim=-1).to(clearance.dtype)
        w, i, j = world[tile], left[tile], right[tile]
        result[w, i, j] = refined
        result[w, j, i] = refined.transpose(-1, -2)
    return result
