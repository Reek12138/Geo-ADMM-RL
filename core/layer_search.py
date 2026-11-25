"""Batched compatibility and local search for discrete altitude commands."""

from functools import lru_cache
from itertools import combinations

import torch


def layer_clearance(height, vertical_speed, target, entry, exit, *, accelerated=False):
    """Minimum vertical clearance over the horizontal overlap."""
    displacement = target - height.unsqueeze(-1)
    sign = displacement.sign()
    rate, acceleration = 7.62, 1.524
    speed = (sign * vertical_speed.unsqueeze(-1)).clamp(-rate, rate)
    reversal = (-speed / acceleration).clamp_min(0.0)
    reversal_height = height.unsqueeze(-1) + sign * speed * reversal * 0.5
    acceleration_time = (rate - speed) / acceleration
    acceleration_distance = (rate + speed) * acceleration_time * 0.5
    distance = displacement.abs()
    early_capture = ((speed.square() + 2 * acceleration * distance).sqrt() - speed) / acceleration
    capture = torch.where(distance <= acceleration_distance, early_capture,
                          acceleration_time + (distance - acceleration_distance) / rate)
    reversal = torch.where(displacement == 0, 0.0, reversal)
    capture = torch.where(displacement == 0, 0.0, capture)

    # Give each aircraft axis its own dimension for all pair and command combinations.
    def first(value):
        return value.unsqueeze(-2).unsqueeze(-1)

    def second(value):
        return value.unsqueeze(-3).unsqueeze(-2)

    critical = (first(reversal), second(reversal), first(capture), second(capture))
    if accelerated:
        accelerate_until = torch.minimum(acceleration_time, capture)

        def altitude_at(time, place):
            stop = place(capture)
            capped = torch.minimum(time, stop)
            ramp = torch.minimum(capped, place(accelerate_until))
            distance = place(speed) * ramp + 0.5 * acceleration * ramp.square() + rate * (capped - ramp)
            return torch.where(time >= stop, place(target), place(height.unsqueeze(-1)) + place(sign) * distance)

        # Relative altitude is quadratic while either aircraft accelerates.
        # Inspect both phase boundaries and stationary points of each possible phase pair.
        first_acc, second_acc = first(sign * acceleration), second(sign * acceleration)
        first_vs, second_vs = first(sign * speed), second(sign * speed)
        first_cruise, second_cruise = first(sign * rate), second(sign * rate)

        def root(delta_speed, delta_acc):
            return delta_speed / torch.where(delta_acc != 0, delta_acc, 1.0)

        critical += (first(accelerate_until), second(accelerate_until),
                     root(second_vs - first_vs, first_acc - second_acc),
                     root(second_cruise - first_vs, first_acc),
                     root(first_cruise - second_vs, second_acc))
    else:
        def altitude_at(time, place):
            initial = place(height.unsqueeze(-1))
            turning = place(reversal)
            midpoint = place(reversal_height)
            duration = place(capture) - turning
            before = initial.lerp(midpoint, (time / turning.clamp_min(1e-5)).clamp(0, 1))
            after = midpoint.lerp(place(target), ((time - turning) / duration.clamp_min(1e-5)).clamp(0, 1))
            return torch.where(time < turning, before, after)

    lower, upper = entry[..., None, None], exit[..., None, None]
    def gap(time):
        return altitude_at(time, first) - altitude_at(time, second)

    start, end = gap(lower), gap(upper)
    minimum, maximum = torch.minimum(start, end), torch.maximum(start, end)
    for time in critical:
        delta = gap(time.clamp(lower, upper))
        minimum = torch.minimum(minimum, delta)
        maximum = torch.maximum(maximum, delta)
    return torch.maximum(minimum, -maximum).clamp_min(0)


def improve_layer_choices(compatible, *, rounds=4, preferred=None, preserve=None):
    """Try paired commands without introducing new nominally unsafe pairs."""
    batch, aircraft, _, options, _ = compatible.shape
    choice = torch.zeros(batch, aircraft, dtype=torch.long, device=compatible.device)
    indices = torch.arange(aircraft, device=compatible.device).unsqueeze(0)
    upper = indices.unsqueeze(-1) < indices.unsqueeze(-2)

    def other_edges(changed):
        # Exclude the shared edge from each candidate, then add it once with both new commands.
        count = changed.sum(dim=-2, keepdim=True, dtype=torch.int32) - changed.to(torch.int32)
        return count.unsqueeze(-1) + count.transpose(1, 2).unsqueeze(-2)

    def edge_deltas(table, other, current):
        by_option = table.gather(-1, other).squeeze(-1)
        safe = by_option.gather(-1, current)
        removed = other_edges(~safe & by_option) + (~safe.unsqueeze(-1) & table).to(torch.int32)
        introduced = other_edges(safe & ~by_option) + (safe.unsqueeze(-1) & ~table).to(torch.int32)
        return removed, introduced

    for _ in range(rounds):
        other = choice[:, None, :, None, None].expand(batch, aircraft, aircraft, options, 1)
        current = choice[:, :, None, None].expand(batch, aircraft, aircraft, 1)
        removed, introduced = edge_deltas(compatible, other, current)
        gains = removed
        if preferred is not None:
            safe_gain, safe_loss = edge_deltas(preferred, other, current)
            # Prefer a wider clearance without trading away a nominal conflict resolution.
            gains = gains * (4 * aircraft + 1) + safe_gain - safe_loss
        if preserve is not None:
            _, lost_clearance = edge_deltas(preserve, other, current)
            introduced = introduced + lost_clearance
        gains = gains.masked_fill((introduced != 0) | ~upper[..., None, None], 0)
        best, selected = gains.flatten(1).max(dim=-1, keepdim=True)
        pair = selected // (options * options)
        choice = torch.where((indices == pair // aircraft) & (best > 0), selected // options % options, choice)
        choice = torch.where((indices == pair % aircraft) & (best > 0), selected % options, choice)
    return choice


@lru_cache(maxsize=8)
def _aircraft_triples(aircraft, device):
    return torch.tensor(list(combinations(range(aircraft), 3)), dtype=torch.long, device=device)


def improve_three_aircraft_choices(compatible, choices, components, *, preferred=None, preserve=None):
    """Improve nominal clearance first and then the preferred margin."""
    batch_size, aircraft, _, option_count, _ = compatible.shape
    if aircraft < 3:
        return choices
    other_options = choices[:, None, :, None, None].expand(-1, aircraft, -1, option_count, 1)
    single = compatible.gather(-1, other_options).squeeze(-1)
    before = single.gather(-1, choices[:, :, None, None].expand(-1, -1, aircraft, 1)).squeeze(-1)
    component_size = (components.unsqueeze(-1) == components.unsqueeze(-2)).sum(dim=-1)
    deficient = ~before
    if preferred is not None:
        wide = preferred.gather(-1, other_options).squeeze(-1)
        deficient |= ~wide.gather(-1, choices[:, :, None, None].expand(-1, -1, aircraft, 1)).squeeze(-1)
    eligible = (deficient & (component_size >= 3).unsqueeze(-1)).any(dim=(-1, -2))
    worlds = eligible.nonzero().flatten()
    if not worlds.numel():
        return choices
    if worlds.numel() == batch_size:
        return _search_triples(compatible, choices, components, single, before, preferred, preserve)

    subset = tuple(value.index_select(0, worlds) for value in (compatible, choices, components, single, before))
    preferred = preferred.index_select(0, worlds) if preferred is not None else None
    reserved = preserve.index_select(0, worlds) if preserve is not None else None
    revised = _search_triples(*subset, preferred, reserved)
    return choices.index_copy(0, worlds, revised)


def _search_triples(compatible, choices, components, single, before, preferred, preserve):
    batch_size, aircraft, _, option_count, _ = compatible.shape
    # Any newly unsafe pair outweighs all possible gains in the airspace.
    pairs = aircraft * (aircraft - 1) // 2
    nominal_weight = 2 * pairs + 1 if preferred is not None else 1
    penalty = pairs * (nominal_weight + (preferred is not None)) + 1

    def score(old, new, permitted=None):
        lost = old & ~new
        if permitted is not None:
            lost |= ~permitted
        return nominal_weight * (~old & new).to(torch.int32) - penalty * lost.to(torch.int32)

    other_options = choices[:, None, :, None, None].expand(-1, aircraft, -1, option_count, 1)
    permitted_single = None
    if preserve is not None:
        permitted_single = preserve.gather(-1, other_options).squeeze(-1)
    single_score = score(before.unsqueeze(-1), single, permitted_single)
    if preferred is not None:
        wide = preferred.gather(-1, other_options).squeeze(-1)
        wide_before = wide.gather(-1, choices[:, :, None, None].expand(-1, -1, aircraft, 1)).squeeze(-1)
        single_score += wide.to(torch.int32) - wide_before.unsqueeze(-1).to(torch.int32)
    incident_score = single_score.sum(dim=2, dtype=torch.int32)
    triples = _aircraft_triples(aircraft, compatible.device)
    first, second, third = triples.unbind(-1)
    linked = (components[:, first] == components[:, second]) & (components[:, first] == components[:, third])
    worlds, triplets = linked.nonzero(as_tuple=True)
    if not worlds.numel():
        return choices
    options = torch.arange(option_count, device=compatible.device)
    other_choice = options != choices.unsqueeze(-1)

    # Score option cubes only for triples that share a component in this world.
    options_per_triple = option_count ** 3
    chunk_size = max(1, (1 << 20) // options_per_triple)
    stride = len(triples) * options_per_triple
    best = choices.new_zeros(batch_size)
    for world, triplet in zip(worlds.split(chunk_size), triplets.split(chunk_size), strict=True):
        first, second, third = triples[triplet].unbind(-1)
        first_external = (incident_score[world, first] - single_score[world, first, second]
                          - single_score[world, first, third])
        second_external = (incident_score[world, second] - single_score[world, second, first]
                           - single_score[world, second, third])
        third_external = (incident_score[world, third] - single_score[world, third, first]
                          - single_score[world, third, second])

        def joint_score(left, right):
            permitted = preserve[world, left, right] if preserve is not None else None
            value = score(before[world, left, right, None, None], compatible[world, left, right], permitted)
            if preferred is not None:
                value += preferred[world, left, right].to(torch.int32)
                value -= wide_before[world, left, right, None, None].to(torch.int32)
            return value

        first_second = joint_score(first, second)
        first_third = joint_score(first, third)
        second_third = joint_score(second, third)

        gains = (first_external[..., :, None, None] + second_external[..., None, :, None]
                 + third_external[..., None, None, :] + first_second.unsqueeze(-1)
                 + first_third.unsqueeze(-2) + second_third.unsqueeze(-3))
        allowed = (other_choice[world, first, :, None, None] & other_choice[world, second, None, :, None]
                   & other_choice[world, third, None, None, :])
        values, index = gains.masked_fill(~allowed, 0).flatten(1).max(dim=-1)
        # The exhaustive order used the first triple and option on a tie.
        candidate = triplet * options_per_triple + index
        ranked = torch.where(values > 0, values.long() * stride + (stride - 1 - candidate), 0)
        best.scatter_reduce_(0, world, ranked, reduce="amax")

    best_index = stride - 1 - best.remainder(stride)
    selected = triples[best_index // options_per_triple]
    targets = torch.stack((best_index // option_count ** 2 % option_count,
                           best_index // option_count % option_count, best_index % option_count), dim=-1)
    return torch.where((best > 0).unsqueeze(-1), choices.scatter(-1, selected, targets), choices)
