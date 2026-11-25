"""Shared graph-policy learner for the BlueSky CPU and PPU rollout backends."""

from collections import deque
from dataclasses import dataclass, replace
from copy import deepcopy
from math import radians

import torch
from torch.nn import functional as F

from envs.bluesky_graph import ObservationProfile
from envs.bluesky_safety import PhysicalState, _directional_heading, coordination_loss, expected_mode_risk, lookahead_risk
from models.graphformer import Actor, TwinCritic, mirror_observations


REPLAY_EVENTS = ("ordinary", "arrival", "collision")


@dataclass
class Rollout:
    graph: tuple
    action: torch.Tensor
    reward: torch.Tensor
    next_graph: tuple
    terminal: torch.Tensor
    teacher: torch.Tensor
    teacher_mask: torch.Tensor
    physical: tuple
    category: torch.Tensor
    events: torch.Tensor | None = None
    collision_aircraft: torch.Tensor | None = None
    observed_collision: torch.Tensor | None = None
    route_count: torch.Tensor | None = None
    collision_return: torch.Tensor | None = None
    unprotected: torch.Tensor | None = None
    next_physical: tuple | None = None

    def tensors(self):
        events = self.events if self.events is not None else torch.zeros_like(self.category)
        labels = tuple(label for label in (self.unprotected, self.collision_aircraft, self.observed_collision,
                                           self.collision_return, self.route_count)
                       if label is not None)
        return (*self.graph, self.action, self.reward, *self.next_graph, self.terminal, self.teacher,
                self.teacher_mask, *self.physical, *(self.next_physical or ()), *labels, self.category, events)

    @classmethod
    def from_tensors(cls, tensors, *, collision_labels=False, observed_collision_labels=False, route_counts=False,
                     collision_returns=False, protection_labels=False, successor_geometry=False):
        collision_index = -2 - sum((collision_labels, observed_collision_labels, route_counts, collision_returns))
        physical_end = collision_index - int(protection_labels)
        collision_aircraft = tensors[collision_index] if collision_labels else None
        observed = tensors[collision_index + int(collision_labels)] if observed_collision_labels else None
        routes = tensors[-3] if route_counts else None
        collision_return = tensors[-3 - int(route_counts)] if collision_returns else None
        unprotected = tensors[physical_end] if protection_labels else None
        physical = tensors[13:physical_end]
        next_physical = None
        if successor_geometry:
            middle = len(physical) // 2
            physical, next_physical = physical[:middle], physical[middle:]
        return cls(tensors[:4], tensors[4], tensors[5], tensors[6:10], tensors[10], tensors[11], tensors[12],
                   physical, tensors[-2], tensors[-1], collision_aircraft, observed, routes,
                   collision_return, unprotected, next_physical)


def mirror_replay_batch(batch):
    """Pair each sampled airspace with its east-west reflection."""
    def signed_pair(value, columns):
        reflected = value.clone()
        reflected[..., columns] *= -1
        return torch.cat((value, reflected), dim=0)

    def pair(value):
        return None if value is None else torch.cat((value, value), dim=0)

    def graph_pair(graph):
        nodes, edges, adjacency, active = graph
        mirror_nodes, mirror_edges = mirror_observations(nodes, edges)
        return torch.cat((nodes, mirror_nodes)), torch.cat((edges, mirror_edges)), pair(adjacency), pair(active)

    def physical_pair(physical):
        position, direction, *scalars = physical
        return signed_pair(position, [0]), signed_pair(direction, [0]), *(pair(x) for x in scalars)

    return replace(
        batch, graph=graph_pair(batch.graph), next_graph=graph_pair(batch.next_graph),
        physical=physical_pair(batch.physical),
        next_physical=None if batch.next_physical is None else physical_pair(batch.next_physical),
        action=signed_pair(batch.action, [-1]), teacher=signed_pair(batch.teacher, [-1]),
        reward=pair(batch.reward), terminal=pair(batch.terminal), teacher_mask=pair(batch.teacher_mask),
        category=pair(batch.category), events=pair(batch.events), collision_aircraft=pair(batch.collision_aircraft),
        observed_collision=pair(batch.observed_collision), route_count=pair(batch.route_count),
        collision_return=pair(batch.collision_return), unprotected=pair(batch.unprotected),
    )


class CollisionHorizon:
    """Release a transition once its per-aircraft collision window is observed."""

    def __init__(self, steps):
        if steps < 1:
            raise ValueError("Collision horizon must be positive")
        self.steps = steps
        self.pending = deque()
        self.cursor = 0
        self.following = None

    @torch.no_grad()
    def append(self, rollout):
        collision = rollout.collision_aircraft
        if collision is None:
            raise ValueError("Collision horizon needs per-aircraft collision events")
        if self.following is None:
            self.following = collision.new_zeros((self.steps, *collision.shape))
            self.observed = torch.zeros_like(self.following)
        self.following[self.cursor].copy_(rollout.graph[-1])
        self.observed[self.cursor].zero_()
        self.observed.logical_or_(self.following & collision)
        # A terminal event applies to the current identity only, even if the slot is filled immediately.
        self.following.logical_and_(~rollout.terminal)
        self.pending.append(rollout)
        self.cursor = (self.cursor + 1) % self.steps
        if len(self.pending) < self.steps:
            return None
        return replace(self.pending.popleft(), observed_collision=self.observed[self.cursor].clone())


class MultiStepReturns:
    """Wait for a complete window and optionally aggregate per-aircraft returns."""

    def __init__(self, steps, discount, *, aggregate, collision_returns=False):
        if steps < 2:
            raise ValueError("Multi-step replay requires a window of at least two decisions")
        self.steps = steps
        self.discount = discount
        self.aggregate = aggregate
        self.collision_returns = collision_returns
        self.pending = deque()

    @torch.no_grad()
    def append(self, transition):
        self.pending.append(transition)
        if len(self.pending) < self.steps:
            return None
        first = self.pending.popleft()
        if not self.aggregate:
            return first

        reward = torch.zeros_like(first.reward)
        terminal = torch.zeros_like(first.terminal)
        collision = torch.zeros_like(reward) if self.collision_returns else None
        for item in (*reversed(self.pending), first):
            keep = ~item.terminal
            reward = item.reward + self.discount * torch.where(keep, reward, 0.0)
            terminal = terminal | item.terminal
            if collision is not None:
                collision = torch.where(item.collision_aircraft, 1.0,
                                        self.discount * torch.where(keep, collision, 0.0))
        last = self.pending[-1]
        return replace(first, reward=reward, terminal=terminal, next_graph=last.next_graph,
                       collision_return=collision, next_physical=last.next_physical)


class DeviceReplay:
    """Store full graph transitions on the learner device."""

    def __init__(self, capacity, device, seed, *, stratified=False, terminal_replay=False, collision_labels=False,
                 observed_collision_labels=False, route_counts=False, topology_replay=False, collision_returns=False,
                 protection_labels=False, successor_geometry=False):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        if terminal_replay and not stratified:
            raise ValueError("Terminal replay requires family-stratified replay")
        if topology_replay and not stratified:
            raise ValueError("Topology replay requires family-stratified replay")
        if (observed_collision_labels or collision_returns) and not collision_labels:
            raise ValueError("Collision horizons require per-step collision labels")
        self.capacity = capacity
        self.device = torch.device(device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.size = 0
        self.cursor = 0
        self.storage = None
        self.stratified = stratified
        self.terminal_replay = terminal_replay
        self.collision_labels = collision_labels
        self.observed_collision_labels = observed_collision_labels
        self.collision_returns = collision_returns
        self.protection_labels = protection_labels
        self.successor_geometry = successor_geometry
        self.route_counts = route_counts or topology_replay
        self.topology_replay = topology_replay
        if topology_replay:
            self.topology_rng = torch.Generator(device=device).manual_seed(seed + 2)
        if terminal_replay:
            self.event_rng = torch.Generator(device=device).manual_seed(seed + 1)

    @torch.no_grad()
    def append(self, rollout):
        if self.collision_labels != (rollout.collision_aircraft is not None):
            raise ValueError("Replay collision label configuration did not match the rollout")
        if self.observed_collision_labels != (rollout.observed_collision is not None):
            raise ValueError("Replay collision horizon configuration did not match the rollout")
        if self.route_counts != (rollout.route_count is not None):
            raise ValueError("Replay route counts did not match the rollout")
        if self.collision_returns != (rollout.collision_return is not None):
            raise ValueError("Replay discounted collision targets did not match the rollout")
        if self.protection_labels != (rollout.unprotected is not None):
            raise ValueError("Replay protection labels did not match the rollout")
        if self.successor_geometry != (rollout.next_physical is not None):
            raise ValueError("Replay successor geometry did not match the rollout")
        if self.successor_geometry and len(rollout.next_physical) != len(rollout.physical):
            raise ValueError("Replay requires matching current and successor physical fields")
        data = rollout.tensors()
        count = len(rollout.action)
        if count > self.capacity:
            data = tuple(value[-self.capacity:] for value in data)
            count = self.capacity
        if self.storage is None:
            self.storage = tuple(torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device)
                                 for value in data)
        leading = min(count, self.capacity - self.cursor)
        trailing = count - leading
        for target, source in zip(self.storage, data, strict=True):
            target[self.cursor:self.cursor + leading].copy_(source[:leading])
            target[:trailing].copy_(source[leading:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    @staticmethod
    def _family_budgets(count):
        crossing, opposing = (4 * count + 5) // 10, (3 * count + 5) // 10
        return crossing, opposing, count - crossing - opposing

    def _indices(self, count):
        if self.size == 0:
            raise ValueError("Replay is empty")
        if self.stratified:
            categories = self.storage[-2][:self.size]
            parts = []
            for category, budget in enumerate(self._family_budgets(count)):
                if budget > 0:
                    available = categories == category
                    weights = torch.where(available.any(), available, True).to(torch.float32)
                    selected = torch.multinomial(weights, budget, replacement=True, generator=self.rng)
                    quota = budget // 4
                    # Retain the near-head-on samples and their random stream.
                    if self.topology_replay and category != 1 and quota:
                        minimum_routes = 6 if category == 0 else 8
                        high_routes = available & (self.storage[-3][:self.size] >= minimum_routes)
                        present = high_routes.any()
                        weights = torch.where(present, high_routes, True).to(torch.float32)
                        difficult = torch.multinomial(weights, quota, replacement=True, generator=self.topology_rng)
                        selected[-quota:] = torch.where(present, difficult, selected[-quota:])
                    parts.append(selected)
            return torch.cat(parts)
        return torch.randint(self.size, (count,), device=self.device, generator=self.rng)

    def _take(self, indices):
        return Rollout.from_tensors(tuple(value[indices] for value in self.storage),
                                    collision_labels=self.collision_labels,
                                    observed_collision_labels=self.observed_collision_labels,
                                    route_counts=self.route_counts, collision_returns=self.collision_returns,
                                    protection_labels=self.protection_labels,
                                    successor_geometry=self.successor_geometry)

    def sample(self, count):
        return self._take(self._indices(count))

    def sample_for_learning(self, count, *, update_actor):
        if not self.terminal_replay:
            return self.sample(count), None

        regular = self._indices(count)
        critic = regular.clone()
        categories, events = (value[:self.size] for value in self.storage[-2:])
        start = 0
        for family, budget in enumerate(self._family_budgets(count)):
            # Reserve one eighth of each family for each available terminal event.
            quota = budget // 8
            if quota:
                belongs = categories == family
                for event in (1, 2):
                    matches = belongs & (events == event)
                    available = matches.any()
                    weights = torch.where(available, matches, True).to(torch.float32)
                    selected = torch.multinomial(weights, quota, replacement=True, generator=self.event_rng)
                    slots = slice(start + (event - 1) * quota, start + event * quota)
                    critic[slots] = torch.where(available, selected, regular[slots])
            start += budget
        return self._take(critic), self._take(regular) if update_actor else None


def encode_action(policy, noise=0.0):
    control = torch.stack((policy.speed[..., 0], policy.altitude, policy.heading), dim=-1)
    if noise:
        perturbation = (torch.randn_like(control) * noise).clamp(-0.5, 0.5)
        control = (control + perturbation).clamp(-1.0, 1.0)
    return torch.cat((policy.mode, policy.mode[..., 1:] * control), dim=-1)


class CrossingExploration:
    """Reuse exploration noise over several decisions of the same aircraft."""

    def __init__(self, active, steps, seed):
        self.steps = steps
        self.generator = torch.Generator(device=active.device).manual_seed(seed)
        self.remaining = torch.zeros_like(active, dtype=torch.int32)
        self.mode_noise = torch.zeros((*active.shape, 4), device=active.device)
        self.control_noise = torch.zeros((*active.shape, 3), device=active.device)

    def draw(self, active, category):
        crossing = active & (category == 0).unsqueeze(-1)
        fresh = crossing & (self.remaining == 0)
        modes = -torch.empty_like(self.mode_noise).exponential_(generator=self.generator).log_()
        controls = torch.empty_like(self.control_noise).normal_(generator=self.generator)
        self.mode_noise = torch.where(fresh.unsqueeze(-1), modes, self.mode_noise)
        self.control_noise = torch.where(fresh.unsqueeze(-1), controls, self.control_noise)
        self.remaining = torch.where(crossing, torch.where(fresh, self.steps - 1, self.remaining - 1), 0)
        return crossing, self.mode_noise, self.control_noise

    def end(self, terminal):
        self.remaining.masked_fill_(terminal, 0)


def _crossing_action(policy, original, exploration, temperature):
    mask, mode_noise, control_noise = exploration
    mode = F.one_hot((policy.logits / temperature + mode_noise).argmax(dim=-1), num_classes=4).to(original.dtype)
    control = torch.stack((policy.speed[..., 0], policy.altitude, policy.heading), dim=-1)
    control = (control + (0.1 * control_noise).clamp(-.5, .5)).clamp(-1, 1)
    alternative = torch.cat((mode, mode[..., 1:] * control), dim=-1)
    return torch.where(mask.unsqueeze(-1), alternative, original)


def encode_executed_action(proposal, commands):
    """Use the delivered commands and describe all axes changed by the projector."""
    changed = (commands - proposal[..., 4:]).abs().amax(dim=-1, keepdim=True) > 1e-5
    commanded = commands.abs() > 1e-5
    modes = torch.cat((~commanded.any(dim=-1, keepdim=True), commanded), dim=-1)
    modes = torch.where(changed, modes.to(proposal.dtype), proposal[..., :4])
    return torch.cat((modes, commands), dim=-1)


@torch.no_grad()
def project_critic_action(projector, policy, proposal, graph, physical, unprotected):
    consensus = policy.consensus_inputs
    commands, _ = projector(proposal, physical, graph[2], graph[-1], consensus)
    if unprotected is not None:
        commands = torch.where(unprotected[:, None, None], proposal[..., 4:], commands)
    return encode_executed_action(proposal, commands)


def projected_action_ste(proposal, executed):
    """Use the delivered action forward and the proposal Jacobian backward."""
    return executed.detach() + (proposal - proposal.detach())


def _active_mean(value, active):
    return value.masked_fill(~active, 0.0).sum() / active.sum().clamp_min(1)


def _collision_target(collision, bootstrap, first_probability, second_probability, discount):
    future_risk = discount * bootstrap * torch.maximum(first_probability, second_probability)
    return torch.where(collision, 1.0, future_risk)


def _navigation_loss(policy, graph, physical, safe_weight):
    node, _, adjacency, active = graph
    heading_error = torch.atan2(node[..., 5], node[..., 6])
    want_heading = (node[..., 8] < 0.5) & (heading_error.abs() > radians(0.25))
    clear = active & ~adjacency.any(dim=-1)
    mode = want_heading.long() * 3
    target = (heading_error / radians(5.0)).clamp(-1, 1)
    route = F.cross_entropy(policy.logits.flatten(0, -2), mode.flatten(), reduction="none").reshape_as(active)
    tracking = F.smooth_l1_loss(policy.heading, target, reduction="none")
    turn = clear & want_heading
    hold = clear & ~want_heading
    loss = 0.5 * (_active_mean(route, turn) + _active_mean(route, hold)) + _active_mean(tracking, turn)
    if safe_weight:
        linked_turn = active & ~clear & want_heading
        proposed = target.masked_fill(~linked_turn, 0.0)
        with torch.no_grad():
            state = PhysicalState(*physical)
            permitted = _directional_heading(state, state.ground_speed, proposed, torch.zeros_like(proposed),
                                             adjacency, active)
        safe_turn = linked_turn & (permitted.abs() > 0.05)
        loss = loss + safe_weight * _active_mean(route + tracking, safe_turn)
    return loss


def _projection_loss(policy, teacher, teacher_mask):
    magnitude = teacher.abs() * teacher.new_tensor([1.0, 3.0, 1.0])
    axis = magnitude.argmax(dim=-1, keepdim=True)
    choice = torch.where(magnitude.amax(dim=-1) > 0.01, axis.squeeze(-1) + 1, 0)
    route = F.cross_entropy(policy.logits.flatten(0, -2), choice.flatten(), reduction="none").reshape_as(choice)
    control = torch.stack((policy.speed[..., 0], policy.altitude, policy.heading), dim=-1)
    predicted = control.gather(-1, axis).squeeze(-1)
    target = teacher.gather(-1, axis).squeeze(-1)
    deviation = F.smooth_l1_loss(predicted, target, reduction="none")
    return _active_mean(route, teacher_mask) + _active_mean(deviation, teacher_mask & (choice > 0))


def _teacher_loss(policy, commands, eligible):
    move = commands.abs().amax(dim=-1) > 0.01
    return _projection_loss(policy, commands, eligible & move) + _projection_loss(
        policy, commands, eligible & ~move,
    )


def load_teacher_data(paths, device, profile=None):
    datasets = []
    for path in paths:
        data = torch.load(path, map_location=device, weights_only=True)
        if profile is not None and ObservationProfile(**data.get("observation_profile", {})) != profile:
            raise ValueError(f"Teacher and learner observation profiles differ: {path}")
        commands = data["commands"]
        if (not len(commands) or not torch.isfinite(commands).all() or (commands.abs() > 1).any()
                or (commands.ne(0).sum(dim=-1) > 1).any()):
            raise ValueError(f"Teacher data needs bounded, single-axis commands: {path}")
        graph = data["graph"]
        mask = data.get("command_mask")
        if mask is None:
            mask = graph[-1] & (graph[2].any(dim=-1) | (commands[..., 2].abs() > 0.05))
        elif (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
              or mask.shape != commands.shape[:-1] or mask.shape != graph[-1].shape
              or (mask & ~graph[-1]).any() or (~mask.any(dim=-1)).any()
              or (commands.ne(0).any(dim=-1) & ~mask).any()):
            raise ValueError(f"Teacher command mask must label active aircraft and cover all commands: {path}")
        datasets.append((graph, commands, mask))
    return tuple(datasets)


def _anchored_aircraft(graph):
    node, _, adjacency, active = graph
    return active & (adjacency.any(dim=-1) | (node[..., 8] > 0.5))


def _routing_divergence(policy, reference):
    return F.kl_div(policy.logits.log_softmax(dim=-1), reference.logits.softmax(dim=-1),
                    reduction="none").sum(dim=-1)


def _anchor_loss(policy, reference, graph, airspaces=None):
    preserve = _anchored_aircraft(graph)
    if airspaces is not None:
        preserve = preserve & airspaces.unsqueeze(-1)
    divergence = _routing_divergence(policy, reference)
    original = torch.stack((reference.speed[..., 0], reference.altitude, reference.heading), dim=-1)
    prediction = torch.stack((policy.speed[..., 0], policy.altitude, policy.heading), dim=-1)
    action_error = ((prediction - original).square() * reference.mode[..., 1:]).sum(dim=-1)
    return _active_mean(divergence + action_error, preserve)


class TD3:
    def __init__(self, device, width=256, actor_state=None, learning_rate=3e-4, discount=0.99, tau=0.005,
                 navigation_weight=0.0, projection_weight=0.0, anchor_state=None, anchor_weight=0.0,
                 raw_risk_weight=0.0, dense_risk=False, overlap_risk=False, head_on_anchor=False,
                 safe_navigation_weight=0.0, teacher_data=(), teacher_weight=0.0, rollout_router_temperature=1.0,
                 collision_critic_weight=0.0, collision_critic_warmup=600, expected_mode_risk_loss=False,
                 collision_critic_target="td", return_steps=1, coordination_weight=0.0, coordination_rounds=(2, 0),
                 anchor_categories=(), equivariant_actor=False):
        self.actor = Actor(width=width, equivariant_actor=equivariant_actor).to(device)
        if actor_state is not None:
            self.actor.load_state_dict(actor_state)
        self.critic = TwinCritic(width=width, action_features=7).to(device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=learning_rate)
        self.collision_critic_weight = collision_critic_weight
        self.collision_critic_warmup = collision_critic_warmup
        self.collision_critic_target = collision_critic_target
        self.collision_heads = None
        self.collision_updates = 0
        self.last_collision_loss = None
        if collision_critic_weight:
            with torch.random.fork_rng(devices=[]):
                self.collision_heads = torch.nn.ModuleList(torch.nn.Linear(width + 7, 1) for _ in range(2)).to(device)
            for head in self.collision_heads:
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.constant_(head.bias, -2.944439)
            self.target_collision_heads = deepcopy(self.collision_heads).requires_grad_(False)
            self.collision_optimizer = torch.optim.AdamW(self.collision_heads.parameters(), lr=learning_rate)
        self.discount = discount
        self.bootstrap_discount = discount ** return_steps
        self.tau = tau
        self.navigation_weight = navigation_weight
        self.safe_navigation_weight = safe_navigation_weight
        self.projection_weight = projection_weight
        self.anchor = None
        self.anchor_weight = anchor_weight
        self.anchor_start_update = 0
        selected_categories = (1,) if head_on_anchor else anchor_categories
        self.anchor_categories = torch.tensor(selected_categories, device=device) if selected_categories else None
        self.raw_risk_weight = raw_risk_weight
        self.expected_mode_risk_loss = expected_mode_risk_loss
        self.dense_risk = dense_risk
        self.overlap_risk = overlap_risk
        self.teacher_data = teacher_data
        self.teacher_weight = teacher_weight
        self.rollout_router_temperature = rollout_router_temperature
        self.coordination_weight = coordination_weight
        self.coordination_rounds = coordination_rounds
        self.last_coordination_loss = None
        self.teacher_rng = torch.Generator(device=device).manual_seed(torch.initial_seed()) if teacher_data else None
        if anchor_state is not None and anchor_weight:
            self.anchor = Actor(width=width).to(device).eval().requires_grad_(False)
            self.anchor.load_state_dict(anchor_state)
        self.updates = 0

    def collision_state(self):
        return {"heads": self.collision_heads.state_dict(), "target_heads": self.target_collision_heads.state_dict(),
                "optimizer": self.collision_optimizer.state_dict(), "updates": self.collision_updates,
                "target": self.collision_critic_target}

    def load_collision_state(self, state):
        if state.get("target", "td") != self.collision_critic_target:
            return
        self.collision_heads.load_state_dict(state["heads"])
        self.target_collision_heads.load_state_dict(state["target_heads"])
        self.collision_optimizer.load_state_dict(state["optimizer"])
        self.collision_updates = int(state["updates"])

    @torch.no_grad()
    def commands(self, graph, *, explore=False):
        return self.proposal_and_consensus(graph, explore=explore)[0]

    @torch.no_grad()
    def proposal_and_consensus(self, graph, *, explore=False, crossing_exploration=None):
        self.actor.train(explore)
        policy = self.actor(*graph, temperature=0.35, sampling_temperature=self.rollout_router_temperature)
        action = encode_action(policy, 0.1 if explore else 0.0)
        if crossing_exploration is not None:
            action = _crossing_action(policy, action, crossing_exploration, self.rollout_router_temperature)
        return (action, *policy.consensus_inputs)

    @property
    def current_anchor_weight(self):
        fraction = (self.updates - self.anchor_start_update) / 2000
        return self.anchor_weight * max(0., 1. - fraction)

    def update(self, batch, *, actor_batch=None, target_projector=None, actor_projector=None):
        active = batch.graph[-1]
        with torch.no_grad():
            next_policy = self.target_actor(*batch.next_graph)
            next_action = encode_action(next_policy, 0.1)
            if target_projector is not None:
                next_action = project_critic_action(target_projector, next_policy, next_action, batch.next_graph,
                                                    batch.next_physical, batch.unprotected)
            bootstrap = batch.next_graph[-1] & ~batch.terminal
            if self.collision_heads is None or self.collision_critic_target == "observed":
                q1, q2 = self.target_critic(*batch.next_graph, next_action)
            else:
                (q1, risk1), (q2, risk2) = self.target_critic(*batch.next_graph, next_action,
                                                             risk_heads=self.target_collision_heads)
                if batch.collision_return is None:
                    risk_expected = _collision_target(batch.collision_aircraft, bootstrap, risk1.sigmoid(),
                                                      risk2.sigmoid(), self.bootstrap_discount)
                else:
                    future_risk = torch.maximum(risk1.sigmoid(), risk2.sigmoid())
                    risk_expected = batch.collision_return + self.bootstrap_discount * bootstrap * future_risk
            if self.collision_heads is not None and self.collision_critic_target == "observed":
                risk_expected = batch.observed_collision.float()
            expected = batch.reward + self.bootstrap_discount * bootstrap * torch.minimum(q1, q2)

        if self.collision_heads is None:
            first, second = self.critic(*batch.graph, batch.action)
        else:
            (first, logits1), (second, logits2) = self.critic(*batch.graph, batch.action, risk_heads=self.collision_heads)
        error = F.smooth_l1_loss(first, expected, reduction="none") + F.smooth_l1_loss(second, expected, reduction="none")
        critic_loss = _active_mean(error, active)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.collision_heads is None:
            critic_loss.backward()
        else:
            bce = F.binary_cross_entropy_with_logits
            risk_loss = _active_mean(bce(logits1, risk_expected, reduction="none")
                                     + bce(logits2, risk_expected, reduction="none"), active)
            self.collision_optimizer.zero_grad(set_to_none=True)
            (critic_loss + risk_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.collision_heads.parameters(), 5.0)
            self.collision_optimizer.step()
            self.collision_updates += 1
            self.last_collision_loss = risk_loss.detach()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()
        self.updates += 1

        actor_loss = None
        if self.updates % 2 == 0:
            if actor_batch is not None:
                batch = actor_batch
                active = batch.graph[-1]
            self.actor.train()
            self.critic.first.requires_grad_(False)
            collision_head = None
            if self.collision_heads is not None and self.collision_updates >= self.collision_critic_warmup:
                collision_head = self.collision_heads[0].requires_grad_(False)
            try:
                policy = self.actor(*batch.graph, temperature=0.35, detach_coordination=bool(self.coordination_weight))
                action = encode_action(policy)
                critic_action = action
                if actor_projector is not None:
                    executed = project_critic_action(actor_projector, policy, action, batch.graph, batch.physical,
                                                     batch.unprotected)
                    critic_action = projected_action_ste(action, executed)
                objective = self.critic.first(*batch.graph, critic_action, risk_head=collision_head)
                if collision_head is not None:
                    value, logits = objective
                    objective = value - self.collision_critic_weight * logits.sigmoid()
                actor_loss = -_active_mean(objective, active)
                if self.raw_risk_weight:
                    if self.expected_mode_risk_loss:
                        controls = torch.stack((policy.speed[..., 0], policy.altitude, policy.heading), dim=-1)
                        risk = expected_mode_risk(batch.physical, active, policy.logits.softmax(dim=-1), controls)
                    else:
                        risk = lookahead_risk(batch.physical, active, action[..., 4:],
                                              dense=self.dense_risk, overlap=self.overlap_risk)
                    actor_loss = actor_loss + self.raw_risk_weight * _active_mean(risk, active)
                if self.navigation_weight:
                    navigation = _navigation_loss(policy, batch.graph, batch.physical, self.safe_navigation_weight)
                    actor_loss = actor_loss + self.navigation_weight * navigation
                if self.projection_weight:
                    actor_loss = actor_loss + self.projection_weight * _projection_loss(
                        policy, batch.teacher, batch.teacher_mask,
                    )
                if self.coordination_weight:
                    tuning = policy.consensus_inputs
                    geometry = coordination_loss(action[..., 4:], batch.physical, batch.graph[2], active, tuning,
                                                 *self.coordination_rounds)
                    self.last_coordination_loss = geometry.detach()
                    actor_loss = actor_loss + self.coordination_weight * geometry
                if self.teacher_data:
                    example_losses = []
                    example_count = min(8, len(batch.action))
                    for graph, commands, mask in self.teacher_data:
                        selected = torch.randint(len(commands), (example_count,), device=commands.device,
                                                 generator=self.teacher_rng)
                        examples = tuple(value[selected] for value in graph)
                        prediction = self.actor(*examples, deterministic=True)
                        example_losses.append(_teacher_loss(prediction, commands[selected], mask[selected]))
                    actor_loss = actor_loss + self.teacher_weight * torch.stack(example_losses).mean()
                anchor_weight = self.current_anchor_weight
                if self.anchor is not None and anchor_weight > 0.:
                    with torch.no_grad():
                        reference = self.anchor(*batch.graph)
                    anchor_airspaces = None
                    if self.anchor_categories is not None:
                        anchor_airspaces = (batch.category.unsqueeze(-1) == self.anchor_categories).any(dim=-1)
                    actor_loss = actor_loss + anchor_weight * _anchor_loss(
                        policy, reference, batch.graph, anchor_airspaces,
                    )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                self.actor_optimizer.step()
            finally:
                self.critic.first.requires_grad_(True)
                if collision_head is not None:
                    collision_head.requires_grad_(True)
            with torch.no_grad():
                targets = [(self.target_actor, self.actor), (self.target_critic, self.critic)]
                if self.collision_heads is not None:
                    targets.append((self.target_collision_heads, self.collision_heads))
                for target_network, network in targets:
                    torch._foreach_lerp_(tuple(target_network.parameters()), tuple(network.parameters()), self.tau)
        return critic_loss.detach(), None if actor_loss is None else actor_loss.detach()
