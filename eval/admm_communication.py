"""Seeded loss and control-tick latency for bilateral ADMM exchanges."""

from dataclasses import dataclass, field
from math import isfinite

import torch


PACKET_SEED_OFFSET = 91453
_NO_MESSAGE_AGE = torch.iinfo(torch.long).max


class PairExchange:
    """A missed exchange leaves the edge's preceding consensus and dual intact."""

    def __init__(self, loss, first_seed, worlds, device):
        if not isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError("ADMM packet loss must be between zero and one")
        self.loss = loss
        self.live = torch.ones(worlds, dtype=torch.bool, device=device)
        self.attempted = torch.zeros(worlds, dtype=torch.long, device=device)
        self.dropped = torch.zeros_like(self.attempted)
        self.coordination_ticks = torch.zeros_like(self.attempted)
        self._solves = torch.zeros_like(self.attempted)
        self._residual_sum = torch.zeros(worlds, 2, device=device)
        self._max_residual = torch.zeros_like(self._residual_sum)
        self._rngs = tuple(torch.Generator(device=device).manual_seed(seed + PACKET_SEED_OFFSET)
                           for seed in range(first_seed, first_seed + worlds))

    def __call__(self, participating):
        if self.loss == 0:
            delivered = participating
        elif self.loss == 1:
            delivered = torch.zeros_like(participating)
        else:
            draws = torch.stack([torch.rand(participating.shape[-2:], device=participating.device, generator=rng)
                                 for rng in self._rngs])
            delivered = participating & (draws >= self.loss)
        self.attempted.add_(participating.sum(dim=(-2, -1)) * self.live)
        self.dropped.add_((participating & ~delivered).sum(dim=(-2, -1)) * self.live)
        return delivered

    def observe_residuals(self, speed, altitude):
        residual = torch.stack((speed, altitude), dim=-1).masked_fill(~self.live.unsqueeze(-1), 0.0)
        self._solves.add_(self.live)
        self._residual_sum.add_(residual)
        self._max_residual.copy_(torch.maximum(self._max_residual, residual))

    def report(self):
        rows = []
        residuals = zip(self._solves.tolist(), self._residual_sum.tolist(), self._max_residual.tolist(), strict=True)
        for attempted, dropped, ticks, (solves, summed, peak) in zip(
                self.attempted.tolist(), self.dropped.tolist(), self.coordination_ticks.tolist(), residuals, strict=True):
            rows.append(dict(attempted_pair_exchanges=attempted, dropped_pair_exchanges=dropped,
                             delivered_pair_exchanges=attempted - dropped,
                             observed_loss=dropped / attempted if attempted else None,
                             coordination_ticks=ticks,
                             projection_solves=solves, speed_primal_residual_mean_mps=summed[0] / solves if solves else None,
                             altitude_primal_residual_mean_m=summed[1] / solves if solves else None,
                             speed_primal_residual_max_mps=peak[0] if solves else None,
                             altitude_primal_residual_max_m=peak[1] if solves else None))
        return rows


@dataclass
class _Packet:
    valid: torch.Tensor
    consensus: torch.Tensor
    dual: torch.Tensor
    sent: torch.Tensor

    def with_valid(self, valid):
        return _Packet(valid, self.consensus, self.dual, self.sent)

    @staticmethod
    def merge(previous, latest):
        if previous is None:
            return latest
        selected = latest.valid.unsqueeze(-1)
        return _Packet(previous.valid | latest.valid, torch.where(selected, latest.consensus, previous.consensus),
                       torch.where(selected, latest.dual, previous.dual),
                       torch.where(latest.valid, latest.sent, previous.sent))


@dataclass
class _Mailbox:
    pending: dict = field(default_factory=dict)
    latest: _Packet | None = None


class _DelayedLayer:
    def __init__(self, exchange, mailbox, origin):
        self.exchange = exchange
        self.mailbox = mailbox
        self.origin = origin

    def receive(self, edges, consensus, dual):
        packet = self.mailbox.latest
        if packet is None:
            return consensus, dual
        valid = edges & packet.valid
        age = (self.exchange.tick - packet.sent).masked_fill(~valid, 0)
        self.exchange.used.add_(valid.sum(dim=(-2, -1)))
        self.exchange.age_sum.add_(age.sum(dim=(-2, -1)))
        minimum = age.masked_fill(~valid, _NO_MESSAGE_AGE).amin(dim=(-2, -1))
        self.exchange.earliest_age.copy_(torch.minimum(self.exchange.earliest_age, minimum))
        self.exchange.latest_age.copy_(torch.maximum(self.exchange.latest_age, age.amax(dim=(-2, -1))))
        values = packet.consensus if self.origin is None else packet.consensus - self.origin
        return (torch.where(valid.unsqueeze(-1), values, consensus),
                torch.where(valid.unsqueeze(-1), packet.dual, dual))

    def send(self, edges, consensus, dual):
        owner = self.exchange
        accepted = owner(edges & owner.alive_pairs)
        arrival = owner.tick + owner.delay
        values = consensus if self.origin is None else consensus + self.origin
        packet = _Packet(accepted, values, dual, torch.full_like(accepted, owner.tick, dtype=torch.long))
        # Several exchanges of one edge may share an arrival tick. Retain the last accepted state.
        self.mailbox.pending[arrival] = _Packet.merge(self.mailbox.pending.get(arrival), packet)
        count = accepted.sum(dim=(-2, -1))
        owner.pending_counts[arrival] = owner.pending_counts.get(arrival, 0) + count


class DelayedPairExchange(PairExchange):
    """Queue messages by absolute control tick, independently for each solver tier."""

    def __init__(self, loss, first_seed, worlds, device, delay):
        super().__init__(loss, first_seed, worlds, device)
        if delay not in (1, 2):
            raise ValueError("Delayed communication requires one or two control ticks")
        self.delay = delay
        self.tick = -1
        self.alive_pairs = None
        self.mailboxes = {}
        self.pending_counts = {}
        self.arrived = torch.zeros_like(self.attempted)
        self.used = torch.zeros_like(self.attempted)
        self.age_sum = torch.zeros_like(self.attempted)
        self.earliest_age = torch.full_like(self.attempted, _NO_MESSAGE_AGE)
        self.latest_age = torch.zeros_like(self.attempted)

    def begin_step(self, tick, active, live):
        if tick != self.tick + 1:
            raise ValueError("Delayed exchange must advance on every control tick")
        self.tick = tick
        self.live = live
        alive = active & live.unsqueeze(-1)
        self.alive_pairs = alive.unsqueeze(-1) & alive.unsqueeze(-2)
        count = self.pending_counts.pop(tick, None)
        if count is not None:
            self.arrived.add_(count * live)
        for mailbox in self.mailboxes.values():
            if mailbox.latest is not None:
                mailbox.latest = mailbox.latest.with_valid(mailbox.latest.valid & self.alive_pairs)
            packet = mailbox.pending.pop(tick, None)
            if packet is not None:
                visible = packet.with_valid(packet.valid & self.alive_pairs)
                mailbox.latest = _Packet.merge(mailbox.latest, visible)
            for arrival, pending in mailbox.pending.items():
                mailbox.pending[arrival] = pending.with_valid(pending.valid & self.alive_pairs)

    def layer(self, name, origin=None):
        if self.alive_pairs is None:
            raise ValueError("Advance the communication clock before solving")
        if origin is not None:
            origin = torch.stack(torch.broadcast_tensors(origin.unsqueeze(-1), origin.unsqueeze(-2)), dim=-1)
        return _DelayedLayer(self, self.mailboxes.setdefault(name, _Mailbox()), origin)

    def report(self):
        rows = super().report()
        for row, arrived, used, age, min_age, max_age in zip(
                rows, self.arrived.tolist(), self.used.tolist(), self.age_sum.tolist(),
                self.earliest_age.tolist(), self.latest_age.tolist(), strict=True):
            if used and min_age < self.delay:
                raise ValueError("A bilateral state was used before its scheduled arrival")
            accepted = row["delivered_pair_exchanges"]
            row.update(delivered_pair_exchanges=arrived, in_flight_pair_exchanges=accepted-arrived,
                       used_delayed_pair_states=used, delay_steps=self.delay,
                       mean_used_message_age_steps=age / used if used else None,
                       min_used_message_age_steps=min_age if used else None,
                       max_used_message_age_steps=max_age if used else None)
        return rows
