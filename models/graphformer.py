"""Edge-biased attention, sparse actor, and twin per-aircraft critics."""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


def mirror_observations(nodes, edges):
    """Reflect the signed heading and lateral features used by the actor."""
    nodes, edges = nodes.clone(), edges.clone()
    nodes[..., [5, 9, 16, 18]] *= -1
    edges[..., [1, 4]] *= -1
    return nodes, edges


class GraphLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.node_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_attention = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2 * width + heads))
        self.output = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff_in = nn.Linear(width, 4 * width)
        self.ff_out = nn.Linear(2 * width, width)
        self.edge_sender = nn.Linear(width, width)
        self.edge_receiver = nn.Linear(width, width)
        self.edge_update = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.edge_norm = nn.LayerNorm(width)

    def forward(self, nodes, edges, adjacency, active):
        shape = (*nodes.shape[:-1], self.heads, self.head_width)
        query, key, value = (x.reshape(shape) for x in self.qkv(self.node_norm(nodes)).chunk(3, dim=-1))
        edge_keys, edge_values, bias = self.edge_attention(edges).split(
            (nodes.shape[-1], nodes.shape[-1], self.heads), dim=-1,
        )
        edge_keys = edge_keys.unflatten(-1, (self.heads, self.head_width))
        edge_values = edge_values.unflatten(-1, (self.heads, self.head_width))
        keys = key.unsqueeze(-4) + edge_keys
        values = value.unsqueeze(-4) + edge_values
        logits = (query.unsqueeze(-3) * keys).sum(dim=-1) / sqrt(self.head_width) + bias
        diagonal = torch.eye(nodes.shape[-2], device=nodes.device, dtype=torch.bool)
        attend = adjacency | diagonal
        weights = logits.masked_fill(~attend.unsqueeze(-1), -torch.inf).softmax(dim=-2)
        message = (weights.unsqueeze(-1) * values).sum(dim=-3).flatten(-2)
        nodes = nodes + self.output(message)
        gate, value = self.ff_in(self.ff_norm(nodes)).chunk(2, dim=-1)
        nodes = (nodes + self.ff_out(F.silu(gate) * value)).masked_fill(~active.unsqueeze(-1), 0.0)
        messages = self.edge_sender(nodes).unsqueeze(-2) + self.edge_receiver(nodes).unsqueeze(-3)
        edges = self.edge_norm(edges + self.edge_update(edges + messages))
        return nodes, edges.masked_fill(~attend.unsqueeze(-1), 0.0)


class GraphFormer(nn.Module):
    def __init__(self, node_features=24, edge_features=18, width=256, heads=8, layers=3):
        super().__init__()
        if width % heads:
            raise ValueError("GraphFormer width must be divisible by its number of heads")
        self.embed_nodes = nn.Linear(node_features, width)
        self.embed_edges = nn.Linear(edge_features, width)
        self.layers = nn.ModuleList(GraphLayer(width, heads) for _ in range(layers))

    def forward(self, node_features, edge_features, adjacency, active):
        adjacency = adjacency & active.unsqueeze(-1) & active.unsqueeze(-2)
        nodes = self.embed_nodes(node_features)
        edges = self.embed_edges(edge_features).masked_fill(~adjacency.unsqueeze(-1), 0.0)
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, adjacency, active)
        return nodes, edges


@dataclass
class Policy:
    logits: torch.Tensor
    mode: torch.Tensor
    speed: torch.Tensor
    altitude: torch.Tensor
    heading: torch.Tensor
    dual: torch.Tensor
    rho: torch.Tensor
    mobility: torch.Tensor
    altitude_dual: torch.Tensor
    altitude_rho: torch.Tensor
    _speed_warm_dual: torch.Tensor | None = None
    _altitude_warm_dual: torch.Tensor | None = None

    @property
    def consensus_inputs(self):
        # ADMM solves scalar speed and altitude increments in aircraft-slot
        # order, so its initial duals have even reflection parity.
        speed_dual = self.dual if self._speed_warm_dual is None else self._speed_warm_dual
        altitude_dual = self.altitude_dual if self._altitude_warm_dual is None else self._altitude_warm_dual
        return speed_dual, self.rho, self.mobility, altitude_dual, self.altitude_rho


class Actor(nn.Module):
    def __init__(self, width=256, *, equivariant_actor=False, **architecture):
        super().__init__()
        self.equivariant_actor = equivariant_actor
        self.encoder = GraphFormer(width=width, **architecture)
        self.router = nn.Linear(width, 4)
        self.speed = nn.Linear(width, 2)
        self.altitude = nn.Linear(width, 3)
        self.heading = nn.Linear(width, 1)
        self.willingness = nn.Linear(width, 1)
        self.consensus = nn.Linear(width, 3)
        self.altitude_consensus = nn.Linear(width, 3)
        for head in (self.consensus, self.altitude_consensus):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            nn.init.constant_(head.bias[2:3], 7.9)

    @staticmethod
    def _warmstart(head, edges):
        dual, penalty = head(edges).split((2, 1), dim=-1)
        dual = 0.5 * (dual + dual.transpose(-2, -3).flip(-1))
        rho = F.softplus(0.5 * (penalty + penalty.transpose(-2, -3))).squeeze(-1) + 0.1
        return dual, rho

    def forward(self, node_features, edge_features, adjacency, active, temperature=1.0, *, deterministic=False,
                sampling_temperature=1.0, detach_coordination=False):
        if self.equivariant_actor:
            mirror_nodes, mirror_edges = mirror_observations(node_features, edge_features)
            node_features = torch.cat((node_features, mirror_nodes))
            edge_features = torch.cat((edge_features, mirror_edges))
            adjacency = torch.cat((adjacency, adjacency))
            active = torch.cat((active, active))
        nodes, edges = self.encoder(node_features, edge_features, adjacency, active)
        logits = self.router(nodes)
        speed = self.speed(nodes).tanh()
        altitude = self.altitude(nodes).softmax(-1) @ nodes.new_tensor([-1.0, 0.0, 1.0])
        raw_heading = self.heading(nodes).squeeze(-1).tanh()
        coordination_edges = edges.detach() if detach_coordination else edges
        coordination_nodes = nodes.detach() if detach_coordination else nodes
        dual, rho = self._warmstart(self.consensus, coordination_edges)
        altitude_dual, altitude_rho = self._warmstart(self.altitude_consensus, coordination_edges)
        willingness = F.softplus(self.willingness(coordination_nodes).squeeze(-1)) + 1e-3
        speed_warm_dual = altitude_warm_dual = None
        if self.equivariant_actor:
            def combine(value, sign):
                original, mirror = value.chunk(2, dim=0)
                return (original + sign * mirror) * 0.5

            logits, speed, altitude, rho, willingness, altitude_rho = (
                combine(value, 1) for value in (logits, speed, altitude, rho, willingness, altitude_rho)
            )
            speed_warm_dual = combine(dual, 1)
            altitude_warm_dual = combine(altitude_dual, 1)
            raw_heading, dual, altitude_dual = (combine(value, -1) for value in (raw_heading, dual, altitude_dual))
        deadband_heading = torch.where(raw_heading.abs() >= 0.05, raw_heading, 0.0)
        heading = raw_heading + (deadband_heading - raw_heading).detach()
        if self.training and not deterministic:
            sampling_logits = logits if sampling_temperature == 1.0 else logits / sampling_temperature
            mode = F.gumbel_softmax(sampling_logits, tau=temperature, hard=True)
        else:
            mode = F.one_hot(logits.argmax(dim=-1), num_classes=4).to(logits.dtype)
        return Policy(logits, mode, speed, altitude, heading, dual, rho, willingness, altitude_dual, altitude_rho,
                      speed_warm_dual, altitude_warm_dual)


class Critic(nn.Module):
    def __init__(self, width=256, action_features=8, **architecture):
        super().__init__()
        self.encoder = GraphFormer(node_features=24 + action_features, width=width, **architecture)
        self.value = nn.Sequential(nn.Linear(width + action_features, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, node_features, edge_features, adjacency, active, action, *, risk_head=None):
        nodes, _ = self.encoder(torch.cat((node_features, action), dim=-1), edge_features, adjacency, active)
        features = torch.cat((nodes, action), dim=-1)
        value = self.value(features).squeeze(-1).masked_fill(~active, 0.0)
        return value if risk_head is None else (value, risk_head(features).squeeze(-1))


class TwinCritic(nn.Module):
    def __init__(self, **architecture):
        super().__init__()
        self.first = Critic(**architecture)
        self.second = Critic(**architecture)

    def forward(self, node_features, edge_features, adjacency, active, action, *, risk_heads=(None, None)):
        inputs = (node_features, edge_features, adjacency, active, action)
        return self.first(*inputs, risk_head=risk_heads[0]), self.second(*inputs, risk_head=risk_heads[1])
