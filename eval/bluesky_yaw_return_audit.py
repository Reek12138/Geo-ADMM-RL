"""Recompute Plan B's Stage 1 return targets along native evaluation trajectories."""

import json

import torch

from envs.bluesky_safety import lookahead_risk, protected_command
from envs.observations import LAYER
from eval.bluesky_pair import _checksum, _load
from models.graphformer import TwinCritic
from train.bluesky_td3 import _shaping
from train.td3 import encode_action, encode_executed_action

class YawReturnAudit:
    def __init__(self, checkpoint, configuration):
        config = json.loads(configuration.read_text())
        if (config["plan"] != "b" or config["cross_track_weight"] or config["unprotected_rollout_fraction"]
                or not config["multi_step_return"] or not config["executed_action_critic"]
                or not config["projected_target_action"]):
            raise ValueError("Return audit expects the guarded, projected Stage 1 training configuration")
        self.config = config
        self.metadata = dict(config_sha256=_checksum(configuration), horizon=config["return_window"],
                             discount=.99, target_noise_seed_start=38601, target_views=["native", "training"])
        self.target_actor = _load(checkpoint, torch.device("cpu"))
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.target_actor.load_state_dict(checkpoint_data["target_actor"])
        self.target_critic = TwinCritic(width=checkpoint_data["width"], action_features=7).eval()
        self.target_critic.load_state_dict(checkpoint_data["target_critic"])
        self.critic = TwinCritic(width=checkpoint_data["width"], action_features=7).eval()
        self.critic.load_state_dict(checkpoint_data["critic"])

    def _training_graph(self, graph, physical):
        if self.target_actor.observation_profile.selected_altitude_observation:
            return graph
        node = graph[0].clone()
        node[..., 3] = ((physical[0][..., 2] - physical[-1]) / LAYER).masked_fill(~graph[-1], 0)
        return (node, *graph[1:])

    def _target(self, graph, physical, seed):
        active = graph[-1]
        config = self.config
        policy = self.target_actor(*graph)
        torch.manual_seed(seed)
        action = encode_action(policy, .1)
        tuning = policy.consensus_inputs
        commands, _ = protected_command(
            action, physical, graph[2], active, tuning, config["admm_k"], config["admm_fallback"],
            by_component=config["component_projection"], vertical_heading=config["vertical_heading"],
            directional_heading=config["directional_heading"], extra_rounds=config["dense_extra_rounds"],
            batched_layer_search=config["batched_layer_search"], joint_layer_search=config["joint_layer_search"],
            joint_layer_reserve=config["joint_layer_reserve"],
        )
        q1, q2 = self.target_critic(*graph, encode_executed_action(action, commands))
        return torch.minimum(q1, q2)

    def values(self, graph, physical, decision):
        seed = self.metadata["target_noise_seed_start"] + decision
        return {name: self._target(observation, physical, seed)[0] for name, observation in
                (("native", graph), ("training", self._training_graph(graph, physical)))}

    def first_action_comparison(self, graph, physical, proposal, projected, guided, decision):
        changed = graph[-1] & ((guided - projected).abs().amax(dim=-1) > 1e-5)
        active = graph[-1]
        results = {}
        for name, view in (("native", graph), ("training", self._training_graph(graph, physical))):
            before = self.critic(*view, encode_executed_action(proposal, projected))
            after = self.critic(*view, encode_executed_action(proposal, guided))
            first = after[0] - before[0]
            minimum = torch.minimum(*after) - torch.minimum(*before)
            results[name] = dict(first_critic_changed_slots=first[changed].tolist(),
                                 minimum_critic_changed_slots=minimum[changed].tolist(),
                                 first_critic_all_active_delta=float(first[active].sum()),
                                 minimum_critic_all_active_delta=float(minimum[active].sum()))
        return dict(decision=decision, changed_slots=changed[0].nonzero().flatten().tolist(), views=results)

    def rewards(self, graph, future, future_physical, action, executed, native_rewards, done, information):
        config = self.config
        active = graph[-1]
        native_rewards = torch.as_tensor(native_rewards)
        completed = torch.as_tensor(done).unsqueeze(-1)
        arrivals = active & (native_rewards == 200)
        loss = active & (native_rewards == -50)
        timeout = active & completed & ~(arrivals | loss)
        if not torch.equal(timeout.sum(dim=-1), torch.as_tensor(information["timeouts"])):
            raise ValueError("Native aircraft identities do not match timeout counts")
        terminal = ~future[-1] | completed
        reward = -.1 * executed.abs().sum(dim=-1) + 200.0 * arrivals - 50.0 * loss - 20.0 * timeout
        projection_cost = -.05 * (executed - action[..., 4:]).square().sum(dim=-1) * active
        shaping = _shaping(graph, future, action, terminal) if config["shaped_reward"] else torch.zeros_like(reward)
        risk = -config["lookahead_weight"] * lookahead_risk(
            future_physical, future[-1], dense=config["dense_risk"], overlap=config["overlap_risk"],
        ).masked_fill(terminal, 0)
        return dict(simulator=reward[0], correction=projection_cost[0], shaping=shaping[0], risk=risk[0],
                    total=(reward + projection_cost + shaping + risk)[0])
