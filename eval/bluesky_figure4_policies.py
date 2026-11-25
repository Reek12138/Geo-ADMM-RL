"""Run Geo, SARL and the original KSS on an identical Figure 4 release manifest."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys
from types import SimpleNamespace

import bluesky as bs
from bluesky.rl import BatchedBlueSkyConfig, set_commands_direct
import numpy as np
import torch

from envs.bluesky_safety import protected_command
from eval.admm_gap_mpc import plan_gap_maneuver
from eval.bluesky_figure4_replay import Figure4Replay
from eval.bluesky_pair import _checksum, _communication_interval, _load
from eval.stage_acceptance import MPC_COMMUNICATION
from train.td3 import encode_action


POLICIES = ("hold-kss", "sarl", "sarl-kss", "geo", "geo-raw", "geo-kss")
OUTCOMES = ("arrivals", "collisions", "exits", "timeouts")


class SarlReplayPolicy:
    """Use the archived controller methods against the currently running native traffic."""

    def __init__(self, args, env, settings):
        if not args.sarl_checkout or not args.sarl_scripts or not args.sarl_config:
            raise ValueError("Historical policies require the SARL checkout, scripts, and resolved settings")
        sys.path[:0] = [str(args.sarl_checkout), str(args.sarl_scripts)]
        upstream = importlib.import_module("plugins.case_step3_SARL")
        alignment = importlib.import_module("paper_alignment")
        self.ctde = importlib.import_module("paper_ctde")
        self.source_hashes = {module.__name__: _checksum(Path(module.__file__))
                              for module in (upstream, alignment, self.ctde)}
        upstream.traf = bs.traf
        self.apply = alignment.apply_sparse_commands
        self.math = upstream.MathUtils
        self.env = env
        self.use_kss = args.policy in ("hold-kss", "sarl-kss", "geo-kss")

        class Controller(self.ctde.JointControl, upstream.ACREnvironmentController):
            def __init__(self):
                # Reuse the original forecast and command methods without starting its learner or spawner.
                self.simulator = bs
                self.rejoin_after_shield = set()
                self.route_reacquisitions = 0

            def _write_scn_log(self, command):
                pass

        controller = self.controller = Controller()
        controller.cfg = SimpleNamespace(
            num_intruders=settings["neighbor_limit"], max_speed_delta=4.5, max_alt_delta=300., max_hdg_delta=5.,
            limits={"speed": (200., 300.), "alt": (20000., 21000.)},
            reward_params=settings["reward_params"], recover_kss_route=self.use_kss,
            recover_policy_route=settings["policy_route_recovery"],
        )
        controller.normalizer = upstream.AircraftStateNormalizer(num_intruders=settings["neighbor_limit"])
        # Use the same origin and goal for each original route ID, regardless of which aircraft remain alive.
        controller.routes = {}
        for flight in env.flights:
            route = (flight.latitude, flight.longitude, flight.goal_latitude, flight.goal_longitude)
            previous = controller.routes.setdefault(flight.route_index, route)
            if previous != route:
                raise ValueError(f"Figure 4 route {flight.route_index} changes endpoints")
        self.settings = settings
        self.actor = None
        if args.policy.startswith("sarl"):
            checkpoint = args.checkpoint or Path(settings["actor_checkpoint"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.actor = alignment.SparseActor(20, 15, state["router.weight"].shape[-1]).eval()
            self.actor.load_state_dict(state)
            self.checkpoint = checkpoint

    def command(self, raw=None):
        ids = tuple(bs.traf.id)
        flights = [self.env.flights[slot] for slot in self.env.ac_slot]
        controller = self.controller
        controller.route_assignments = dict(zip(ids, (flight.route_index for flight in flights), strict=True))
        controller.assigned_levels = dict(zip(ids, (flight.altitude_ft for flight in flights), strict=True))
        controller.rejoin_after_shield.intersection_update(ids)
        neighbors = controller._get_neighbors_info_with_id()
        for index, acid in enumerate(ids):
            neighbors[acid] = [peer for peer in neighbors[acid] if self.math.calculate_haversine_distance(
                bs.traf.lat[index], bs.traf.lon[index], peer[0], peer[1]) <= self.settings["detection_range_km"]]
        if self.actor is not None:
            states = controller._collect_gat_states(neighbors)
            if self.settings["physics_neighbors"]:
                self.ctde.populate_physics_neighbors(states, neighbors, bs.traf,
                                                     controller.cfg.reward_params["lookahead_time"])
            if self.settings["navigation_state"]:
                self.ctde.populate_navigation(states, bs.traf, controller.routes, controller.route_assignments,
                                               controller.cfg.max_hdg_delta, False)
            obs = self.ctde.Observation.collect(ids, states, controller.cfg.num_intruders)
            raw = self.actor.route(*(torch.from_numpy(x) for x in (obs.ego, obs.neighbors, obs.mask)))[0].numpy()
        elif raw is None:
            raw = np.zeros((len(ids), 3), dtype=np.float32)
        if not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
            raise ValueError("The policy proposed an invalid command")
        applied = np.empty_like(raw)
        interventions = 0
        for offset, (acid, proposal) in enumerate(zip(ids, raw, strict=True)):
            executed, level = (controller._apply_phase2_shield(acid, proposal, neighbors[acid])
                               if self.use_kss else (proposal, 0))
            if self.use_kss:
                controller._resume_shielded_route(acid, proposal, executed, level, neighbors[acid])
            applied[offset] = self.apply(controller, bs.traf, bs.stack, acid, executed)
            interventions += int(level > 0)
        return applied, interventions


@torch.inference_mode()
def run(args):
    layer_options = dict(early_layer_search=getattr(args, "early_layer_search", False),
                         exact_layer_forecast=getattr(args, "exact_layer_forecast", False),
                         batched_layer_margin=getattr(args, "batched_layer_margin", 0.))
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    device = torch.device(args.device)
    if device.type == "cuda" and importlib.util.find_spec("torch_ppu") is not None:
        importlib.import_module("torch_ppu")
    geo = args.policy.startswith("geo")
    mpc = getattr(args, "mpc_gap", False)
    if mpc and args.policy != "geo":
        raise ValueError("MPC gap control requires the projected Geo policy")
    actor = _load(args.checkpoint, device) if geo else None
    args.workdir.mkdir()
    if args.navdata:
        cache = args.workdir / "cache"
        cache.mkdir()
        (cache / "navdata.p").hardlink_to(args.navdata)
    config = BatchedBlueSkyConfig(num_envs=1, aircraft_per_env=len(manifest["flights"]), sim_dt=.05, control_dt=6.,
                                 max_steps=2000, hdg_deadband_deg=.25, seed=manifest["seed"])
    interval = _communication_interval(MPC_COMMUNICATION["admm_communication_period"] if mpc else None,
                                       config.control_dt)
    env = Figure4Replay(config, workdir=args.workdir, manifest=manifest,
                        observation_profile=actor.observation_profile if geo else None)
    graph = env.reset()
    historical = None
    if not geo or args.policy == "geo-kss":
        settings = json.loads(args.sarl_config.read_text())
        if settings["case"] != manifest["case"] or not settings["sparse_controls"]:
            raise ValueError("The matched SARL settings must use this Figure 4 case and sparse native commands")
        historical = SarlReplayPolicy(args, env, settings)
    zero = np.zeros((1, config.aircraft_per_env, 3), dtype=np.float32)
    totals = dict.fromkeys(OUTCOMES, 0)
    minimum = {name: np.full(config.aircraft_per_env, np.inf) for name in ("min_horizontal_km", "min_3d_km")}
    absolute_cost = np.zeros(3)
    visits = shielded = 0
    gap_ticks = gap_adjusted = 0
    for decision in range(config.max_steps):
        if bs.traf.ntraf:
            visits += bs.traf.ntraf
            if geo:
                current = tuple(value.to(device) for value in graph)
                prediction = actor(*current)
                proposal = encode_action(prediction)
                commands = proposal[..., 4:]
                level = 0
                if args.policy == "geo":
                    physical = tuple(value.to(device) for value in env.physical)
                    if decision % interval:
                        commands, _, _ = plan_gap_maneuver(physical, current[-1], commands)
                        gap_ticks += 1
                        gap_adjusted += int(commands.abs().amax(dim=-1).gt(1e-5).sum())
                    else:
                        tuning = prediction.consensus_inputs
                        commands, _ = protected_command(proposal, physical, current[2], current[-1], tuning, 2, 0,
                                                        directional_heading=True, extra_rounds=2, batched_layer_search=True,
                                                        joint_layer_search=True, joint_layer_reserve=10., **layer_options)
                        level = int(((commands - proposal[..., 4:]).square().sum(-1).gt(1e-5) & current[-1]).sum())
                actions = commands.cpu().numpy()[0, env.ac_slot]
                if historical is not None:
                    applied, level = historical.command(actions)
                else:
                    scales = np.array([config.max_speed_delta_kts, config.max_alt_delta_ft, config.max_hdg_delta_deg])
                    delivered = set_commands_direct(np.arange(bs.traf.ntraf), *(actions * scales).T,
                                                    hdg_deadband_deg=config.hdg_deadband_deg)
                    applied = np.stack([delivered[k] for k in ("applied_cas_kts", "applied_alt_ft", "applied_hdg_deg")],
                                       axis=-1) / scales
            else:
                applied, level = historical.command()
            absolute_cost += np.abs(applied).sum(axis=0)
            shielded += level
        graph, _, done, info = env.step_native_audited(zero)
        for key in OUTCOMES:
            totals[key] += int(info[key][0])
        for key in minimum:
            np.minimum(minimum[key], info[key][0], out=minimum[key])
        if done[0]:
            break
    aircraft = config.aircraft_per_env
    if env.cursor != aircraft or sum(totals.values()) != aircraft:
        raise RuntimeError("A common-flow episode did not account for every release and outcome")
    checkpoint = args.checkpoint
    if historical is not None and historical.actor is not None:
        checkpoint = historical.checkpoint
    result = dict(case=manifest["case"], seed=manifest["seed"], policy=args.policy, sim_dt=.05, control_dt=6.,
                  layer_safety_options=layer_options, manifest_sha256=_checksum(args.manifest), checkpoint=None if checkpoint is None else str(checkpoint),
                  checkpoint_sha256=None if checkpoint is None else _checksum(checkpoint), departures=env.departures,
                  historical_config_sha256=None if historical is None else _checksum(args.sarl_config),
                  historical_source_sha256=None if historical is None else historical.source_hashes,
                  aircraft=aircraft, outcomes=totals, normalized_axis_cost_per_aircraft=(absolute_cost / aircraft).tolist(),
                  action_cost_per_aircraft=float((absolute_cost @ np.array([1., 2., 4.])) / aircraft),
                  control_steps=decision+1, aircraft_decisions=visits, interventions=shielded,
                  collision_h_km=config.collision_h_km, collision_v_ft=config.collision_v_ft,
                  arrival_dist_km=config.arrival_dist_km, max_xtk_km=config.max_xtk_km)
    if mpc:
        result["communication"] = MPC_COMMUNICATION | dict(gap_ticks=gap_ticks, adjusted_aircraft=gap_adjusted)
    for key, values in minimum.items():
        samples = values[np.isfinite(values)].tolist()
        result[key] = min(samples, default=None)
        result[f"mean_{key}"] = mean(samples) if samples else None
    pending = args.output.with_suffix(".tmp")
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(args.output)
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "workdir", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    for name in ("checkpoint", "sarl-checkout", "sarl-scripts", "sarl-config", "navdata"):
        cli.add_argument(f"--{name}", type=Path)
    cli.add_argument("--policy", choices=POLICIES, required=True)
    cli.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    cli.add_argument("--early-layer-search", action="store_true")
    cli.add_argument("--exact-layer-forecast", action="store_true")
    cli.add_argument("--batched-layer-margin", type=float, default=0.)
    cli.add_argument("--mpc-gap", action="store_true")
    options = cli.parse_args()
    if options.policy.startswith("geo") and options.checkpoint is None:
        cli.error("Geo policies need --checkpoint")
    if options.policy not in ("geo", "geo-raw") and not all(
            (options.sarl_checkout, options.sarl_scripts, options.sarl_config)):
        cli.error("SARL or KSS policies need historical controller inputs")
    run(options)
