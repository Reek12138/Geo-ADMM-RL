"""Extract archived Figure 4 traffic for a shared native replay."""

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np


CASES = (1, 2, 3)
ARMS = ("hold_kss", "hold_without_kss", "actor_kss", "actor_without_kss")
FRAME_SECONDS = Decimal(".05")


@dataclass(frozen=True)
class Flight:
    callsign: str
    release_frame: int
    latitude: float
    longitude: float
    heading: float
    altitude_ft: float
    cas_kts: float
    goal_latitude: float
    goal_longitude: float
    route_index: int


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_frame(timestamp):
    hours, minutes, seconds = map(Decimal, timestamp.split(":"))
    ticks = (hours * 3600 + minutes * 60 + seconds) / FRAME_SECONDS
    if ticks != ticks.to_integral_value() or ticks < 0:
        raise ValueError(f"Off-grid timestamp {timestamp}")
    return int(ticks)


def read_scenario(path, routes):
    created, goals = {}, {}
    for line in path.read_text().splitlines():
        timestamp, _, command = line.partition(">")
        operation, _, args = command.strip().partition(" ")
        if operation == "CRE":
            callsign, aircraft_type, *raw = (value.strip() for value in args.split(","))
            if aircraft_type != "B737" or callsign in created or len(raw) != 5:
                raise ValueError(f"{path}: ambiguous B737 departure")
            position = tuple(map(float, raw))
            if not np.isfinite(position).all() or position[4] <= 0 or not 0 <= position[2] < 360:
                raise ValueError(f"{path}: invalid departure state for {callsign}")
            created[callsign] = release_frame(timestamp), position
        if operation == "ADDWPT":
            callsign, _, raw = args.strip().partition(" ")
            if callsign not in created or callsign in goals or release_frame(timestamp) != created[callsign][0]:
                raise ValueError(f"{path}: route for {callsign} is not uniquely defined at departure")
            goal = tuple(map(float, raw.split(",")))
            if len(goal) != 2:
                raise ValueError(f"{path}: expected one geographic waypoint")
            goals[callsign] = goal
    if not created or created.keys() != goals.keys():
        raise ValueError(f"{path}: departures and destinations do not match")
    flights = []
    for callsign, (frame, position) in created.items():
        endpoints = (*position[:2], *goals[callsign])
        matched = np.flatnonzero(np.isclose(routes[:, :4], endpoints, rtol=0, atol=1e-7).all(axis=1))
        if len(matched) != 1:
            raise ValueError(f"{path}: {callsign} does not match one Figure 4 route")
        flights.append(Flight(callsign, frame, *position, *goals[callsign], int(matched[0])))
    if any(b.release_frame < a.release_frame for a, b in zip(flights, flights[1:])):
        raise ValueError(f"{path}: departures are out of order")
    return flights


def compare_flows(args):
    protocol_file = args.archive / "protocol.json"
    protocol = json.loads(protocol_file.read_text())
    if protocol["evaluation_sim_dt"] != float(FRAME_SECONDS) or tuple(protocol["cases"]) != CASES:
        raise ValueError("Expected the three native Figure 4 cases")
    result = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        route_path = args.routes / f"figure4_case{case}.npy"
        routes_digest = sha256(route_path)
        if routes_digest != protocol["routes_sha256"][str(case)]:
            raise ValueError(f"Case {case} route digest differs from the archive")
        routes = np.load(route_path, allow_pickle=False)
        if routes.shape != ((3, 4, 6)[case - 1], 5) or not np.isfinite(routes).all():
            raise ValueError(f"Invalid Figure 4 route table for case {case}")
        result[case] = {}
        for seed in args.seeds:
            flows, sources = {}, {}
            for arm in ARMS:
                evaluation = args.archive / "evaluation" / f"case{case}" / arm / str(seed)
                config_path = evaluation / "config_resolved.json"
                config = json.loads(config_path.read_text())
                if (config["case"], config["seed"], config["sim_dt_seconds"], config["decision_interval_seconds"]
                    ) != (case, seed, .05, 6):
                    raise ValueError(f"Simulator or route configuration differs in {config_path}")
                override = config["route_file_override_sha256"]
                if override != routes_digest and not (override is None and config["route_directions"] == "upstream"):
                    raise ValueError(f"Unexpected route override in {config_path}")
                log = evaluation / "output/result/SARL/eval_scn/Ep1.scn"
                flows[arm] = read_scenario(log, routes)
                if {flight.route_index for flight in flows[arm]} != set(range(len(routes))):
                    raise ValueError(f"{log}: a Figure 4 route was not exercised")
                sources[arm] = dict(path=str(log), sha256=sha256(log), config_sha256=sha256(config_path),
                                    route_override_sha256=override)
            reference = flows["hold_kss"]
            matching = {arm: flights == reference for arm, flights in flows.items()}
            destination = args.output / f"case{case}-{seed}.json"
            manifest = dict(case=case, seed=seed, sim_dt_seconds=.05, control_dt_seconds=6,
                            reference_arm="hold_kss", route_sha256=routes_digest,
                            source=sources["hold_kss"], logged_timestamp_resolution_seconds=1,
                            flights=[asdict(flight) for flight in reference])
            destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            result[case][seed] = dict(path=str(destination), sha256=sha256(destination),
                                     matches_hold_kss=matching, flight_counts={arm: len(v) for arm, v in flows.items()},
                                     sources=sources)
    (args.output / "comparison.json").write_text(json.dumps(
        dict(protocol_sha256=sha256(protocol_file), cases=result), indent=2, allow_nan=False
    ) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "routes", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12001, 12002, 12003, 12004, 12005])
    compare_flows(parser.parse_args())
