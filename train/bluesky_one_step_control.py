"""Train the Exp 6 one-step control through the selected checkpoint's training decision."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from envs.bluesky_graph import ObservationProfile
from eval.bluesky_pair import _checksum
from train.bluesky_td3 import train


def run(args):
    config = json.loads(args.control_config.read_text())
    if (not config["multi_step_return"] or config["return_window"] != 24
            or not config["initialize"] or config["resume"] is not None):
        raise ValueError("Exp 6 needs a run initialized with 24-step returns and no resume")
    primary_digest = _checksum(args.checkpoint)
    primary = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    decisions, updates = primary["decisions"], primary["updates"]
    if not (0 < decisions <= config["decisions"] and updates > 0):
        raise ValueError("Choose a trained checkpoint within the Stage 2 decision budget")
    profile = ObservationProfile(**primary["observation_profile"])
    reference_digest = _checksum(Path(config["initialize"]))
    config.update(multi_step_return=False, decisions=decisions, target_episodes=0, target_updates=0,
                  output=args.output / "training")
    for name in ("initialize", "anchor", "performance_calibration"):
        if config[name] is not None:
            config[name] = Path(config[name])
    config["teacher_data"] = [Path(path) for path in config["teacher_data"]]
    train(argparse.Namespace(**config))
    training = config["output"]
    saved = training / "checkpoint.pt"
    control = torch.load(saved, map_location="cpu", weights_only=True)
    if (control["decisions"], control["updates"]) != (decisions, updates):
        raise ValueError("One-step control reached a different training decision or learner update count")
    training_profile = ObservationProfile(**control["observation_profile"])
    if replace(profile, detection_metres=training_profile.detection_metres) != training_profile:
        raise ValueError("Primary and one-step checkpoint observation features differ")
    if _checksum(args.checkpoint) != primary_digest or _checksum(config["initialize"]) != reference_digest:
        raise ValueError("An input checkpoint changed during one-step training")

    output = args.output / "selected-one-step.pt"
    if training_profile == profile:
        output.hardlink_to(saved)
    else:
        control["observation_profile"] = asdict(profile)
        torch.save(control, output)
    result = {
        "checkpoint_sha256": _checksum(output),
        "primary_checkpoint_sha256": primary_digest,
        "control_config_sha256": _checksum(args.control_config),
        "one_step_config_sha256": _checksum(training / "config.json"),
        "initialization_sha256": reference_digest,
        "decisions": decisions,
        "updates": updates,
        "training_observation_profile": asdict(training_profile),
        "deployment_observation_profile": asdict(profile),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-config", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())
