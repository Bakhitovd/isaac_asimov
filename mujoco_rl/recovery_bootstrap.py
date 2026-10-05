"""Bootstrap recovery PPO from successful, physics-tested simulated rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .skill_env import RECOVERY_TILT_RANGES, SkillEnv
from .skill_eval import evaluate
from .skill_train import _recovery_level_from_checkpoint, _stats_path


CONTROL_JOINTS = np.array([0, 3, 4, 6, 9, 10])


def demonstration_action(controls: np.ndarray, step: int, active: bool,
                         feedback: np.ndarray | None = None,
                         observation: np.ndarray | None = None, pose: str = "front",
                         phase_ends: tuple[int, int, int] = (5, 15, 40)) -> np.ndarray:
    """Three short leg phases, followed by neutral standing commands."""
    action = np.zeros(23, dtype=np.float32)
    if active and step < phase_ends[2]:
        phase = 0 if step < phase_ends[0] else 1 if step < phase_ends[1] else 2
        if controls.shape[1] == 23:
            action[:] = controls[phase]
        else:
            action[CONTROL_JOINTS] = controls[phase]
    if active and feedback is not None and observation is not None:
        sideways = pose in {"left", "right"}
        lean = float(np.arctan2(-observation[4] if sideways else observation[3], -observation[5]))
        rate = float(observation[0 if sideways else 1]) * 4.0
        hip = feedback[0] * lean + feedback[1] * rate
        ankle = feedback[2] * lean + feedback[3] * rate
        if len(feedback) == 6:
            velocity = float(observation[80 if sideways else 79]) * 2.0
            hip += feedback[4] * velocity
            ankle += feedback[5] * velocity
        action[[1, 7] if sideways else [0, 6]] += [hip, hip] if sideways else [hip, -hip]
        action[[5, 11] if sideways else [4, 10]] += [ankle, ankle] if sideways else [ankle, -ankle]
    return np.clip(action, -1.0, 1.0)


def collect_demonstrations(controls: np.ndarray | list[dict], level: int, episodes: int, seed: int,
                           preview_next_level: bool = False, action_noise: float = 0.0,
                           recovery_sensor_memory: bool = False, recovery_phase_features: bool = False,
                           learner=None, normalizer=None, learner_fraction: float = 0.0,
                           keep_failed: bool = False, collection_poses: tuple[str, ...] | None = None) -> dict:
    """Label successful trajectories, optionally perturbing executed commands.

    Perturbations expose the actor to states slightly off the teacher's path;
    labels retain the teacher's sensor-dependent correction.
    """
    if not np.isfinite(action_noise) or not 0.0 <= action_noise <= 0.20:
        raise ValueError("Demonstration action noise must be in [0, 0.20]")
    if (not np.isfinite(learner_fraction) or not 0.0 <= learner_fraction <= 1.0
            or learner_fraction and (learner is None or normalizer is None)):
        raise ValueError("Learner-state collection requires a policy, normalizer, and fraction in [0, 1]")
    if collection_poses is not None and (not collection_poses or any(
            pose not in {"front", "back", "left", "right"} for pose in collection_poses)):
        raise ValueError("Invalid collection poses")
    observations, actions = [], []
    successes = 0
    episode_starts = []
    rng = np.random.default_rng(seed)
    env = SkillEnv("recover", recovery_level=level, recovery_sensor_memory=recovery_sensor_memory,
                   recovery_phase_features=recovery_phase_features)
    try:
        for index in range(episodes):
            # Rehearse neutral standing at earlier levels as well as recovery.
            pose_cycle = index // (len(collection_poses) if collection_poses else 4)
            episode_level = level if pose_cycle % 5 else (pose_cycle // 5) % max(1, level)
            if preview_next_level and pose_cycle % 8 == 0 and pose_cycle % 5 != 0:
                episode_level = min(level + 1, len(RECOVERY_TILT_RANGES) - 1)
            env.set_recovery_level(episode_level)
            reset_options = ({"recovery_pose": collection_poses[index % len(collection_poses)]}
                             if collection_poses else None)
            observation, _ = (env.reset(seed=seed + index, options=reset_options) if reset_options
                              else env.reset(seed=seed + index))
            if isinstance(controls, list):
                selected = next((item for item in controls
                                 if item["pose"] == env.recovery_pose
                                 and item["min_tilt_deg"] < env.recovery_tilt_deg <= item["max_tilt_deg"]), None)
                active = selected is not None
                episode_controls = selected["controls"] if active else np.zeros((3, 23))
                feedback = np.asarray(selected["feedback"]) if active and "feedback" in selected else None
                phase_ends = tuple(selected.get("phase_ends", (5, 15, 40))) if active else (5, 15, 40)
            else:
                active = env.recovery_pose == "front" and env.recovery_tilt_deg > 16.0
                episode_controls = controls
                feedback = None
                phase_ends = (5, 15, 40)
            trajectory = []
            for step in range(600):
                action = demonstration_action(episode_controls, step, active, feedback, observation,
                                              env.recovery_pose, phase_ends)
                trajectory.append((observation.copy(), action))
                executed = action
                if learner_fraction:
                    prediction, _ = learner.predict(normalizer.normalize_obs(observation[None, :]), deterministic=True)
                    executed = (1.0 - learner_fraction) * action + learner_fraction * prediction[0]
                if action_noise:
                    executed = executed + rng.normal(0.0, action_noise, 23)
                executed = np.clip(executed, -1.0, 1.0)
                observation, _, terminated, truncated, info = env.step(executed)
                if terminated or truncated:
                    break
            if info["success"]:
                successes += 1
            if info["success"] or keep_failed:
                episode_starts.append(len(observations))
                observations.extend(item[0] for item in trajectory)
                actions.extend(item[1] for item in trajectory)
            if (index + 1) % 64 == 0:
                print(f"[bootstrap] demonstrations={index + 1}, successful={successes}", flush=True)
    finally:
        env.close()
    if not observations or not any(np.any(action) for action in actions):
        raise RuntimeError("No successful active recovery demonstrations were collected")
    return {"observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "successes": successes, "episodes": episodes, "episode_starts": np.asarray(episode_starts),
            "retained_episodes": len(episode_starts)}


def load_controls(path: Path) -> np.ndarray | list[dict]:
    """Load a pose manifest, or a search array with its pose/feedback metadata."""
    if path.suffix == ".json":
        controls = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(controls, list):
            raise ValueError("Use the search .npy file or a JSON list of pose/angle controls")
    else:
        controls = np.load(path, allow_pickle=False)
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            controls = [{"pose": metadata["pose"], "min_tilt_deg": 16.0, "max_tilt_deg": 25.0,
                         "controls": controls}]
            if metadata.get("feedback") is not None:
                controls[0]["feedback"] = metadata["feedback"]
            if "phase_ends" in metadata:
                controls[0]["phase_ends"] = metadata["phase_ends"]
    if isinstance(controls, list):
        for item in controls:
            item["controls"] = np.asarray(item["controls"], dtype=np.float32)
            if (item["pose"] not in {"front", "back", "left", "right"}
                    or not 0.0 <= item["min_tilt_deg"] < item["max_tilt_deg"] <= 90.0):
                raise ValueError("Invalid pose or angle range in control manifest")
            ends = item.get("phase_ends", (5, 15, 40))
            if (len(ends) != 3 or any(type(value) is not int for value in ends)
                    or not 0 < ends[0] < ends[1] < ends[2] <= 600):
                raise ValueError("Phase ends must contain three increasing integer steps in [1, 600]")
            if "feedback" in item:
                gains = np.asarray(item["feedback"])
                if gains.shape not in {(4,), (6,)} or not np.isfinite(gains).all() or np.max(np.abs(gains)) > 4.0:
                    raise ValueError("Feedback must contain four or six finite gains in [-4, 4]")
        arrays = [item["controls"] for item in controls]
    else:
        arrays = [controls]
    if not arrays or any(array.shape not in {(3, 6), (3, 23)} or not np.isfinite(array).all()
                         or np.max(np.abs(array)) > 1.0 for array in arrays):
        raise ValueError("Controls must be finite 3x6 or 3x23 arrays in [-1, 1]")
    return controls


def _demonstration_arrays(dataset: dict) -> dict:
    """Keep simulator actions bounded while preserving reference-mean labels."""
    arrays = {key: dataset[key] for key in ("observations", "actions", "episode_starts") if key in dataset}
    if np.max(np.abs(arrays["actions"])) > 1.0:
        arrays["policy_means"] = arrays["actions"]
        arrays["actions"] = np.clip(arrays["actions"], -1.0, 1.0)
    return arrays


def fit_actor(model: PPO, normalizer: VecNormalize, dataset: dict, updates: int, seed: int,
              preserve_motor_history: bool = False) -> dict:
    observations = torch.as_tensor(normalizer.normalize_obs(dataset["observations"]), dtype=torch.float32)
    # Teacher forcing makes previous-action copying an easy shortcut. Learn
    # the motion from sensors and phase; PPO can later use motor history.
    first_layer = next(layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear))
    if not preserve_motor_history:
        observations[:, 53:76] = 0.0
        with torch.no_grad():
            first_layer.weight[:, 53:76].zero_()
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    active = np.flatnonzero(np.any(dataset["actions"] != 0.0, axis=1))
    neutral = np.flatnonzero(~np.any(dataset["actions"] != 0.0, axis=1))
    if not len(active) or not len(neutral):
        raise ValueError("Demonstrations must contain active recovery and neutral standing")
    changed = np.max(np.abs(dataset["actions"] - dataset["observations"][:, 53:76]), axis=1) > 0.10
    # Include the frame before a transition; otherwise oversampling only the
    # new command trains an anticipatory switch that can destabilize recovery.
    transitions = np.max(np.abs(np.diff(dataset["actions"], axis=0)), axis=1) > 0.10
    if "episode_starts" in dataset:
        transitions[dataset["episode_starts"][1:] - 1] = False
    changed[:-1] |= transitions
    critical = np.flatnonzero(changed)
    if not len(critical):
        critical = active
    parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    rng = np.random.default_rng(seed)
    losses = []
    model.policy.set_training_mode(True)
    for update in range(updates):
        indices = np.concatenate((rng.choice(critical, 256), rng.choice(active, 128), rng.choice(neutral, 128)))
        predicted = model.policy.get_distribution(observations[indices]).distribution.mean
        loss = torch.nn.functional.mse_loss(predicted, actions[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 250 == 0:
            print(f"[bootstrap] update={update + 1}, action_mse={np.mean(losses[-100:]):.6f}", flush=True)
    for parameter in parameters:
        model.policy.optimizer.state.pop(parameter, None)
    model.policy.set_training_mode(False)
    return {"updates": updates, "final_action_mse": float(np.mean(losses[-100:])),
            "active_frames": len(active), "neutral_frames": len(neutral), "critical_frames": len(critical),
            "ignore_previous_action_during_bootstrap": not preserve_motor_history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("controls", type=Path, help="3x6/3x23 action array or a JSON pose/angle control manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for the small actor network")
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--eval-seed-start", type=int, default=60000)
    parser.add_argument("--preview-next-level", action="store_true")
    parser.add_argument("--action-noise", type=float, default=0.0,
                        help="Executed-command perturbation std; successful frames retain clean teacher labels")
    parser.add_argument("--sensor-memory", action="store_true",
                        help="Remember the initial noisy IMU tilt in unused recovery observation channels")
    parser.add_argument("--phase-features", action="store_true",
                        help="Expose the recovery motion phase in unused command channels")
    parser.add_argument("--reuse-demonstrations", type=Path)
    parser.add_argument("--preserve-motor-history", action="store_true",
                        help="Retain learned previous-command features when repairing a PPO checkpoint")
    parser.add_argument("--learner-fraction", type=float, default=0.0,
                        help="Execute a mixture of checkpoint and expert actions; expert labels stay unchanged")
    parser.add_argument("--keep-failed-demonstrations", action="store_true",
                        help="Retain queried expert labels from failed learner rollouts")
    parser.add_argument("--replay-demonstrations", type=Path,
                        help="Append earlier demonstrations to retain successful behavior during learner-state fitting")
    parser.add_argument("--collection-poses", nargs="+", choices=("front", "back", "left", "right"),
                        help="Collect selected directions while retaining others through replay")
    args = parser.parse_args()
    if args.episodes < 8 or args.updates < 1 or args.torch_threads < 1:
        parser.error("Use at least eight episodes and one update")
    if not np.isfinite(args.action_noise) or not 0.0 <= args.action_noise <= 0.20:
        parser.error("--action-noise must be in [0, 0.20]")
    if args.reuse_demonstrations and args.action_noise:
        parser.error("--action-noise requires collecting new demonstrations")
    if args.reuse_demonstrations and args.sensor_memory:
        parser.error("Collect new demonstrations when enabling sensor memory")
    if not np.isfinite(args.learner_fraction) or not 0.0 <= args.learner_fraction <= 1.0:
        parser.error("--learner-fraction must be in [0, 1]")
    if args.reuse_demonstrations and (args.learner_fraction or args.keep_failed_demonstrations):
        parser.error("Learner-state collection requires new demonstrations")
    if args.reuse_demonstrations and args.collection_poses:
        parser.error("--collection-poses requires new demonstrations")
    try:
        controls = load_controls(args.controls)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.torch_threads)
    level = _recovery_level_from_checkpoint(args.checkpoint)
    normalizer = VecNormalize.load(str(_stats_path(args.checkpoint)), DummyVecEnv([lambda: SkillEnv("recover")]))
    normalizer.training = False
    try:
        model = PPO.load(str(args.checkpoint), device="cpu")
        model.recovery_sensor_memory = args.sensor_memory or getattr(model, "recovery_sensor_memory", False)
        model.recovery_phase_features = args.phase_features or getattr(model, "recovery_phase_features", False)
        if args.reuse_demonstrations:
            saved = np.load(args.reuse_demonstrations, allow_pickle=False)
            previous_report = json.loads(args.reuse_demonstrations.with_name(
                args.reuse_demonstrations.stem.removesuffix("_demonstrations") + "_report.json").read_text())
            dataset = {"observations": saved["observations"], "actions": saved["actions"],
                       "successes": previous_report["successful_demonstrations"],
                       "episodes": previous_report["episodes"]}
            if "policy_means" in saved:
                dataset["actions"] = saved["policy_means"]
            if "episode_starts" in saved:
                dataset["episode_starts"] = saved["episode_starts"]
            elif args.phase_features and model.recovery_sensor_memory:
                raw = dataset["observations"]
                starts = (raw[:, 76] == 0) & (raw[:, 77] == 1) & np.all(raw[:, 3:5] == raw[:, 81:83], axis=1)
                dataset["episode_starts"] = np.flatnonzero(starts)
                if len(dataset["episode_starts"]) != dataset["successes"]:
                    raise ValueError("Could not identify every episode for phase feature migration")
            if args.phase_features:
                if "episode_starts" not in dataset:
                    raise ValueError("Collect new demonstrations to enable phase features")
                starts = dataset["episode_starts"]
                for start, end in zip(starts, [*starts[1:], len(dataset["observations"])]):
                    step = np.arange(end - start)
                    phase = np.where(step < 5, 0, np.where(step < 15, 1, np.where(step < 40, 2, 3)))
                    dataset["observations"][start:end, 6] = phase & 1
                    dataset["observations"][start:end, 78] = phase >> 1
        else:
            dataset = collect_demonstrations(controls, level, args.episodes, args.seed,
                                             args.preview_next_level, args.action_noise, model.recovery_sensor_memory,
                                             model.recovery_phase_features, model, normalizer,
                                             args.learner_fraction, args.keep_failed_demonstrations,
                                             tuple(args.collection_poses) if args.collection_poses else None)
        if args.replay_demonstrations:
            with np.load(args.replay_demonstrations, allow_pickle=False) as replay:
                count = len(dataset["observations"])
                dataset["episode_starts"] = np.concatenate((dataset["episode_starts"],
                                                            replay["episode_starts"] + count))
                for key in ("observations", "actions"):
                    dataset[key] = np.concatenate((dataset[key], replay[key]))
        print(f"[bootstrap] collected {dataset['successes']}/{dataset['episodes']} successful rollouts", flush=True)
        normalizer.obs_rms.count = min(normalizer.obs_rms.count, 5000.0)
        normalizer.obs_rms.update(dataset["observations"])
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
        for parameter in model.policy.value_net.parameters():
            model.policy.optimizer.state.pop(parameter, None)
        report = fit_actor(model, normalizer, dataset, args.updates, args.seed, args.preserve_motor_history)
        report.update({"source_checkpoint": str(args.checkpoint), "controls": str(args.controls),
                       "demonstration_seed_start": (previous_report["demonstration_seed_start"]
                                                    if args.reuse_demonstrations else args.seed),
                       "episodes": dataset["episodes"],
                       "successful_demonstrations": dataset["successes"],
                       "retained_episodes": len(dataset["episode_starts"]),
                       "learner_fraction": (previous_report.get("learner_fraction", 0.0)
                                             if args.reuse_demonstrations else args.learner_fraction),
                       "keep_failed_demonstrations": (previous_report.get("keep_failed_demonstrations", False)
                                                      if args.reuse_demonstrations else args.keep_failed_demonstrations),
                       "replay_demonstrations": (previous_report.get("replay_demonstrations")
                                                 if args.reuse_demonstrations else
                                                 str(args.replay_demonstrations) if args.replay_demonstrations else None),
                       "collection_poses": (previous_report.get("collection_poses")
                                            if args.reuse_demonstrations else args.collection_poses),
                       "torch_threads": args.torch_threads,
                       "observation_statistics_refit": True, "critic_output_reset": True})
        report["next_level_preview"] = (previous_report["next_level_preview"]
                                        if args.reuse_demonstrations else args.preview_next_level)
        report["demonstration_action_noise"] = (previous_report.get("demonstration_action_noise", 0.0)
                                                if args.reuse_demonstrations else args.action_noise)
        report["reused_demonstrations"] = str(args.reuse_demonstrations) if args.reuse_demonstrations else None
        report["recovery_sensor_memory"] = model.recovery_sensor_memory
        report["recovery_phase_features"] = model.recovery_phase_features
        report["development"] = evaluate(model, normalizer, "recover", 20000, 48, level)
        report["current_validation"] = evaluate(model, normalizer, "recover", 30000, 48, level)
        report["fresh_evaluation"] = evaluate(model, normalizer, "recover", args.eval_seed_start, 50, level)
        report["retention"] = evaluate(model, normalizer, "recover", 30000, 48, max(0, level - 1))
        source_config = json.loads((args.checkpoint.parent / "config.json").read_text(encoding="utf-8"))
        stress_seed = source_config.get("recovery_stress_seed_start")
        if stress_seed is not None:
            report["stress_validation"] = evaluate(model, normalizer, "recover", stress_seed, 50, level)
            if level - 1 in getattr(model, "recovery_stress_validated_levels", []):
                report["stress_retention"] = evaluate(model, normalizer, "recover", stress_seed, 50, level - 1)
        retry = getattr(model, "recovery_validated_retry_panels", {}).get(str(level - 1))
        if retry is not None:
            report["retry_retention"] = evaluate(model, normalizer, "recover", retry["seed_start"], 50, level - 1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        arrays = _demonstration_arrays(dataset)
        np.savez_compressed(args.output.with_name(args.output.stem + "_demonstrations.npz"), **arrays)
        model.save(str(args.output))
        normalizer.save(str(_stats_path(args.output)))
        args.output.with_name(args.output.stem + "_training.json").write_text(
            json.dumps({"recovery_level": level}) + "\n", encoding="utf-8")
        config = source_config
        config["recovery_version"] = 11 if model.recovery_phase_features else 10 if model.recovery_sensor_memory else 9
        config["recovery_sensor_memory"] = model.recovery_sensor_memory
        config["recovery_phase_features"] = model.recovery_phase_features
        config["recovery_tilt_ranges_deg"] = RECOVERY_TILT_RANGES
        config["recovery_reset_controller"] = "hold_initial_pose"
        snapshot = args.output.parent / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        config["source_sha256"] = {}
        for source in Path(__file__).parent.glob("*.py"):
            contents = source.read_bytes()
            (snapshot / source.name).write_bytes(contents)
            config["source_sha256"][source.name] = hashlib.sha256(contents).hexdigest()
        config["bootstrap"] = {key: value for key, value in report.items()
                               if not isinstance(value, dict) or "pass_rate" not in value}
        if config.get("recovery_demonstrations"):
            config["recovery_demonstrations"] = str(args.output.with_name(
                args.output.stem + "_demonstrations.npz").resolve())
        (args.output.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        args.output.with_name(args.output.stem + "_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("[bootstrap] " + json.dumps({key: value["pass_rate"] for key, value in report.items()
                                            if isinstance(value, dict) and "pass_rate" in value}), flush=True)
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
