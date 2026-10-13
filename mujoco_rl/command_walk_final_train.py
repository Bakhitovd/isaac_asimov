"""Resumable bounded training blocks and successful-motion reference protection."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .command_walk import FAMILIES
from .command_walk_final import FinalEnv, contract
from .command_walk_guarded_ppo import GuardedPPO
from .command_walk_robust_eval import load_pair
from .command_walk_robust_train import create_model
from .command_walk_train import save_bundle
from .run_training import actor_parameters
from .skill_campaign import atomic_json, sha256


class FinalPPO(GuardedPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["anchor_observations", "anchor_means"]

    def proposal_limits(self):
        with torch.no_grad():
            means = self.policy.get_distribution(self.anchor_observations).distribution.mean
            per_state = (means - self.anchor_means).square().mean(dim=1).sqrt()
            rms = float(per_state.square().mean().sqrt())
            p95 = float(torch.quantile(per_state, 0.95))
        return rms <= 0.01 and p95 <= 0.02, dict(reference_rms=rms, reference_p95=p95)

    def attach_anchor(self, path, norm):
        with np.load(path, allow_pickle=False) as data:
            self.anchor_observations = torch.as_tensor(norm.normalize_obs(data["observations"]), device=self.device)
            self.anchor_means = torch.as_tensor(data["means"], device=self.device)


def reference(checkpoint, output, deadline):
    output = Path(output)
    if output.exists():
        return
    model, norm = load_pair(checkpoint)
    rows, means, labels, seeds = [], [], [], []
    try:
        for family_index, case in enumerate(FAMILIES):
            successes = 0
            env = FinalEnv(case=case)
            try:
                for index in range(100):
                    if time.time() >= deadline:
                        raise TimeoutError("Reference collection deadline")
                    seed = 70_000_001 + family_index * 100_000 + index
                    obs, _ = env.reset(seed=seed)
                    observations, actions = [], []
                    while True:
                        with torch.no_grad():
                            normalized = norm.normalize_obs(obs[None, : model.observation_space.shape[0]])
                            mean = (
                                model.policy.get_distribution(torch.as_tensor(normalized, device=model.device))
                                .distribution.mean.cpu()
                                .numpy()[0]
                            )
                        observations.append(obs.copy())
                        actions.append(mean.copy())
                        obs, _, done, truncated, info = env.step(np.clip(mean, -1, 1))
                        if done or truncated:
                            break
                    if info["success"]:
                        positions = np.linspace(0, len(observations) - 1, 32, dtype=int)
                        rows.extend(np.array(observations)[positions])
                        means.extend(np.array(actions)[positions])
                        labels.extend([case] * 32)
                        seeds.extend([seed] * 32)
                        successes += 1
                    if successes == 16:
                        break
                if successes < 16:
                    raise RuntimeError(f"Insufficient successful reference episodes: {case}")
                print(f"[reference] {case}: 512 states from 16 successful episodes", flush=True)
            finally:
                env.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observations=np.asarray(rows, dtype=np.float32),
            means=np.asarray(means, dtype=np.float32),
            families=np.asarray(labels),
            seeds=np.asarray(seeds),
        )
        atomic_json(
            output.with_suffix(".json"),
            dict(
                policy_sha256=sha256(Path(checkpoint)),
                samples=len(rows),
                per_family=512,
                seed_base=70_000_001,
                successful_episodes_only=True,
                sha256=sha256(output),
            ),
        )
    finally:
        norm.close()


class Callback(BaseCallback):
    def __init__(self, output, deadline):
        super().__init__()
        self.output, self.deadline = Path(output), deadline
        self.last = 0.0
        self.simulation_steps = {}

    def _on_rollout_start(self):
        enabled = self.model.num_timesteps >= self.model.command_walk_state["warmup_until"]
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)
        self.model.policy.log_std.requires_grad_(False)

    def _on_step(self):
        now = time.time()
        for info in self.locals.get("infos", []):
            family = info.get("family", "unknown")
            self.simulation_steps[family] = self.simulation_steps.get(family, 0) + 1
        if now - self.last > 10:
            atomic_json(
                self.output / "status.json",
                dict(
                    state="training",
                    heartbeat=now,
                    steps=self.model.num_timesteps,
                    phase=self.model.final_config["phase"],
                    seconds=self.model.final_config["seconds"],
                ),
            )
            self.last = now
        return now < self.deadline


def train_block(job):
    checkpoint, output, anchor, seed, reward_kind, phase, seconds, actor_rollouts, fresh, deadline = job
    checkpoint, output, anchor = Path(checkpoint), Path(output), Path(anchor)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    norm = model = None
    with (
        (output / "train.log").open("a", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            model, norm, _, rng = create_model(
                checkpoint,
                variant="control",
                seed=seed,
                workers=8,
                rollout_steps=768,
                model_class=FinalPPO,
                transfer=fresh,
                preserve_critic=True,
                warmup_rollouts=8,
                env_factory=lambda: FinalEnv(training=True, reward_kind=reward_kind, phase=phase, seconds=seconds),
            )
            if fresh:
                with torch.no_grad():
                    model.policy.log_std.add_(float(np.log(0.1)))
                model.final_config = dict(
                    contract=contract(),
                    reward_kind=reward_kind,
                    seed=seed,
                    phase=phase,
                    seconds=seconds,
                    anchor_sha256=sha256(anchor),
                )
                model.guard_enabled = True
                model.guard_mean_kl, model.guard_max_kl = 0.01, 0.1
                model.command_walk_state["final_origin"] = model.num_timesteps
            else:
                if (
                    model.final_config["anchor_sha256"] != sha256(anchor)
                    or model.final_config["reward_kind"] != reward_kind
                ):
                    raise ValueError("Resume changed reference or reward configuration")
                model.final_config.update(phase=phase, seconds=seconds)
            model.guard_path = output / "guard.jsonl"
            model._diagnostics_path = output / "updates.jsonl"
            model.attach_anchor(anchor, norm)
            model.audit_hook = lambda _: None
            callback = Callback(output, deadline)
            remaining_warmup = max(0, model.command_walk_state["warmup_until"] - model.num_timesteps)
            start = model.num_timesteps
            model.learn(
                total_timesteps=remaining_warmup + actor_rollouts * 6144, reset_num_timesteps=False, callback=callback
            )
            for p in actor_parameters(model):
                p.requires_grad_(True)
            model.policy.log_std.requires_grad_(False)
            # A resumed process resets environments at a rollout boundary. It preserves
            # weights, Adam, observation/return statistics, counters, and curriculum.
            saved = save_bundle(model, norm, output, f"block_{time.time_ns()}")
            guards = [json.loads(line) for line in (output / "guard.jsonl").read_text().splitlines()]
            actors = [g for g in guards if g["enabled"] and g["training_steps"] > start]
            technical = len(actors) == actor_rollouts and all(
                g["accepted"] > 0
                and g["mean_action_change_rms"] > 1e-8
                and g["restoration_checks"] == g["rejected"]
                and g["final_mean_kl"] <= 0.0100001
                and g["final_max_kl"] <= 0.1000001
                and g["reference_rms"] <= 0.0100001
                and g["reference_p95"] <= 0.0200001
                for g in actors
            )
            result = dict(
                checkpoint=str(saved.resolve()),
                technical_passed=technical,
                simulation_steps=callback.simulation_steps,
                steps=model.num_timesteps,
                additional_steps=model.num_timesteps - start,
                actor_steps=max(0, model.num_timesteps - model.command_walk_state["warmup_until"]),
                reward_kind=reward_kind,
                seed=seed,
                phase=phase,
                seconds=seconds,
                rng_hash=rng,
                state="completed" if time.time() < deadline else "budget_reached",
                heartbeat=time.time(),
            )
            atomic_json(output / "block.json", result)
            atomic_json(output / "status.json", result)
            return result
        except BaseException as error:
            atomic_json(output / "status.json", dict(state="failed", heartbeat=time.time(), error=repr(error)))
            raise
        finally:
            if norm is not None:
                norm.close()
