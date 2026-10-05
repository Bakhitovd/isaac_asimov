"""PPO with bounded imitation updates that retain demonstrated recovery skills."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


class RecoveryPPO(PPO):
    """Keep SB3's PPO update, then reinforce successful demonstration actions.

    Each additional actor update must improve imitation loss and stay within
    the rollout KL budget. The critic and exploration parameters are untouched.
    Demonstration arrays are excluded from checkpoints and reloaded on resume.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._demo_observations = None
        self._demo_actions = None
        self._demo_groups = None
        self._demo_rng = None
        self.imitation_weight = 100.0
        self.imitation_updates = 16
        self._diagnostics_path = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_demo_observations", "_demo_actions", "_demo_groups", "_demo_rng",
                                                  "_diagnostics_path"]

    def set_demonstrations(self, observations: np.ndarray, actions: np.ndarray, normalizer,
                           weight: float = 100.0, updates: int = 16, seed: int = 42,
                           policy_means: np.ndarray | None = None,
                           episode_starts: np.ndarray | None = None) -> None:
        if (observations.ndim != 2 or observations.shape[1:] != self.observation_space.shape
                or actions.shape != (len(observations), *self.action_space.shape)
                or not len(observations) or not np.isfinite(observations).all()
                or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.0
                or not np.isfinite(weight) or weight < 0 or updates < 0):
            raise ValueError("Invalid recovery demonstrations or imitation settings")
        if policy_means is not None and (policy_means.shape != actions.shape or not np.isfinite(policy_means).all()):
            raise ValueError("Invalid reference policy means")
        if normalizer.training:
            raise ValueError("Freeze normalization statistics before attaching demonstrations")
        self._demo_observations = torch.as_tensor(normalizer.normalize_obs(observations),
                                                  dtype=torch.float32, device=self.device)
        self._demo_actions = torch.as_tensor(actions if policy_means is None else policy_means,
                                             dtype=torch.float32, device=self.device)
        self.imitation_target_kind = "demonstrated_action" if policy_means is None else "unclipped_policy_mean"
        active = np.flatnonzero(np.max(np.abs(actions), axis=1) > 0.01)
        neutral = np.flatnonzero(np.max(np.abs(actions), axis=1) <= 0.01)
        critical = np.flatnonzero(np.max(np.abs(actions - observations[:, 53:76]), axis=1) > 0.10)
        preceding = np.flatnonzero(np.max(np.abs(np.diff(actions, axis=0)), axis=1) > 0.10)
        if episode_starts is not None:
            if (episode_starts.ndim != 1 or not np.issubdtype(episode_starts.dtype, np.integer)
                    or not len(episode_starts) or episode_starts[0] != 0
                    or np.any(np.diff(episode_starts) <= 0) or episode_starts[-1] >= len(observations)):
                raise ValueError("Invalid demonstration episode boundaries")
            preceding = np.setdiff1d(preceding, episode_starts[1:] - 1)
        critical = np.union1d(critical, preceding)
        all_frames = np.arange(len(observations))
        self._demo_groups = [group if len(group) else all_frames for group in (critical, active, neutral)]
        self._demo_rng = np.random.default_rng(seed)
        self.imitation_weight = weight
        self.imitation_updates = updates

    def load_demonstrations(self, path: Path, normalizer, **kwargs) -> None:
        with np.load(path, allow_pickle=False) as dataset:
            self.set_demonstrations(dataset["observations"], dataset["actions"], normalizer,
                                    policy_means=dataset["policy_means"] if "policy_means" in dataset else None,
                                    episode_starts=dataset["episode_starts"] if "episode_starts" in dataset else None,
                                    **kwargs)

    def _imitation_update(self) -> None:
        if (self._demo_observations is None or not self.policy.action_net.weight.requires_grad
                or not self.imitation_updates or not self.imitation_weight):
            return
        parameters = list(self.policy.mlp_extractor.policy_net.parameters()) + list(self.policy.action_net.parameters())
        observations = torch.as_tensor(self.rollout_buffer.observations, device=self.device)
        actions = torch.as_tensor(self.rollout_buffer.actions, device=self.device)
        old_log_prob = torch.as_tensor(self.rollout_buffer.log_probs, device=self.device).flatten()

        def rollout_kl():
            with torch.no_grad():
                _, log_prob, _ = self.policy.evaluate_actions(observations, actions)
                log_ratio = log_prob - old_log_prob
                return float(((log_ratio.exp() - 1.0) - log_ratio).mean())

        baseline_kl = rollout_kl()
        limit = max(baseline_kl, 1.5 * self.target_kl) if self.target_kl is not None else baseline_kl + 0.015
        accepted, rejected, losses = 0, 0, []
        for _ in range(self.imitation_updates):
            indices = np.concatenate([self._demo_rng.choice(group, count)
                                      for group, count in zip(self._demo_groups, (256, 128, 128))])
            demo_obs = self._demo_observations[indices]
            demo_actions = self._demo_actions[indices]
            before_parameters = [parameter.detach().clone() for parameter in parameters]
            before_optimizer = copy.deepcopy(self.policy.optimizer.state_dict())
            predicted = self.policy.get_distribution(demo_obs).distribution.mean
            loss = torch.nn.functional.mse_loss(predicted, demo_actions)
            before_loss = float(loss.detach())
            self.policy.optimizer.zero_grad()
            (self.imitation_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.policy.optimizer.step()
            with torch.no_grad():
                after_loss = float(torch.nn.functional.mse_loss(
                    self.policy.get_distribution(demo_obs).distribution.mean, demo_actions))
            if after_loss > before_loss or rollout_kl() > limit + 1e-7:
                with torch.no_grad():
                    for parameter, saved in zip(parameters, before_parameters):
                        parameter.copy_(saved)
                self.policy.optimizer.load_state_dict(before_optimizer)
                rejected += 1
                break
            accepted += 1
            losses.append(after_loss)
        self.logger.record("train/imitation_loss", float(np.mean(losses)) if losses else before_loss)
        self.logger.record("train/imitation_updates", accepted)
        self.logger.record("train/imitation_rejected_updates", rejected)
        self.logger.record("train/post_imitation_kl", rollout_kl())

    def train(self) -> None:
        raw_obs = self.rollout_buffer.observations
        if not self.rollout_buffer.generator_ready:
            raw_obs = self.rollout_buffer.swap_and_flatten(raw_obs)
        # Fixed subsampling adds no random draws to the training stream.
        indices = np.linspace(0, len(raw_obs) - 1, min(1024, len(raw_obs)), dtype=int)
        diagnostic_obs = torch.as_tensor(raw_obs[indices], device=self.device)

        def means():
            with torch.no_grad():
                return self.policy.get_distribution(diagnostic_obs).distribution.mean.detach().clone()

        before = means()
        super().train()
        after_ppo = means()
        observations = torch.as_tensor(self.rollout_buffer.observations, device=self.device)
        actions = torch.as_tensor(self.rollout_buffer.actions, device=self.device)
        old_log_prob = torch.as_tensor(self.rollout_buffer.log_probs, device=self.device).flatten()
        with torch.no_grad():
            _, log_prob, _ = self.policy.evaluate_actions(observations, actions)
            log_ratio = log_prob - old_log_prob
            ppo_kl = float(((log_ratio.exp() - 1.0) - log_ratio).mean())
        self.logger.record("train/ppo_kl", ppo_kl)
        self.logger.record("train/imitation_updates", 0)
        self.logger.record("train/imitation_rejected_updates", 0)
        self.logger.record("train/post_imitation_kl", ppo_kl)
        self._imitation_update()
        after_imitation = means()
        ppo_change, imitation_change = after_ppo - before, after_imitation - after_ppo
        self.logger.record("train/ppo_action_change_rms", float(ppo_change.square().mean().sqrt()))
        self.logger.record("train/imitation_action_change_rms", float(imitation_change.square().mean().sqrt()))
        self.logger.record("train/net_action_change_rms", float((after_imitation - before).square().mean().sqrt()))
        denominator = ppo_change.norm() * imitation_change.norm()
        self.logger.record("train/imitation_alignment", float((ppo_change * imitation_change).sum() / denominator)
                           if denominator > 0 else 0.0)
        if self._diagnostics_path is not None:
            values = self.logger.name_to_value
            record = {"training_steps": self.num_timesteps, **{
                key.removeprefix("train/"): float(value) for key, value in values.items()
                if key.startswith("train/") and np.isscalar(value) and np.isfinite(value)}}
            with Path(self._diagnostics_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
