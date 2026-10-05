"""Reference-policy rollouts retain successful observed behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_rl import recovery_retention


@pytest.mark.parametrize("index,expected_level", [(0, 7), (20, 8), (32, 0), (36, 1), (72, 2)])
@pytest.mark.parametrize("success", [True, False])
def test_reference_rollouts_keep_only_successful_trajectories(monkeypatch, index, expected_level, success):
    class Environment:
        def set_recovery_level(self, level):
            self.level = level

        def reset(self, seed, options):
            self.seed = seed
            self.pose = options["recovery_pose"]
            self.recovery_tilt_deg = 20.0
            self.steps = 0
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.zeros(83, dtype=np.float32), 0.0, self.steps == 2, False, {
                "success": success, "recovery_pose": self.pose}

    env = Environment()
    monkeypatch.setattr(recovery_retention, "_environment", env)
    monkeypatch.setattr(recovery_retention, "_level", 7)
    monkeypatch.setattr(recovery_retention, "_seed_start", 310000)
    monkeypatch.setattr(recovery_retention, "_normalizer", SimpleNamespace(normalize_obs=lambda obs: obs))
    monkeypatch.setattr(recovery_retention, "_model", SimpleNamespace(
        device="cpu", policy=SimpleNamespace(get_distribution=lambda obs: SimpleNamespace(
            distribution=SimpleNamespace(mean=torch.full((1, 23), 1.2))))))
    summary, observations, actions, means = recovery_retention._trial(index)
    assert env.level == expected_level
    assert env.seed == 310000 + index
    assert summary["success"] is success
    assert summary["tilt_deg"] == 20.0
    assert observations.shape == (2 if success else 0, 83)
    assert actions.shape == (2 if success else 0, 23)
    assert means.shape == actions.shape
    if success:
        assert np.allclose(actions, 1.0)
        assert np.allclose(means, 1.2)


def test_collection_covers_every_direction_at_every_rehearsal_level():
    from collections import Counter

    counts = Counter(recovery_retention.collection_case(index, 8) for index in range(160))
    for level in range(10):
        by_pose = [counts[level, pose] for pose in ("front", "back", "left", "right")]
        assert min(by_pose) > 0
        assert len(set(by_pose)) == 1
    assert sum(n for (level, _), n in counts.items() if level == 8) == 80
    assert sum(n for (level, _), n in counts.items() if level == 9) == 48
