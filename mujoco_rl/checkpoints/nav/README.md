# Full-body waypoint navigation policy

This directory publishes the qualified navigation policy from the
`skills_campaign_v1/revision_2` campaign. The neural network controls 23 leg,
waist, and arm joints at 50 Hz using 83 observations. It walks toward a waypoint
and stops near the goal in the flat-ground MuJoCo task.

- `accepted.zip`: Stable-Baselines3 PPO checkpoint.
- `accepted_vecnormalize.pkl`: matching observation normalization statistics.
- `qualification.json`: original qualification record, file hashes, and all
  development and holdout episodes. Its source path records the original local
  training artifact.

Keep the policy and normalization file together. Both were copied without
modification from the qualified local artifacts. The environment version is 12.
The policy passed 44/50 development episodes and 44/50 separate holdout episodes
(88% each). Results are simulation-only; physical-robot performance is untested.

Evaluate from the repository root with the CPU environment installed and the
`third_party/asimov-1` submodule initialized:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_eval \
    mujoco_rl/checkpoints/nav/accepted.zip --skill nav \
    --seed-start 2000900000 --seeds 50
```

This command reproduces the recorded holdout panel; use separate seeds for new
evaluations. The [navigation demo](../../demos/2026-10-05/02_waypoint_navigation_selected_success.mp4)
shows selected passing seed 2000800000. See the [multi-skill guide](../../MULTI_SKILL.md)
for training and evaluation details.
