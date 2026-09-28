# Forward walking policy v1

This directory contains the validated CPU MuJoCo walking policy:

- `policy.zip`: Stable-Baselines3 PPO model, including actor and critic.
- `policy_vecnormalize.pkl`: observation normalization required for replay.
- `evaluations.jsonl`: deterministic evaluation records from the refinement run.
- `preview.mp4`: ten-second replay at seed 10001.

Training began with random weights and used the two stages described in
[`../../README.md`](../../README.md). The final policy was saved after
1,000,056 environment steps. Across five fixed seeds (10001–10005), ten-second
replays traveled 3.073–3.174 m forward, drifted at most 0.058 m sideways, and
had no falls. The requested forward command was 0.35 m/s.

From the repository root, after installing `mujoco_rl/requirements.txt`:

```bash
.venv-mujoco/bin/python -m mujoco_rl.evaluate \
    mujoco_rl/checkpoints/forward_v1/policy.zip --seed 10001
```

This checkpoint is for the flat-ground simulator. Its observations include
simulator velocity, heading, and lateral position; those signals need an
estimator or a revised policy before real-robot use.
