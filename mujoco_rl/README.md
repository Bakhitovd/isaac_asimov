# CPU MuJoCo forward walking

This package trains a 12-leg-joint Asimov policy from random weights on the
Loudbox CPU. It uses a fixed `(vx, vy, yaw_rate) = (0.35, 0, 0)` command and
the pinned Menlo walking reference as a small reward term. Isaac Sim is not
needed. The first stage learns sustained forward motion; the second stage
refines speed to the command.

From the repository root:

```bash
python3 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install -r mujoco_rl/requirements.txt
.venv-mujoco/bin/python -m mujoco_rl.evaluate \
    mujoco_rl/checkpoints/forward_v1/policy.zip --seed 10001
```

The validated [checkpoint and preview](checkpoints/forward_v1/README.md)
are included in Git. To reproduce training from random weights:

```bash
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 8
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 16
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 24
.venv-mujoco/bin/python -m mujoco_rl.train --workers 24 \
    --steps 750000 --reward-stage progress \
    --run-dir logs/mujoco_rl/forward_stage1
.venv-mujoco/bin/python -m mujoco_rl.train --workers 24 \
    --steps 10000000 --learning-rate 0.0001 --reward-stage target \
    --resume logs/mujoco_rl/forward_stage1/checkpoint_750024.zip \
    --run-dir logs/mujoco_rl/forward_stage2
```

To record a new replay on this Loudbox:

```bash
LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu \
    .venv-mujoco/bin/python -m mujoco_rl.evaluate \
    mujoco_rl/checkpoints/forward_v1/policy.zip --seed 10001 \
    --video logs/mujoco_rl/forward_replay.mp4
```

The training run checks five deterministic 10-second replays every 250,000
steps. The target stage stops when all five stay upright, travel 2–4.5 m
forward, and drift at most 0.75 m sideways. It saves periodic checkpoints,
normalization statistics, metrics, and a best checkpoint in the chosen
`--run-dir`. To resume an interrupted run, pass the checkpoint ZIP to
`--resume` with the same `--run-dir`, `--workers`, and `--reward-stage`. The
matching `_vecnormalize.pkl` is required.

The run completed on this Loudbox at 1,000,056 total environment steps. Its
five target-stage replays traveled 3.07–3.17 m forward and at most 0.06 m
sideways in 10 seconds, without a fall. The trained checkpoint is at
`logs/mujoco_rl/forward_v5/best.zip`; the first-stage checkpoint is at
`logs/mujoco_rl/forward_v4/checkpoint_750024.zip`.

This Loudbox has no system OpenGL libraries. To render headlessly without
administrator access, `libosmesa6`, `libglapi-mesa`, and `libllvm15` were
downloaded as Ubuntu packages and extracted under
`/home/ailevate/.local/mujoco-osmesa`; the video command uses that path.
On a machine with system OSMesa, omit `LD_LIBRARY_PATH`.

The asset origin and license are in `assets/asimov/SOURCE.md`. Only the actor
is needed to command the simulated robot after training; the critic is used
during PPO training.
The actor observes simulator linear velocity, heading, and lateral position.
Those signals will need an estimator or a changed observation design before
real-robot deployment. This milestone uses a flat, fixed-friction simulation.
