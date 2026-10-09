# Continuous forward walking and turning

The `asimov_command_walk_v1` contract trains one policy for forward walking,
left/right walking turns, left/right stationary turns, and stopping. Start from
`checkpoints/nav/accepted.zip` and its matching normalization file. Commands are
forward speed in 0–0.3 m/s and yaw rate in −0.6–0.6 rad/s; positive yaw turns left.
All results are simulation-only.

## Interface and control

`CommandWalkEnv.set_command(forward_m_s, yaw_rad_s)` accepts continuous commands
when the environment is constructed with `external=True`. Refresh a live command
at least every 0.5 seconds; stale commands request a stop. Command changes preserve
physical state and sensor history. Commands ramp at at most 0.3 m/s² and 1.2 rad/s².

The actor has 85 observations: the navigation policy's first 83 channels, with
its waypoint channels held at zero, followed by sine/cosine of reference-heading
error. The reference heading integrates commanded turn rate. New input weights
start at zero, old observation statistics are frozen, and new channels use unit
scaling. The critic output and reward statistics restart for the new objective.
Joint targets use a fixed 0.25-radian action scale and the existing joint,
speed, and torque limits. The policy runs at 50 Hz with 5 ms physics substeps.

The new actor's velocity and heading inputs use simulator state. Hardware use
requires a suitable state estimator and actuator/sensor calibration.

## Train and monitor

From the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_campaign \
    logs/mujoco_rl/command_walk_v1 --hours 72 --publish
```

Choose a new output directory. The runner launches seeds 44 and 45, each with
eight environments and at most ten million steps. The 72-hour limit is shared
wall time; six hours are reserved for qualification and rendering. `--publish`
commits a qualified checkpoint, normalization, results, and annotated videos to
`main` and pushes to `fork`. Publication requires `main`, an empty staging area,
and an unused `checkpoints/command_walk_v1/` destination.

The campaign's `status.json` lists child logs, PIDs, stages, and recent progress.
Each seed directory contains `updates.jsonl`, `episodes.jsonl`, `evaluations/`,
`reviews/`, `best.json`, `latest.json`, and immutable checkpoint bundles.
The runner stops a branch on numerical instability, a behavioral plateau,
qualification readiness, or its budget limit. A completed process does not
establish a learned skill.

To resume a saved bundle into a new output directory:

```bash
.venv-mujoco/bin/python -m mujoco_rl.command_walk_train \
    /path/to/checkpoints/000999424_resume/policy.zip \
    --output logs/mujoco_rl/command_walk_resumed --seed 44 --hours 4
```

The optimizer, reward statistics, and curriculum persist; simulator episodes
restart. Pass the original `--config` if the initial run used nondefault settings.
Manual resumes must respect the original campaign budget.

## Curriculum and acceptance

Stages progress from gentle commands (0.1–0.2 m/s, yaw magnitude 0.15–0.3 rad/s),
to the full command range, then half and full physics randomization. Every stage
samples all seven command families, including stopping and mixed sequences.
Starting poses and headings vary even before physics randomization is enabled.
Velocity disturbances apply during movement commands. Stop segments retain
randomized physics and sensors but receive no new velocity impulses during the
controlled braking/hold measurement.
Two consecutive development panels with at least 80% success in every family
advance a stage. Evaluation runs every 100,000 steps on 50 cases per family.

Every million steps, continuation requires improved pass rates or at least a
10% reduction in normalized violations of unmet criteria without reduced survival.
Improving speed accuracy after the speed criterion already passes does not count.
The best development checkpoint is retained separately from the latest checkpoint.

Qualification requires at least 90% success in each family on development and
separate holdout cases (50 per family). Development starts at seed 1001; first
holdout starts at 10051, with disjoint family offsets. Failed holdouts become
regression panels, and later candidates receive fresh holdout seeds.

After two seconds of settling per command segment, speed error must be at most
0.05 m/s, yaw-rate error 0.10 rad/s, heading RMS error 10°, and peak heading error
20°. Straight walking must stay within 0.25 m sideways throughout the segment and
make the commanded forward progress. Stationary turns must stay within 0.25 m
of their starting position. Turning direction and unwrapped heading reject extra
revolutions. Stopping must settle within two seconds and hold below 0.05 m/s and
0.10 rad/s for at least two seconds. All episodes must avoid falls/self-collision
and maintain mean stance-foot slip below 0.05 m/s.

## Evaluate and render

```bash
.venv-mujoco/bin/python -m mujoco_rl.command_walk_eval \
    /path/to/policy.zip --stage 3 --seeds 50 --seed-start 10051 \
    --output logs/mujoco_rl/command_walk_evaluation.json
.venv-mujoco/bin/python -m mujoco_rl.command_walk_eval \
    /path/to/policy.zip --stage 0 --family mixed --seeds 1 \
    --output logs/mujoco_rl/command_walk_demo.json \
    --video logs/mujoco_rl/command_walk_demo.mp4
```

Videos display commanded speed/turn rate, heading error, and lateral displacement;
JSON sidecars include the full path and individual failure reasons. Legacy
navigation and running checkpoints use their original evaluators.

Run focused regressions with:

```bash
.venv-mujoco/bin/python -m pytest -q tests/test_command_walk.py tests/test_running_curriculum.py
```
