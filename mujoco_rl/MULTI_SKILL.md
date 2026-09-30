# Flat-ground multi-skill training

`skill_env.py` adds five 23-joint MuJoCo tasks to the published full-body
walking baseline: arbitrary-direction point-to-point travel, squat, recovery
from front/back/side falls, airborne running, and up/forward jumps. All runs
use the Loudbox CPU and the pinned `third_party/asimov-1` model. The original
walking checkpoint remains usable through `full_body_eval.py`.

From the repository root, after installing `mujoco_rl/requirements.txt` and
initializing the robot submodule, train one gated milestone at a time:

```bash
.venv-mujoco/bin/python -u -m mujoco_rl.skill_train \
    --stage pipeline --workers 24 --hours-per-stage 10 \
    --run-dir logs/mujoco_rl/multi_skill_v1
```

The pipeline begins with the published walking policy. It trains navigation,
squat, recovery, running, and jumping in that order. Each stage evaluates 50
fixed development seeds every 100,000 steps. A development pass triggers a
50-seed holdout evaluation; later attempts use fresh holdout seeds. The next
stage starts only after both sets reach 80% success. Recovery must also pass
at least 75% for each lying orientation, and jumping at least 75% for each
variant. Running must retain 80% navigation success. If a stage exhausts its
time budget, the pipeline stops and saves its best and final checkpoints.

For a single-stage refinement, pass `--stage nav|squat|recover|run|jump`,
`--warm-start PATH` or `--resume PATH`, and a new `--run-dir`. Every checkpoint
requires the matching `*_vecnormalize.pkl` file. Training runs store source
snapshots, settings, checkpoints, and per-seed evaluation records under their
stage directory. Inspect a checkpoint with:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_eval \
    logs/mujoco_rl/multi_skill_v1/nav/accepted.zip --skill nav --seeds 50
```

After all five skills pass, `skill_supervisor.py` loads the accepted policies.
It supports `go_to`, `run_to`, `squat`, `jump_up`, and `jump_forward`; a fall
switches to recovery and resumes a pending travel target. Evaluate chained
commands and an injected fall with:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_supervisor \
    logs/mujoco_rl/multi_skill_v1 --seeds 50
```

The waypoint controller uses exact MuJoCo position and orientation. Arm
support contacts are local collision overlays for recovery. These are
simulation tools, not calibrated hardware behavior. A physical robot would
need localization, actuator validation, and staged safety trials before using
any of these policies.
