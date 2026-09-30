# Repository Guidelines

## Project Structure & Module Organization

`source/isaac_asimov/isaac_asimov/` contains the Isaac Lab extension: robot assets, locomotion tasks, rewards, and AMP/PPO components. `scripts/rsl_rl/` provides Isaac training and playback entry points. `mujoco_rl/` contains the separate CPU MuJoCo environments, trainers, evaluators, and documented checkpoints. `tests/` holds Python tests for AMP components. Robot meshes and motion data live under `mujoco_rl/assets/` and the pinned `third_party/` submodules; do not edit submodule contents as if they were local source. Generated runs belong under ignored `logs/`.

## Build, Test, and Development Commands

- `./quick_install.sh`: set up the Isaac Sim/Isaac Lab environment on a supported NVIDIA machine.
- `git submodule update --init third_party/asimov-1`: fetch the robot model required by full-body MuJoCo.
- `python3 -m venv .venv-mujoco && .venv-mujoco/bin/python -m pip install -r mujoco_rl/requirements.txt`: set up the CPU trainer.
- `.venv-mujoco/bin/python -m mujoco_rl.full_body_eval mujoco_rl/checkpoints/full_body_v1/policy.zip --task walk --seeds 50`: evaluate the published walking policy.
- `./isaac_asimov.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 128 --headless --max_iterations 100`: run a short Isaac training smoke check.

## Coding Style & Naming Conventions

Use Python 3.10-compatible syntax, four-space indentation, and descriptive `snake_case` names for modules, functions, and variables. Keep task configuration in `source/isaac_asimov/isaac_asimov/tasks/` and MuJoCo code in `mujoco_rl/`. `pyproject.toml` configures Black at 120 columns, isort with the Black profile, and basic Pyright checks. Format changed Python files with `black` and `isort` when those tools are available.

## Testing Guidelines

Tests use `pytest`; name new files `tests/test_*.py` and test functions `test_*`. Run `python -m pytest tests` in an environment with Isaac dependencies. For MuJoCo policy changes, report the 50-seed evaluation and a separate holdout set using `--seed-start 10051`; keep the matching `*_vecnormalize.pkl` with every checkpoint.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, as in `Add full-body MuJoCo PPO standing and walking pipeline`. Keep generated logs out of Git. Include a concise change summary, relevant commands or evaluation results, and links to related issues in pull requests. Attach a replay or screenshot when motion or visuals change; state when results are simulation-only.
