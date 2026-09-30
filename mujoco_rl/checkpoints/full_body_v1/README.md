# Full-body standing and walking policy v1

This directory contains the first 23-joint PPO walking policy trained with
[`full_body_train.py`](../../full_body_train.py) in the flat-ground MuJoCo
simulator. It controls the legs, waist, and arms at 50 Hz. The policy started
from random weights in the standing stage, then continued through walking
refinements. The final walking checkpoint was selected at 8,025,456 total
training steps.

Files:

- `policy.zip`: Stable-Baselines3 PPO checkpoint.
- `policy_vecnormalize.pkl`: matching observation normalization statistics.
- `preview.mp4`: ten-second replay of passing seed 10006.
- `train_evaluation.json`: 50 randomized episodes, seeds 10001–10050.
- `holdout_evaluation.json`: separate 50 episodes, seeds 10051–10100.
- `training_config.json`: final refinement configuration and source hashes.

The policy passed 41/50 training evaluation episodes (82%) and 43/50 holdout
episodes (86%). The mean forward distances were 1.998 m and 2.019 m over ten
seconds, with mean stance-foot slip of 0.0125 m/s and 0.0134 m/s respectively.
A passing episode must stay upright, travel 1.5–2.5 m, drift less than 0.3 m
sideways, stay below 0.05 m/s mean stance-foot slip, use smooth actions, have
at most 20 contact entries per foot, and avoid self contact.

Evaluate the policy from the repository root after installing
`mujoco_rl/requirements.txt` and initializing `third_party/asimov-1`:

```bash
.venv-mujoco/bin/python -m mujoco_rl.full_body_eval \
    mujoco_rl/checkpoints/full_body_v1/policy.zip --task walk --seeds 50
```

The MP4 shows one selected passing episode; the JSON files contain results
for all evaluation seeds. This policy has only been trained and evaluated in
simulation. Motor and sensor behavior are provisional estimates and have not
been calibrated against a physical Asimov robot.
