# Command walking v2

V2 continues a saved command policy on the Loudbox CPU. It adds measured pivot/hold displacement,
progressive delays, nominal retention, and bounded experiments. The acceptance rules in
`command_walk.assess_trace` are unchanged. Results are simulation-only.

## Why this changes the training

The v1 investigation found that a saved nominal policy passed 69/70 paired diagnostic cases,
but only 24/70 with action and sensor delays together. Removing delays from half-strength
randomization restored 64/70. The final checkpoint passed only 41/70 of the same nominal cases.
All ten diagnostic left pivots failed for drift while averaging approximately 6.53/7 reward.
See `reports/command_walk_v1_investigation.json`; these small diagnostic samples are not qualification.

## Observation and reward contract

`asimov_command_walk_v2` has 87 observations and 23 joint position targets at 50 Hz.
The first 85 inputs preserve v1 semantics. Two appended inputs describe body-frame displacement
from a latched pivot/hold origin, scaled by 0.25 m. They use integrated simulated velocity with
noise. This assumes an odometry estimate: an IMU alone does not directly measure linear velocity.
A new origin is latched on entering a scripted segment or entering a live zero-forward command;
repeated live command messages do not reset the origin.

The aligned variant penalizes accumulated drift beyond a 5 cm margin, horizontal motion during
pivots, and residual linear/angular speed while stopping. Stop penalties ramp over the existing
two-second settling period. Turn rewards remain active during pivots. The control variant uses
the original reward and zero appended inputs. Both start from the same actor, preserve observation
statistics and exploration scale, and warm up a fresh critic for two rollouts. Migration pads new
input weights with zeros, preserving the original action means.

## Curriculum and retention

Named profiles start with nominal physics and half-strength variation without delays, then add
20 ms action delays with 10%, 25%, and 50% episode probability. Sensor delay follows at 10%, 25%,
and 50%. Both reach 2/3 probability; continuous variation then reaches full strength, and 40 ms
delays are introduced gradually. The final delay distribution is uniform over 0, 20, and 40 ms.
Noise, gains, mass, friction and pushes have explicit independent profile multipliers. Case,
physics, noise, odometry, and push RNG streams are separated. The same evaluation seed always
selects the same pose and command schedule across profiles.

Training mixes 30% nominal episodes with earlier and current profiles. Command sampling allocates
more simulation time to pivots and mixed commands, compensating for unequal episode durations.
Each probe includes nominal retention. Two full probes with a decline greater than ten percentage
points in any nominal family trigger checkpoint rollback, reduced learning rate, and easier
physics. Three rollbacks stop for review. Advancing requires two full 50-case panels with at least
90% success in every nominal and current-profile family. Plateau, interruption, deadline,
numerical instability, and step-budget stops retain distinct reasons and resumable reports.

## Run and inspect

```bash
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_robust_campaign \
  logs/mujoco_rl/command_walk_v2 \
  logs/mujoco_rl/command_walk_v1/seed45/checkpoints/005953536_probe/policy.zip \
  logs/mujoco_rl/command_walk_v1/seed45/checkpoints/007311360_probe/policy.zip \
  --publish
```

The coordinator compares sources on 50 nominal and 50 half-physics cases per family. It then
runs four trials: control/aligned, seeds 44/45, each with 497,664 additional steps (500,000 rounded
to complete rollouts). Two trials run concurrently, using eight environments each. Pilots hold
curriculum level 2 fixed, preserving equal difficulty. Only a variant showing improvement with
nominal retention in both seeds can receive a further six-million-step budget per seed. If
neither improves, status becomes `review_required`; the next experiment is a sensor/action
history ablation, not an automatic restart of a failed trial.

`status.json`, per-trial logs, `pilot_comparison.json`, `evaluations/`, `best.json`, and immutable
checkpoint/statistics bundles record the evidence. `--steps` is additional work, even for a
checkpoint older than ten million steps. Resuming requires the same variant and rollout shape.
Short ten-case probes use batched neural inference. Coordinator panels use four isolated CPU workers. Full panels run at least every 250,000 steps
(rounded to a probe boundary), on promising candidates, and at the end of each trial.

Final publication requires 50 development and 50 separate holdout cases per family at full
physics, plus nominal holdout retention. Failed holdouts become nominal and full-physics regression panels;
subsequent candidates get fresh holdout seeds. Qualified bundles include the policy, matching
normalization, all qualification evidence, hashes, and seven reproducible annotated videos.
Smoke runs with fewer than 50 cases cannot publish.

## Validation

`MUJOCO_GL=disable .venv-mujoco/bin/python -m pytest -q tests/test_command_walk_robust.py tests/test_command_walk.py`

Checks cover actor-preserving migration, profile-independent case generation, batched evaluation,
command slew and anchor latching, reward failure modes, per-family retention, stop causes,
additional budgets and checkpoint resume. Graphics require OSMesa; training can use
`MUJOCO_GL=disable`. Local v1 checkpoint integration tests skip when those artifacts are absent.

Watch a running campaign with:

```bash
.venv-mujoco/bin/python -m mujoco_rl.command_walk_robust_status logs/mujoco_rl/command_walk_v2 --watch
```
