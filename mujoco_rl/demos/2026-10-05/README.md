# Asimov simulation demos — October 5, 2026

Selected replays of separately trained neural policies controlling 23 joints
(legs, waist, and arms) at 50 Hz in MuJoCo. Training ran on the Loudbox CPU.
These recordings show simulation behavior; the policies have not been tested
on a physical Asimov robot.

## Watch or download

[Download the complete demo bundle](demo_bundle.zip?raw=true), extract it, and
open the numbered MP4 files in a video player. Individual clips are linked below.

| Demo | Video length | Holdout successes | Video |
| --- | --- | --- | --- |
| Slow full-body walking | 10 seconds | 43/50 (86%) | [Watch](01_slow_walking_selected_success.mp4) |
| Walking to a waypoint and stopping | 9.52 seconds | 44/50 (88%) | [Watch](02_waypoint_navigation_selected_success.mp4) |
| Repeated squatting and standing | 15 seconds | 47/50 (94%) | [Watch](03_squatting_selected_success.mp4) |
| Balance recovery from a prepared 18–25° lean | 2.28 seconds | 44/50 (88%) | [Watch](04_prepared_lean_balance_recovery_selected_success.mp4) |

Each clip shows a selected successful episode. The percentages describe separate
50-episode evaluations in the recorded simulation scenarios. Prepared-lean
recovery does not establish floor get-up. Running, full floor recovery, jumping,
and object manipulation remain unproven.

## Present the results

Show walking, navigation, and squatting first; together they take about 35
seconds. Explain that each clip uses a separate trained policy. Show the short
balance-recovery example with its starting tilt clearly identified.

For a technical presentation, include the recorded [navigation
failure](failure_examples/navigation_failure.mp4) or [squat
failure](failure_examples/squat_failure.mp4) alongside the success rates.

## Evidence and provenance

The [manifest](demo_manifest.json) records the replay seeds and original source
paths. [Evaluation records](evaluation_results/) contain development and
holdout results, including individual episodes. The original run paths refer
to local, ignored training artifacts; all demo videos and evaluation records
needed for this presentation are included here.

The published slow-walking [checkpoint and normalization
statistics](../../checkpoints/full_body_v1/README.md) are also in the repository.
The [waypoint navigation policy and matching normalization
statistics](../../checkpoints/nav/README.md) are available under `checkpoints/nav/`.
See the [multi-skill guide](../../MULTI_SKILL.md) for training and evaluation
commands and the [running report](../../RUNNING.md) for current limitations.
