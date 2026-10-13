# Trained moderate walking and turning candidates

[Download all models and evaluation records](../moderate_walk_v1_bundle.zip?raw=true).
The ZIP contains this directory, including actual model binaries, matching
normalization files, controller settings, and development/confirmation reports.
The files are stored directly in Git; Git LFS is not required.

These policies control 23 joints at 50 Hz in nominal MuJoCo simulation. Supported
requests are forward speed zero or 0.1–0.3 m/s and yaw rate zero or
±0.15–0.5 rad/s. Heading correction and the internal turn-speed reduction are
part of the controller and must be used with the network.

## Which model to download

| Directory | Artifact | Pilot improvement checks |
| --- | --- | --- |
| [original_44](original_44/) | Trained candidate, original reward, seed 44 | Passed both panels |
| [aligned_44](aligned_44/) | Trained candidate, adjusted reward, seed 44 | Passed both panels |
| [original_45](original_45/) | Trained comparison, original reward, seed 45 | Did not pass |
| [aligned_45](aligned_45/) | Trained comparison, adjusted reward, seed 45 | Did not pass |
| [baseline](baseline/) | Corrected starting policy; unchanged weights | Reference |

Start with `original_44` to inspect the candidate with the higher seed-44
confirmation result on long sequences. Compare `aligned_44` for the stronger
left-turn-exit confirmation result. Each trained candidate received 393,216
actor-training steps plus eight critic-only warmup rollouts.

**None is a qualified release.** All four preserved established motions within
the allowed tolerance, but neither reward configuration improved reliably across
both training seeds. The campaign stopped at the pilot gate; extended refinement
and final locked-holdout qualification did not run. The local campaign's exported
`controller/` remained the baseline. This package explicitly includes the trained
candidates separately.

## Recorded confirmation results

Each entry is passing episodes out of 50. The separate confirmation panel used
seed start 10051; this historical panel was not a new, unseen final holdout.

| Motion | Baseline | Original 44 | Aligned 44 | Original 45 | Aligned 45 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward | 50 | 50 | 50 | 50 | 50 |
| Walking left turn | 50 | 50 | 50 | 50 | 50 |
| Walking right turn | 50 | 50 | 50 | 50 | 50 |
| Left pivot | 48 | 49 | 49 | 48 | 48 |
| Right pivot | 50 | 50 | 50 | 49 | 49 |
| Stop | 50 | 50 | 50 | 50 | 50 |
| Short mixed commands | 50 | 50 | 50 | 50 | 49 |
| Left turn → forward | 42 | 47 | 48 | 37 | 43 |
| Right turn → forward | 50 | 50 | 50 | 50 | 50 |
| 60-second sequence | 43 | 44 | 42 | 41 | 42 |

All five braking/restart cases passed 50/50 for every controller on this panel.
Development results, every episode, failure reasons, and physical metrics are in
[results.json](results.json) and each candidate's `evaluations/` directory.
See [pilot_decision.json](pilot_decision.json) for the promotion decision and
[training_manifest.json](training_manifest.json) for training-time provenance.

## Files required together

Each model directory contains:

- `policy.zip`: Stable-Baselines3 PPO weights and saved optimizer state.
- `policy_vecnormalize.pkl`: that checkpoint's matching normalization statistics.
- `controller.json`: correction settings, range, hashes, and qualification status.
- `evaluations/development.json` and `evaluations/confirmation.json`: physical evidence.

The top-level `SHA256SUMS` covers all package files. After extracting the download:

```bash
cd moderate_walk_v1
sha256sum -c SHA256SUMS
```

## Run from a repository checkout

The source code, MuJoCo runtime, and pinned robot assets are required. From the
repository root:

```bash
git submodule update --init third_party/asimov-1
python3 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install -r mujoco_rl/requirements.txt
```

All model directories are already present after cloning this repository. To
render independent demonstrations of a trained candidate with OSMesa installed:

```bash
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-mujoco/bin/python -m mujoco_rl.command_walk_final_replay \
  mujoco_rl/checkpoints/moderate_walk_v1/original_44 \
  logs/mujoco_rl/original_44_replays
```

The script writes labelled MP4s and measured outcomes. It selects recorded
failure seeds where available. These scalar-inference replays are independent
illustrations; their numerical results can differ slightly from batched panels.
On Loudbox, also set
`LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu`.

For integration, load the whole controller directory:

```python
from mujoco_rl.command_walk_final_eval import Controller

controller = Controller("mujoco_rl/checkpoints/moderate_walk_v1/original_44")
controller.reset()
# One call per 20 ms tick. Supply 87 raw observation values before correction.
action = controller.act(raw_observation, requested_v=0.2, requested_w=0.3)
controller.close()
```

All results are simulation-only. Real-robot behavior has not been evaluated.
