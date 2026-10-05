"""Run bounded recovery comparisons from one checkpoint and fixed demonstrations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def read_records(path: Path) -> list[dict]:
    # A trainer may be midway through appending the final line when polled.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def summarize(directory: Path) -> dict:
    result = {"directory": str(directory)}
    curriculum = directory / "curriculum.jsonl"
    if curriculum.exists():
        rows = read_records(curriculum)
        if rows:
            last = rows[-1]
            result.update(steps=last["training_steps"], level=last["recovery_level"],
                          next_level=last["next_recovery_level"], abort_reason=last.get("abort_reason"),
                          directions=last["recovery_probe"]["group_rates"])
            for role in ("probe", "current_validation", "stress_validation", "retention"):
                panel = last.get(f"recovery_{role}")
                if panel:
                    result[role] = panel["pass_rate"]
    updates = directory / "updates.jsonl"
    if updates.exists():
        rows = read_records(updates)
        if rows:
            result["steps"] = max(result.get("steps", 0), rows[-1]["training_steps"])
            result["last_update"] = {key: rows[-1][key] for key in (
                "ppo_kl", "post_imitation_kl", "ppo_action_change_rms", "net_action_change_rms",
                "imitation_alignment")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True,
                        help="JSON list of name, learning_rate, imitation_weight, imitation_updates, leg_std, seed")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=196608)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=768)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--critic-warmup-steps", type=int, default=0)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    names = [setting["name"] for setting in settings]
    if (args.steps < 1 or args.workers < 1 or args.rollout_steps < 1 or args.parallel < 1
            or args.critic_warmup_steps < 0
            or len(names) != len(set(names)) or any(not re.fullmatch(r"[\w-]+", name) for name in names)):
        parser.error("Invalid budgets or variant names")
    with zipfile.ZipFile(args.checkpoint) as archive:
        start = json.loads(archive.read("data"))["num_timesteps"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pending = list(settings)
    active = {}
    completed = {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        while pending or active:
            while pending and len(active) < args.parallel:
                setting = pending.pop(0)
                directory = args.run_dir / setting["name"]
                if directory.exists():
                    raise FileExistsError(f"Refusing to reuse comparison directory {directory}")
                directory.mkdir()
                command = [sys.executable, "-u", "-m", "mujoco_rl.skill_train", "--stage", "recover",
                           "--resume", str(args.checkpoint.resolve()), "--run-dir", str(directory.resolve()),
                           "--workers", str(args.workers), "--rollout-steps", str(args.rollout_steps),
                           "--torch-threads", "1", "--max-steps", str(start + args.steps),
                           "--seed", str(setting.get("seed", 44)), "--recovery-lr", str(setting["learning_rate"]),
                           "--recovery-critic-warmup-steps", str(args.critic_warmup_steps),
                           "--recovery-imitation-weight", str(setting["imitation_weight"]),
                           "--recovery-imitation-updates", str(setting["imitation_updates"]),
                           "--recovery-leg-std", str(setting.get("leg_std", 0.05)),
                           "--recovery-demonstrations", str(args.demonstrations.resolve())]
                (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                handle = (directory / "training.log").open("w")
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[setting["name"]] = (process, handle, directory)
                print(f"[comparison] started {setting['name']}: pid={process.pid}, budget={args.steps:,}", flush=True)
            report = {"checkpoint": str(args.checkpoint), "start_steps": start,
                      "additional_steps_per_variant": args.steps, "settings": settings, "variants": dict(completed)}
            for name, (process, handle, directory) in list(active.items()):
                state = summarize(directory)
                state["additional_steps"] = state.get("steps", start) - start
                state["running"] = process.poll() is None
                report["variants"][name] = state
                if not state["running"]:
                    handle.close()
                    state["exit_code"] = process.returncode
                    completed[name] = state
                    del active[name]
                    print(f"[comparison] finished {name}: {json.dumps(state)}", flush=True)
                else:
                    print(f"[comparison] {name}: {state['additional_steps']:,} steps, "
                          f"success={state.get('probe', 'pending')}, retention={state.get('retention', 'pending')}",
                          flush=True)
            (args.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if active:
                time.sleep(20)
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            handle.close()


if __name__ == "__main__":
    main()
