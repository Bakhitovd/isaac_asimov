"""Show campaign progress and the latest per-motion development results."""

import argparse
import json
import time
from pathlib import Path

from .command_walk import FAMILIES


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def show(root):
    campaign = read(root / "status.json")
    print(
        f"Campaign: {campaign.get('state', 'waiting for startup')} | elapsed {campaign.get('elapsed_hours', 0):.2f} h"
    )
    if campaign.get("reason"):
        print(campaign["reason"])
    if campaign.get("source"):
        print(f"Source: {campaign.get('selected_source', campaign['source'])}")
    if campaign.get("family") and campaign.get("state") in {"selecting_source", "comparing_pilots", "qualifying"}:
        print(
            f"Evaluator: {campaign['family']}, case {campaign.get('episode', 0)+1}, step {campaign.get('episode_step', 0)}"
        )
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(("pilot_", "extended_")):
            continue
        status = read(folder / "status.json")
        latest = status.get("evaluation")
        if not latest:
            latest = read(folder / "report.json").get("final")
        print(
            f"\n{folder.name}: {status.get('state', 'starting')} / {status.get('phase', status.get('stop_reason', ''))}"
            f" | +{status.get('additional_steps', 0):,} steps | level {status.get('level', '?')}"
        )
        if latest:
            print(f"{'Motion':24} {'Nominal':>9} {'Current':>9}")
            for family in FAMILIES:
                a = latest["nominal"]["families"][family]["pass_rate"]
                b = latest["current"]["families"][family]["pass_rate"]
                print(f"{family:24} {a:9.0%} {b:9.0%}")
            count = latest.get("count", latest["nominal"]["families"]["forward"]["count"])
            print(f"{count} cases per motion; development results, not final qualification.")
    print("\nLogs and immutable checkpoint/statistics pairs are under this campaign directory.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        show(args.root)
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
