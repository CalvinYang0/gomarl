#!/usr/bin/env python3
"""Final-sync exactly the three completed 10M 5m6m obs baseline runs.

No training submissions or file deletions. Abort if a selected job is active
or any local offline run is missing. --discover-only prints selected paths.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from plot_linear_directkl_four_model_3seeds import local_run_index

NAMES = tuple("smac_5m6m_linear_obs_baseline_10m_s{}_valuediag".format(s) for s in (1, 2, 3))


def select_paths(index):
    return {name: max(index[name], key=lambda path: path.parent.name).parent
            for name in NAMES if index.get(name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    wandb_root = Path(os.environ.get("WANDB_ROOT", str(root / "wandb")))
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    active = set(subprocess.check_output(["squeue", "-u", user, "-h", "-o", "%j"], text=True).splitlines())
    if active.intersection(NAMES):
        raise SystemExit("Selected obs jobs still active; refusing final-sync: " + str(active.intersection(NAMES)))
    found = select_paths(local_run_index(wandb_root, set(NAMES)))
    for name in NAMES:
        print("{}: {}".format(name, found.get(name, "MISSING local offline run")), flush=True)
    if len(found) != 3:
        raise SystemExit("Need all three local offline runs; nothing uploaded or deleted")
    if args.discover_only:
        return
    with (wandb_root / ".gomarl-sync-once.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in NAMES:
            print("Final syncing: " + name, flush=True)
            subprocess.run([
                sys.executable, "-m", "wandb", "sync", "--append",
                "--include-offline", "--include-synced", "--no-mark-synced",
                "--skip-console", str(found[name]),
            ], check=True, timeout=1200)
    print("5m6m obs final sync complete: 3/3", flush=True)


if __name__ == "__main__":
    main()
