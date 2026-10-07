#!/usr/bin/env python3
"""Append individual offline W&B runs for currently running 5m6m ID jobs.

No figures, analysis runs, artifacts, training submission, cancellation or
deletion. Final-sync completed jobs separately; these are RUNNING-only updates.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from plot_linear_directkl_four_model_3seeds import local_run_index

NAMES = tuple("smac_5m6m_id_baseline_10m_s{}_valuediag".format(s) for s in (1, 2, 3))


def select_paths(index, running_names):
    return {name: max(index[name], key=lambda path: path.parent.name).parent
            for name in NAMES if name in running_names and index.get(name)}


def main():
    root = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    wandb_root = Path(os.environ.get("WANDB_ROOT", str(root / "wandb")))
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    running = set(subprocess.check_output(
        ["squeue", "-u", user, "-t", "RUNNING", "-h", "-o", "%j"], text=True,
    ).splitlines()).intersection(NAMES)
    if not running:
        print("No running 5m6m ID jobs; nothing synced.", flush=True)
        return
    index = local_run_index(wandb_root, running)
    found = select_paths(index, running)
    failures = []
    for name in sorted(running - set(found)):
        print("MISSING local offline data: " + name, flush=True)
        failures.append(name)
    with (wandb_root / ".gomarl-sync-once.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, directory in found.items():
            print("SYNC individual training run: {} -> {}".format(name, directory), flush=True)
            try:
                subprocess.run([
                    sys.executable, "-m", "wandb", "sync", "--append",
                    "--include-offline", "--include-synced", "--no-mark-synced",
                    "--skip-console", str(directory),
                ], check=True, timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print("SYNC failed; retry next cycle: {}: {}".format(name, exc), flush=True)
                failures.append(name)
    if failures:
        raise SystemExit("Some running runs were not synced: " + ", ".join(failures))
    print("Individual running-job sync complete: {}/{}".format(len(found), len(running)), flush=True)


if __name__ == "__main__":
    main()
