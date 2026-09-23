#!/usr/bin/env python3
"""Upload all 12 completed paper-QMIX offline W&B runs.

Unlike the periodic uploader, discovery is log-based and therefore also works
after jobs have left ``squeue``.  Both stdout and stderr are inspected because
different W&B versions print the offline directory to different streams.
"""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys


SCENES = ("grf_counter", "grf_pass", "smac_5m6m", "smac_mmm2")
SEEDS = (1, 2, 3)
OFFLINE_PATTERN = re.compile(
    r"(/[^\s]+/wandb/offline-run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9]+)"
)


def expected_runs():
    return tuple(
        "{}_paper_qmix_5m_s{}".format(scene, seed)
        for scene in SCENES
        for seed in SEEDS
    )


def discover(log_root):
    found = {}
    for run_name in expected_runs():
        paths = []
        for suffix in ("out", "err"):
            paths.extend(sorted(log_root.glob(run_name + "_*." + suffix)))
        candidates = []
        for path in paths:
            text = path.read_text(errors="ignore")
            candidates.extend(OFFLINE_PATTERN.findall(text))
        existing = [Path(value) for value in candidates if Path(value).is_dir()]
        if existing:
            # Slurm logs are unique for this suite; choose the latest matching
            # directory if a retried job wrote more than one W&B startup line.
            found[run_name] = sorted(set(existing), key=str)[-1]
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    log_root = Path(os.environ.get(
        "LOG_ROOT", str(runtime_root / "ozstar_logs")
    )).resolve()
    wandb_root = Path(os.environ.get(
        "WANDB_ROOT", str(runtime_root / "wandb")
    )).resolve()

    found = discover(log_root)
    for run_name in expected_runs():
        directory = found.get(run_name)
        print("{}: {}".format(
            "FOUND" if directory else "MISSING",
            "{} -> {}".format(run_name, directory or "no offline directory"),
        ), flush=True)
    if len(found) != 12:
        raise SystemExit(
            "Discovered {} of 12 QMIX runs; nothing uploaded".format(len(found))
        )
    if args.discover_only:
        print("Discovery complete: 12/12", flush=True)
        return

    wandb_root.mkdir(parents=True, exist_ok=True)
    lock_path = wandb_root / ".gomarl-sync-once.lock"
    with lock_path.open("a") as lock:
        print("Waiting for the periodic W&B sync lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for run_name in expected_runs():
            directory = found[run_name]
            print("SYNC: {} -> {}".format(run_name, directory), flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wandb",
                    "sync",
                    "--append",
                    "--include-offline",
                    "--include-synced",
                    "--no-mark-synced",
                    "--skip-console",
                    str(directory),
                ],
                check=True,
            )
    print("QMIX W&B sync complete: 12/12", flush=True)


if __name__ == "__main__":
    main()
