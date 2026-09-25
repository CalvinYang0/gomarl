#!/usr/bin/env python3
"""Incrementally upload every local W&B run created on/after a date.

Unlike the running-job synchronizer, discovery is based on offline-run
directories, so completed jobs that have left squeue are included.  Nothing
is deleted or marked synced; later rounds append newly written records.
"""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys


RUN_PATTERN = re.compile(
    r"^offline-run-([0-9]{8})_[0-9]{6}-[A-Za-z0-9]+$"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True, help="inclusive YYYYMMDD")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{8}", args.since):
        raise SystemExit("--since must be YYYYMMDD")

    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    wandb_root = Path(os.environ.get(
        "WANDB_ROOT", str(runtime_root / "wandb")
    )).resolve()
    python_bin = os.environ.get(
        "PYTHON_BIN", "/home/kyang/.conda/envs/marl_cpu/bin/python"
    )
    timeout = int(os.environ.get("SYNC_TIMEOUT", "900"))

    directories = []
    for directory in wandb_root.glob("offline-run-*"):
        match = RUN_PATTERN.fullmatch(directory.name)
        if not directory.is_dir() or not match or match.group(1) < args.since:
            continue
        run_id = directory.name.rsplit("-", 1)[-1]
        if (directory / ("run-" + run_id + ".wandb")).is_file():
            directories.append(directory)
    # Active jobs are normally the newest directories. Upload newest first so
    # a backlog of already-synced historical runs cannot delay live curves.
    directories.sort(key=lambda path: path.name, reverse=True)

    if not directories:
        print("No W&B offline runs found on/after " + args.since, flush=True)
        return

    wandb_root.mkdir(parents=True, exist_ok=True)
    uploaded = 0
    failed = 0
    with (wandb_root / ".gomarl-sync-once.lock").open("a") as lock:
        print("Waiting for the shared W&B sync lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for directory in directories:
            print("SYNC: " + str(directory), flush=True)
            try:
                result = subprocess.run(
                    [
                        python_bin,
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
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                print("TIMEOUT: " + str(directory), file=sys.stderr, flush=True)
                failed += 1
                continue
            if result.returncode == 0:
                uploaded += 1
            else:
                failed += 1

    print(
        "Recent W&B sync result: discovered={} uploaded={} failed={}".format(
            len(directories), uploaded, failed
        ),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
