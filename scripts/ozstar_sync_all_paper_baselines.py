#!/usr/bin/env python3
"""Upload all paper-baseline W&B runs, including jobs no longer in squeue."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys


SCENES = ("grf_counter", "grf_pass", "smac_5m6m", "smac_mmm2")
METHODS = ("qmix", "vdn", "id_hypernet", "obs_hypernet")
SEEDS = (1, 2, 3)
OFFLINE_PATTERN = re.compile(
    r"(/[^\s]+/wandb/offline-run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9]+)"
)


def expected_runs(methods):
    return tuple(
        "{}_paper_{}_5m_s{}".format(scene, method, seed)
        for method in methods
        for scene in SCENES
        for seed in SEEDS
    )


def directory_text(directory):
    chunks = []
    for child_root in (directory / "files", directory / "logs"):
        if not child_root.is_dir():
            continue
        for path in child_root.rglob("*"):
            if path.is_file():
                chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def discover(log_root, wandb_root, run_names):
    found = {}
    for run_name in run_names:
        log_paths = []
        for suffix in ("out", "err"):
            log_paths.extend(sorted(log_root.glob(run_name + "_*." + suffix)))
        candidates = []
        for log_path in log_paths:
            candidates.extend(OFFLINE_PATTERN.findall(
                log_path.read_text(errors="ignore")
            ))
        directories = [
            Path(candidate) for candidate in candidates
            if Path(candidate).is_dir()
        ]
        if directories:
            found[run_name] = sorted(set(directories), key=str)[-1]
    missing = set(run_names) - set(found)
    if not missing:
        return found

    offline_directories = sorted(
        wandb_root.glob("offline-run-*"), key=str, reverse=True
    )
    for directory in offline_directories:
        # Completed jobs may no longer be visible through Slurm, and some W&B
        # versions omit the local directory from their console output.  Search
        # each offline directory once as a name-based fallback.
        text = directory_text(directory)
        matched = {run_name for run_name in missing if run_name in text}
        if not matched:
            for path in directory.glob("run-*.wandb"):
                payload = path.read_bytes()
                matched.update(
                    run_name for run_name in missing
                    if run_name.encode() in payload
                )
        for run_name in matched:
            found[run_name] = directory
        missing.difference_update(matched)
        if not missing:
            break
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    methods = tuple(args.methods)
    if len(set(methods)) != len(methods):
        raise SystemExit("Duplicate method in --methods")

    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    log_root = Path(os.environ.get(
        "LOG_ROOT", str(runtime_root / "ozstar_logs")
    )).resolve()
    wandb_root = Path(os.environ.get(
        "WANDB_ROOT", str(runtime_root / "wandb")
    )).resolve()

    run_names = expected_runs(methods)
    found = discover(log_root, wandb_root, run_names)
    for run_name in run_names:
        directory = found.get(run_name)
        print("{}: {} -> {}".format(
            "FOUND" if directory else "MISSING",
            run_name,
            directory or "no offline directory",
        ), flush=True)
    if len(found) != len(run_names):
        raise SystemExit(
            "Discovered {} of {} selected runs; nothing uploaded".format(
                len(found), len(run_names)
            )
        )
    if args.discover_only:
        print("Discovery complete: {}/{}".format(
            len(found), len(run_names)
        ), flush=True)
        return

    wandb_root.mkdir(parents=True, exist_ok=True)
    with (wandb_root / ".gomarl-sync-once.lock").open("a") as lock:
        print("Waiting for the periodic W&B sync lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for run_name in run_names:
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
    print("Paper-baseline W&B sync complete: {}/{}".format(
        len(found), len(run_names)
    ), flush=True)


if __name__ == "__main__":
    main()
