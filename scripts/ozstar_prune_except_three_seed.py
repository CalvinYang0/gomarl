#!/usr/bin/env python3
"""Remove retained OzSTAR artifacts except the paper three-seed suites.

The default mode is a read-only preview.  Pass ``--execute`` to delete the
listed artifacts.  Preservation is intentionally based on the exact paper
suite run/group names used by this repository, not merely on the presence of
``s1`` in a run name (most one-seed ablations also contain that suffix).
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


EXPECTED_RUNTIME_ROOT = Path(
    "/home/kyang/gomarl-runtime/gomarl-dual-branch"
)

SCENE = r"(?:grf_(?:counter|pass)|smac_(?:5m6m|mmm2))"
THREE_SEED_RUN = re.compile(
    r"(?:^|[^a-z0-9_])(?:"
    + SCENE
    + r"_paper_(?:vdn|qmix|id_hypernet|obs_hypernet|hyperselect|"
      r"linear_singlehead)_(?:5m|10m)_s[123]"
      r"|"
    + SCENE
    + r"_linear_kl80_direct_5m_s[123]_controlled15(?:_retry1)?"
      r")(?:$|[^a-z0-9_])"
)

THREE_SEED_GROUPS = {
    "paper_main_baselines_5m_3seeds",
    "hyperselect_paper_main_10m_3seeds",
    "paper_linear_singlehead_5m_3seeds",
    "paper_linear_kl80_direct_5m_3seeds",
    # Seed 1 of these studies was submitted before the missing seeds were
    # scheduled, so its group name does not say ``3seeds``.
    "paper_linear_singlehead_5m_seed1",
    "counter_linear_kl80_qme_controlled15_s1",
    "counter_linear_worker_recovery_retries",
}


def retained_run(path):
    config = path / "files" / "config.yaml"
    try:
        text = config.read_text(errors="ignore")
    except OSError:
        return False
    return THREE_SEED_RUN.search(text.lower()) is not None


def retained_group_entry(path, retained_groups):
    return any(
        path.name == group or path.name.startswith(group + "__")
        for group in retained_groups
    )


def retained_log(path):
    name = re.sub(
        r"_[0-9]+\.(?:out|err)$", "", path.name.lower()
    )
    return THREE_SEED_RUN.search(name) is not None


def children(path):
    try:
        return list(path.iterdir())
    except FileNotFoundError:
        return []


def discover_retained_groups(sacred_root):
    """Find legacy groups containing a run that belongs to a kept study."""
    retained = set(THREE_SEED_GROUPS)
    for scene_dir in children(sacred_root):
        if not scene_dir.is_dir():
            continue
        for group_dir in children(scene_dir):
            if not group_dir.is_dir():
                continue
            for config in group_dir.glob("*/config.json"):
                try:
                    text = config.read_text(errors="ignore").lower()
                except OSError:
                    continue
                if THREE_SEED_RUN.search(text):
                    retained.add(group_dir.name)
                    break
    return retained


def deletion_plan(runtime_root):
    targets = []
    kept = []

    results_root = runtime_root / "results"
    sacred_root = results_root / "sacred"
    retained_groups = discover_retained_groups(sacred_root)

    wandb_root = runtime_root / "wandb"
    for entry in children(wandb_root):
        if entry.is_dir() and entry.name.startswith("offline-run-"):
            (kept if retained_run(entry) else targets).append(entry)
        else:
            # Caches, debug logs, lock files and latest-run symlinks are all
            # reproducible and are not experiment records.
            targets.append(entry)

    for category in ("models", "tb_logs", "battle_traces"):
        for entry in children(results_root / category):
            (
                kept if retained_group_entry(entry, retained_groups)
                else targets
            ).append(entry)

    for scene_dir in children(sacred_root):
        if not scene_dir.is_dir():
            targets.append(scene_dir)
            continue
        for group_dir in children(scene_dir):
            (
                kept if retained_group_entry(group_dir, retained_groups)
                else targets
            ).append(group_dir)

    logs_root = runtime_root / "ozstar_logs"
    for entry in children(logs_root):
        (kept if retained_log(entry) else targets).append(entry)

    # De-duplicate without resolving symlinks outside the runtime root.
    targets = sorted(set(targets), key=lambda item: str(item))
    kept = sorted(set(kept), key=lambda item: str(item))
    return targets, kept, retained_groups


def allocated_bytes(path):
    try:
        if path.is_symlink() or path.is_file():
            return path.lstat().st_blocks * 512
    except OSError:
        return 0
    total = 0
    try:
        for root, directories, files in os.walk(path):
            for name in directories + files:
                candidate = Path(root) / name
                try:
                    total += candidate.lstat().st_blocks * 512
                except OSError:
                    pass
    except OSError:
        pass
    return total


def human_size(value):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--runtime-root", default=str(EXPECTED_RUNTIME_ROOT)
    )
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    if runtime_root != EXPECTED_RUNTIME_ROOT:
        raise SystemExit(
            "Refusing unexpected runtime root: {}".format(runtime_root)
        )
    if not runtime_root.is_dir():
        raise SystemExit("Runtime root does not exist: {}".format(runtime_root))

    if args.execute:
        queue = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "kyang"), "-h"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if queue.returncode != 0:
            raise SystemExit("Unable to verify an empty Slurm queue")
        if queue.stdout.strip():
            raise SystemExit(
                "Refusing cleanup while Slurm jobs are active or pending"
            )

    targets, kept, retained_groups = deletion_plan(runtime_root)
    reclaimable = sum(allocated_bytes(path) for path in targets)
    kept_bytes = sum(allocated_bytes(path) for path in kept)

    print("RETAINED GROUPS:")
    for group in sorted(retained_groups):
        print("GROUP " + group)
    print("KEEP {} entries ({})".format(len(kept), human_size(kept_bytes)))
    for path in kept:
        print("KEEP " + str(path))
    print(
        "DELETE {} entries ({})".format(
            len(targets), human_size(reclaimable)
        )
    )
    for path in targets:
        print("DELETE " + str(path))

    if not args.execute:
        print("PREVIEW ONLY: rerun with --execute to delete", file=sys.stderr)
        return

    for path in targets:
        remove(path)
    print("Deleted {} entries; reclaimed approximately {}".format(
        len(targets), human_size(reclaimable)
    ))


if __name__ == "__main__":
    main()
