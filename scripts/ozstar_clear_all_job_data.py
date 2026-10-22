#!/usr/bin/env python3
"""Explicit whole-project local experiment-data cleanup; preview by default.

Only the hardcoded job-data directories below are eligible. Never delete code,
Conda, Slurm logs/manifests, cloud runs, data roots or shared lock inodes. Requires
an empty Slurm queue, no known writers on either login node and all sync locks.
No backup/upload, process termination, cancellation or training submission.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import time

RUNTIME = Path('/home/kyang/gomarl-runtime/gomarl-dual-branch')
REPO = Path('/home/kyang/code/gomarl-dual-branch')
BASES = tuple(RUNTIME / name for name in (
    'results', 'wandb', 'figures', 'grouping_backfill', 'qcore_upload',
    'td_audit_upload', 'wandb_cache', 'wandb_data',
)) + tuple(REPO / name for name in ('results', 'wandb', 'figures'))
LOCKS = ('.gomarl-calibration-sync-loop.lock', '.gomarl-sync-loop.lock', '.gomarl-sync-once.lock')
NODES = ('farnarkle1', 'farnarkle2')


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=25).strip()


def ordinary_directory(path):
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise RuntimeError('Expected exact non-symlink directory: ' + str(path))


def safety_state():
    if command(['id', '-un']) != 'kyang':
        raise RuntimeError('This entry point is only for the explicitly authorized kyang directories')
    rows = command(['squeue', '-u', 'kyang', '-h', '-o', '%A|%200j'])
    if rows:
        raise RuntimeError('Slurm queue is not empty; no automatic cancellation:\n' + rows)
    local = socket.gethostname().split('.')[0]
    for node in NODES:
        args = ['ps', '-ww', '-u', 'kyang', '-o', 'pid=,args=']
        if node != local:
            args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', node,
                    'ps -ww -u kyang -o pid=,args=']
        output = command(args)  # Any failed node/process check aborts.
        writers = []
        for line in output.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                raise RuntimeError('Unrecognized process listing on ' + node)
            pid, argv = int(fields[0]), fields[1]
            if node == local and pid == os.getpid():
                continue
            if (str(RUNTIME) in argv or str(REPO) in argv or re.search(
                    r'ozstar_\S*(?:sync|upload|clean|clear)|plot_\S*\.py|'
                    r'wandb(?:-core|\s+sync)|grouping_\S*backfill|src/main\.py|ffmpeg', argv)):
                writers.append(line)
        if writers:
            raise RuntimeError('Possible writer/cleanup on {} (not killed):\n{}'.format(node, '\n'.join(writers)))


def reserved(path):
    # Preserve all lock-looking entries, not merely the three known locks.
    return path.name.endswith('.lock') or path.name in LOCKS


def inventory(bases):
    return [(base, child) for base in bases for child in sorted(base.iterdir()) if not reserved(child)]


def snapshot(path, base):
    if path.parent != base or reserved(path):
        raise RuntimeError('Refusing broad/reserved target: ' + str(path))
    ordinary_directory(base)
    device = base.stat().st_dev
    entries, size = {}, 0
    def add(item):
        nonlocal size
        info = item.lstat()
        if info.st_uid != os.getuid() or info.st_dev != device:
            raise RuntimeError('Foreign owner/device: ' + str(item))
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise RuntimeError('Special file: ' + str(item))
        entries[str(item.relative_to(path))] = (
            info.st_mode, info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev, info.st_uid)
        size += info.st_blocks * 512
    add(path)
    if path.is_dir() and not path.is_symlink():
        def onerror(error):
            raise error
        for parent, dirs, files in os.walk(path, followlinks=False, onerror=onerror):
            for name in dirs + files:
                add(Path(parent) / name)
    return entries, size


def audit_event(audit, record):
    audit.write(json.dumps(record) + '\n')
    audit.flush()
    os.fsync(audit.fileno())  # Failed audit => no next deletion.


def remove_target(base, path, previous, audit):
    if snapshot(path, base)[0] != previous:
        raise RuntimeError('Data changed since inventory; retain remaining targets: ' + str(path))
    audit_event(audit, dict(time=time.time(), action='delete_local', target=str(path), entries=len(previous)))
    if path.is_symlink() or path.is_file():
        path.unlink()  # Unlink a link itself, never its destination.
    else:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError('This Python lacks symlink-resistant rmtree; no recursive deletion')
        shutil.rmtree(path)
    print('DELETED ' + str(path), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Permanently delete ALL listed local experiment data, without backup')
    args = parser.parse_args(argv)
    for root in (RUNTIME, REPO, RUNTIME / 'ozstar_logs'):
        ordinary_directory(root)
    # Do not let a pre-existing symlink turn a missing data root into a target.
    bases = []
    for base in BASES:
        if base.is_symlink():
            raise RuntimeError('Symlink data root; no deletion: ' + str(base))
        if base.exists():
            ordinary_directory(base)
            bases.append(base)
    safety_state()
    before = command(['quota', '-sv'])
    print('BEFORE QUOTA:\n' + before, flush=True)
    with ExitStack() as stack:
        if args.apply:
            for base in (RUNTIME / 'wandb', REPO / 'wandb'):
                if not base.exists():
                    continue
                for name in LOCKS:
                    path = base / name
                    if path.is_symlink():
                        raise RuntimeError('Symlink lock: ' + str(path))
                    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    lock = stack.enter_context(os.fdopen(descriptor, 'a'))
                    info = os.fstat(lock.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                        raise RuntimeError('Unexpected lock type/owner: ' + str(path))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            safety_state()
        planned = []
        for base, path in inventory(bases):
            state, size = snapshot(path, base)
            planned.append((base, path, state, size))
            print('TARGET {:.3f} GiB, {} entries: {}'.format(size/2**30, len(state), path), flush=True)
        print('PLAN: {} targets, {:.3f} GiB allocated, {} entries (not exact quota recovery)'.format(
            len(planned), sum(p[3] for p in planned)/2**30, sum(len(p[2]) for p in planned)), flush=True)
        print('KEEP: code, Conda, all Slurm logs/receipts, audits, root directories and lock inodes', flush=True)
        if not args.apply:
            print('PREVIEW ONLY. --apply permanently deletes the listed data including the last 18 partial runs.', flush=True)
            return
        audit_path = RUNTIME / 'ozstar_logs' / ('cleanup_all_job_data_{}_{}.jsonl'.format(
            time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
        deleted = 0
        with audit_path.open('x') as audit:
            audit_event(audit, dict(time=time.time(), action='begin_all_job_data_cleanup',
                before_quota=before, targets=[str(p[1]) for p in planned], backup_verified=False))
            previous_base = None
            for base, path, state, _ in planned:
                if base != previous_base:
                    safety_state()  # Recheck both nodes before each data category.
                    previous_base = base
                if command(['squeue', '-u', 'kyang', '-h', '-o', '%A|%200j']):
                    raise RuntimeError('New Slurm job detected; stopping after {} deletions'.format(deleted))
                remove_target(base, path, state, audit)
                deleted += 1
            remaining = inventory(bases)
            if remaining:
                raise RuntimeError('New/unremoved data appeared; retain and inspect: ' + repr(remaining))
            after = command(['quota', '-sv'])
            audit_event(audit, dict(time=time.time(), action='complete', deleted_targets=deleted, after_quota=after))
        print('CLEANUP COMPLETE: {} targets removed; audit: {}'.format(deleted, audit_path), flush=True)
        print('AFTER QUOTA:\n' + after, flush=True)
        print('Local files are not backed up by this command; cloud data unchanged. No training submitted.', flush=True)


if __name__ == '__main__':
    main()
