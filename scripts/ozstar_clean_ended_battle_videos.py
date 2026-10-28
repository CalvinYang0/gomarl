#!/usr/bin/env python3
"""Remove only receipt-owned old battle-video leaves; no cloud calls or uploads.

Preview by default. --apply is irreversible, including video episode JSONs.
Keep the new soft/refresh cohort, unknown directories, Sacred/diagnostics/logs.
Requires fresh terminal Slurm proof, an empty queue, no writers on either login
node, unchanged snapshots and the existing shared sync locks.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

import ozstar_clear_all_job_data as safe
from ozstar_clean_uploaded_completed_runs import job_finished

KEEP_GROUP = 'soft_onefill_refresh_three_maps_10m_18jobs_20261028'
LEAF = re.compile(r'seed_([0-9]+)_([0-9a-f]{12})')
STAMP = re.compile(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}')


def receipt_index(logs):
    index = {}
    for path in sorted(logs.glob('*.json')):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or not isinstance(record.get('plans'), list):
            continue
        group = record.get('group')
        if not isinstance(group, str) or group == KEEP_GROUP:
            continue
        submitted = {}
        for field in ('submitted', 'retained'):
            values = record.get(field, {})
            if isinstance(values, dict):
                for name, job in values.items():
                    job = str(job).split(';')[0]
                    if re.fullmatch(r'[0-9]+', job):
                        submitted.setdefault(name, set()).add(job)
        for plan in record['plans']:
            if not isinstance(plan, dict):
                continue
            name, seed = plan.get('job_name'), plan.get('seed')
            if (not isinstance(name, str) or type(seed) is not int or seed <= 0 or
                    name not in submitted or plan.get('exports', {}).get('GROUP_NAME') != group):
                continue
            digest = hashlib.sha256(name.encode()).hexdigest()[:12]
            index.setdefault((group, seed, digest), {}).setdefault(name, set()).update(submitted[name])
    return index


def inventory(videos, index, finished):
    selected = []
    for token in sorted(videos.iterdir()):
        group, sep, stamp = token.name.rpartition('__')
        if (not sep or not STAMP.fullmatch(stamp) or group == KEEP_GROUP or
                token.is_symlink() or not token.is_dir()):
            print('KEEP unknown/current video token: ' + str(token), flush=True)
            continue
        safe.ordinary_directory(token)
        for leaf in sorted(token.iterdir()):
            match = LEAF.fullmatch(leaf.name)
            owners = index.get((group, int(match.group(1)), match.group(2)), {}) if match else {}
            if leaf.is_symlink() or not leaf.is_dir() or len(owners) != 1:
                print('KEEP unknown video leaf: ' + str(leaf), flush=True)
                continue
            name, jobs = next(iter(owners.items()))
            if not jobs or not all(finished(job) for job in sorted(jobs)):
                print('KEEP nonterminal/unknown video owner: ' + name, flush=True)
                continue
            selected.append((token, leaf, name, jobs))
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    videos = safe.RUNTIME/'results/test_battle_videos'
    logs = safe.RUNTIME/'ozstar_logs'
    for root in (safe.RUNTIME, safe.REPO, safe.RUNTIME/'results', videos, logs):
        safe.ordinary_directory(root)
    safe.safety_state()
    before = safe.command(['quota', '-v'])
    print('BEFORE QUOTA:\n' + before, flush=True)
    print('NO CLOUD CHECK: only old receipt-owned video directories and their episode JSONs; no backup', flush=True)
    with ExitStack() as stack:
        if args.apply:
            for root in (safe.RUNTIME/'wandb', safe.REPO/'wandb'):
                if not root.exists():
                    continue
                safe.ordinary_directory(root)
                for name in safe.LOCKS:
                    fd = os.open(str(root/name), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    lock = stack.enter_context(os.fdopen(fd, 'a'))
                    info = os.fstat(lock.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                        raise RuntimeError('Unexpected lock owner/type')
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            safe.safety_state()
        index = receipt_index(logs)
        cache = {}
        def finished(job):
            if job not in cache:
                cache[job] = job_finished(job, safe.REPO)
            return cache[job]
        planned = []
        for base, path, name, jobs in inventory(videos, index, finished):
            state, size = safe.snapshot(path, base)
            if state and time.time()-max(v[2] for v in state.values())/1e9 < 300:
                print('KEEP recently modified video leaf: ' + str(path), flush=True)
                continue
            planned.append((base, path, name, jobs, state, size))
            print('TARGET {:.3f} GiB / {} entries / jobs={}: {}'.format(
                size/2**30, len(state), ','.join(sorted(jobs)), path), flush=True)
        print('PLAN: {} video leaves, {:.3f} GiB allocated, {} entries (not exact quota recovery)'.format(
            len(planned), sum(p[5] for p in planned)/2**30, sum(len(p[4]) for p in planned)), flush=True)
        if not args.apply:
            print('PREVIEW ONLY: --apply deletes these local videos and video episode JSONs permanently', flush=True)
            return
        audit_path = logs/'cleanup_ended_battle_videos_{}_{}.jsonl'.format(time.strftime('%Y%m%d_%H%M%S'), os.getpid())
        with audit_path.open('x') as audit:
            safe.audit_event(audit, dict(action='begin_old_video_cleanup', time=time.time(),
                backup_verified=False, before_quota=before, protected_group=KEEP_GROUP,
                targets=[str(p[1]) for p in planned]))
            for base, path, name, jobs, state, _ in planned:
                safe.safety_state()
                if not all(job_finished(job, safe.REPO) for job in sorted(jobs)):
                    raise RuntimeError('Job state changed; retain remaining videos')
                if path.parents[1] != videos:
                    raise RuntimeError('Video target escaped exact category')
                safe.remove_target(base, path, state, audit)
            after = safe.command(['quota', '-v'])
            safe.audit_event(audit, dict(action='complete_old_video_cleanup', time=time.time(),
                removed_leaves=len(planned), after_quota=after))
        print('AFTER QUOTA:\n'+after, flush=True)
        print('Removed {} old video leaves; audit: {}'.format(len(planned), audit_path), flush=True)
        print('KEEP: Sacred, gate heatmaps/JSON, figures, job logs/manifests, Conda, new cohort and unknown data', flush=True)


if __name__ == '__main__':
    main()
