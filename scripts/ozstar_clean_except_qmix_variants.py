#!/usr/bin/env python3
"""Delete ended local experiments except the three mixer-control families.

Preview by default. Never upload/delete cloud data or cancel jobs. Unknown
ownership/state stays. --apply is irreversible and includes Sacred histories.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

import yaml
from ozstar_clean_uploaded_completed_runs import command, job_finished, RUN_PATH_RE

ROOT = Path('/home/kyang/gomarl-runtime/gomarl-dual-branch')
REPO = Path('/home/kyang/code/gomarl-dual-branch')
KEEP_GROUP = '5m6m_qmix_mixer_input_controls_10m_9jobs'
KEEP_GROUPS = {KEEP_GROUP, 'qmix_vdn_calibration_gate_10m_3seeds_fresh20261014',
               '5m6m_score_gradient_factorial_10m_6jobs_20261014',
               '5m6m_qchange_three_ablation_10m_9jobs_20261015',
               '8m9m_ones_gate_regularization_10m_9jobs_20261015',
               'qchange_gate_ablation_10m_18jobs_20261015',
               '5m6m_ones_10m_3seeds_20261018',
               '5m6m_calibration_mean_shared_10m_3seeds_20261018',
               'calibration_mean_two_map_ablation_10m_12jobs_20261018',
               'static_gate_score_supervision_10m_12jobs_20261018',
               '5m6m_onefill_gate_10m_15jobs_20261019',
               '5m6m_residual_zerofill_10m_3jobs_20261020',
               'weight_composition_10m_15jobs_20261020'}
KEEP = re.compile(r'^smac_5m6m_qmix_state_(?:static10|group_qchange|group_tderror|group_variance)_')
LOG = re.compile(r'(.+)_([0-9]+)\.(?:out|err)$')
CATEGORIES = ('models', 'tb_logs', 'battle_traces', 'test_battle_videos',
              'policy_importance', 'hyper_obs_importance', 'slot_input_control',
              'mixer_input_control')


def children(path):
    return sorted(path.iterdir()) if path.is_dir() and not path.is_symlink() else []


def read_json(path):
    if path.is_symlink():
        return {}
    try:
        with path.open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def index_logs(root):
    names, runs = {}, {}
    for path in children(root / 'ozstar_logs'):
        match = LOG.fullmatch(path.name)
        if not match or path.is_symlink() or not path.is_file():
            continue
        name, job = match.groups()
        names.setdefault(name, set()).add(job)
        with path.open(errors='replace') as handle:
            for line in handle:
                for item in RUN_PATH_RE.finditer(line):
                    # Only a directory basename is used; never follow logged paths.
                    runs.setdefault(Path(item.group()).name, set()).add(job)
    return names, runs


def queue_state():
    user = command(['id', '-un'])
    rows = command(['squeue', '-u', user, '-h', '-o', '%A|%200j'])
    jobs, names = set(), set()
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
            raise RuntimeError('Unrecognized scheduler response; no deletion')
        jobs.add(fields[0].strip().split('_', 1)[0])
        names.add(fields[1].strip())
    return jobs, names


def inventory(root, live, finished, live_names=()):
    """Return disjoint leaf targets, preserving entire mixed artifact groups."""
    names, runs = index_logs(root)
    targets, retained, groups = [], [], {}

    def ended(name, jobs=None):
        jobs = names.get(name, set()) if jobs is None else jobs
        return (name not in live_names and bool(jobs) and not (jobs & live)
                and all(finished(j) for j in jobs))

    def record(path, name, group, closed):
        keep = bool(KEEP.match(str(name))) or group in KEEP_GROUPS
        removable = bool(name) and not keep and closed
        if group:
            groups.setdefault(group, []).append(removable)
        (targets if removable else retained).append(path)

    for scene in children(root / 'results' / 'sacred'):
        for group in children(scene):
            leaves = children(group)
            if not leaves:
                groups.setdefault(group.name, []).append(False)
            for run in leaves:
                config = read_json(run / 'config.json') if run.is_dir() and not run.is_symlink() else {}
                name = config.get('wandb_run_name', '')
                # Slurm terminal state, not stale Sacred RUNNING text, is authority.
                record(run, name, group.name, ended(name))

    for run in children(root / 'wandb'):
        if not re.fullmatch(r'(?:offline-)?run-\d{8}_\d{6}-[A-Za-z0-9]+', run.name):
            continue
        config_path = run / 'files' / 'config.yaml'
        config = {}
        if not run.is_symlink() and not config_path.is_symlink():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
            except (OSError, yaml.YAMLError):
                pass
        if not isinstance(config, dict):
            config = {}
        def value(key):
            val = config.get(key, '')
            return val.get('value', '') if isinstance(val, dict) else val
        name, group = value('wandb_run_name'), value('name')
        jobs = runs.get(run.name, set())
        # Log job names also protect retries whose config was never written.
        log_names = {n for n, ids in names.items() if ids & jobs}
        if any(KEEP.match(n) for n in log_names):
            retained.append(run)
            groups.setdefault(KEEP_GROUP, []).append(False)
        else:
            record(run, name or (next(iter(log_names)) if len(log_names) == 1 else ''),
                   group, ended(name, jobs | names.get(name, set())))

    # An active job with no identifiable config could own a shared group folder.
    mapped_jobs = set()
    for group_path in children(root / 'results' / 'sacred'):
        for group in children(group_path):
            for run in children(group):
                cfg = read_json(run / 'config.json') if not run.is_symlink() else {}
                mapped_jobs.update(names.get(cfg.get('wandb_run_name', ''), set()))
    unresolved_live = live - mapped_jobs
    for category in CATEGORIES:
        for path in children(root / 'results' / category):
            matches = [g for g in groups if path.name == g or path.name.startswith(g + '__')]
            group = max(matches, key=len) if matches else None
            safe = (group and group not in KEEP_GROUPS and all(groups[group])
                    and not unresolved_live)
            (targets if safe else retained).append(path)
    # Shared artifacts first; their ownership evidence (Sacred configs) must
    # remain available until those artifacts and W&B records have been removed.
    def order(path):
        rel = path.relative_to(root).parts
        return (2 if rel[:2] == ('results', 'sacred') else
                1 if rel[0] == 'wandb' else 0, str(path))
    return sorted(set(targets), key=order), sorted(set(retained))


def snapshot(path, root):
    """lstat only: symlinks are unlinked, never traversed or counted as targets."""
    relative = path.relative_to(root)
    if (relative.parts[0] not in ('wandb', 'results') or
            len(relative.parts) < (3 if relative.parts[0] == 'results' else 2)):
        raise RuntimeError('Refusing broad/out-of-scope target: ' + str(path))
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise RuntimeError('Symlink ancestor: ' + str(parent))
    result, size = {}, 0
    def add(entry):
        nonlocal size
        info = entry.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise RuntimeError('Special file: ' + str(entry))
        result[str(entry.relative_to(path))] = (info.st_mode, info.st_size,
                                               info.st_mtime_ns, info.st_ino)
        size += info.st_blocks * 512
    add(path)
    if path.is_dir() and not path.is_symlink():
        def onerror(error):
            raise error
        for parent, dirs, files in os.walk(path, followlinks=False, onerror=onerror):
            for name in dirs + files:
                add(Path(parent) / name)
    return result, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--min-age', type=int, default=300)
    args = parser.parse_args()
    if ROOT.is_symlink() or ROOT.resolve() != ROOT or not ROOT.is_dir():
        raise RuntimeError('Expected exact non-symlink runtime root: ' + str(ROOT))
    for base in ('wandb', 'results', 'ozstar_logs'):
        if (ROOT / base).is_symlink():
            raise RuntimeError('Refusing symlink root: ' + base)
    # Same lock as the uploader: no simultaneous sync/removal.
    with (ROOT / 'wandb' / '.gomarl-sync-once.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, active_names = queue_state()  # Scheduler errors abort, never assume empty.
        cache = {}
        def terminal(job):
            if job not in cache:
                cache[job] = job_finished(job, REPO)
            return cache[job]
        targets, retained = inventory(ROOT, active, terminal, active_names)
        verified, total = {}, 0
        for path in targets:
            state, size = snapshot(path, ROOT)
            newest = max(v[2] for v in state.values()) / 1e9
            if time.time() - newest < max(300, args.min_age):
                retained.append(path)
                continue
            verified[path] = state
            total += size
            print('DELETE {:.3f} GiB {}'.format(size / 2**30, path), flush=True)
        print('Retain {} entries; eligible {} entries, {:.3f} GiB'.format(
            len(retained), len(verified), total / 2**30), flush=True)
        for path in retained:
            print('KEEP ' + str(path), flush=True)
        if not args.apply:
            print('Preview only. --apply deletes local histories/media permanently; cloud unchanged.')
            return
        deleted = 0
        cache.clear()  # Fresh terminal accounting immediately before apply.
        audit_path = ROOT / 'ozstar_logs' / 'cleanup_except_qmix_variants.jsonl'
        with audit_path.open('a') as audit:
            for path, previous in verified.items():
                # Re-index current scheduler + accounting and all ownership before EVERY deletion.
                active, active_names = queue_state()
                current, _ = inventory(ROOT, active, terminal, active_names)
                if path not in current or snapshot(path, ROOT)[0] != previous:
                    print('SKIP changed state/data: ' + str(path), flush=True)
                    continue
                record = dict(time=time.time(), target=str(path), action='delete_local',
                              keep_group=KEEP_GROUP)
                audit.write(json.dumps(record) + '\n')
                audit.flush()
                os.fsync(audit.fileno())
                if path.is_symlink() or path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                deleted += 1
                print('DELETED ' + str(path), flush=True)
        print('Deleted {} entries; audit: {}'.format(deleted, audit_path))


if __name__ == '__main__':
    main()
