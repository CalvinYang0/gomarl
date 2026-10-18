#!/usr/bin/env python3
"""Manifest-scoped W&B sync; tmux defaults to currently RUNNING jobs only.

No deletion, cancellation, resubmission or historical project-wide scan.
Ledger v1 records uploads verified visible through their pre-upload history step,
not a full per-metric/media audit or proof of completed training.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback

import yaml
from ozstar_clean_uploaded_completed_runs import RUN_RE, command, job_index, TERMINAL

GROUP = 'qmix_vdn_calibration_gate_10m_3seeds_fresh20261014'
GROUPS = (GROUP, '5m6m_score_gradient_factorial_10m_6jobs_20261014',
          '5m6m_qchange_three_ablation_10m_9jobs_20261015',
          '8m9m_ones_gate_regularization_10m_9jobs_20261015',
          'qchange_gate_ablation_10m_18jobs_20261015',
          '5m6m_qmix_mixer_input_controls_10m_9jobs',
          '5m6m_ones_10m_3seeds_20261018')
SESSION = 'gomarl-calibration-sync'


def manifest_jobs(logs):
    jobs = {}
    paths = sorted({p for group in GROUPS for p in logs.glob(group + '_*.json')})
    for path in paths:
        record = json.loads(path.read_text())
        if record.get('group') not in GROUPS:
            continue
        wanted = {p['job_name'] for p in record.get('plans', [])}
        for field in ('retained', 'submitted'):
            for name, job in record.get(field, {}).items():
                job = str(job).split(';')[0]
                if name in wanted and re.fullmatch(r'[0-9]+', job):
                    jobs[job] = name
    if not jobs:
        raise RuntimeError('No submitted/retained job IDs in this cohort manifest; cannot guess job IDs')
    return jobs


def accounting(jobs, repo):
    rows = command(['sacct', '-j', ','.join(sorted(jobs)), '-S', '2026-10-01',
                    '-X', '-n', '-P', '--format=JobIDRaw,State,ExitCode,Elapsed,WorkDir%4096'])
    result = {}
    for row in rows.splitlines():
        fields = row.rstrip('|').split('|')
        if len(fields) != 5 or fields[0].strip() not in jobs:
            continue
        job, state, exit_code, elapsed, workdir = [f.strip() for f in fields]
        if Path(workdir) != repo:
            continue
        result[job] = dict(state=state.split()[0].rstrip('+'), exit_code=exit_code, elapsed=elapsed)
    return result


def running_jobs(jobs):
    """Intersect exact manifest IDs with this user's live RUNNING queue rows."""
    user = command(['id', '-un'])
    rows = command(['squeue', '-u', user, '-h', '-o', '%A|%T'])
    result = {}
    for row in rows.splitlines():
        fields = [field.strip() for field in row.split('|')]
        if len(fields) == 2 and fields[0] in jobs and fields[1] == 'RUNNING':
            result[fields[0]] = jobs[fields[0]]
    return result


def selected_jobs(logs, repo, running_only=False, ended_only=False):
    jobs = manifest_jobs(logs)
    if running_only:
        jobs = running_jobs(jobs)
    if not jobs:
        return {}, {}
    states = accounting(jobs, repo)
    if running_only:
        # Keep the WorkDir/known-accounting safeguard as well as the live queue.
        jobs = {job: name for job, name in jobs.items()
                if states.get(job, {}).get('state') == 'RUNNING'}
        states = {job: states[job] for job in jobs}
    elif ended_only:
        jobs = {job: name for job, name in jobs.items()
                if states.get(job, {}).get('state') in TERMINAL}
        states = {job: states[job] for job in jobs}
    return jobs, states


def fingerprint(run):
    """Detect new records/media; no content read and no directory-link traversal."""
    digest = hashlib.sha256()
    run_id = run.name.rsplit('-', 1)[-1]
    entries = [run / ('run-' + run_id + '.wandb')]
    for parent, dirs, files in os.walk(run / 'files', followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        entries.extend(Path(parent) / f for f in files)
    for path in sorted(entries):
        info = path.lstat()
        metadata = [str(path.relative_to(run)), info.st_size, info.st_mtime_ns]
        if path.is_symlink():
            metadata.append(os.readlink(path))
            try:
                target = path.stat()
                metadata.extend([target.st_size, target.st_mtime_ns])
            except FileNotFoundError:
                metadata.append('missing')
        digest.update(json.dumps(metadata).encode())
    return digest.hexdigest()


def discover(wandb, logs, repo, jobs):
    index = job_index(logs, wandb, repo, job_ids=jobs.keys())
    selected = []
    for run in sorted(wandb.glob('offline-run-*')):
        if run.is_symlink() or not run.is_dir() or not RUN_RE.fullmatch(run.name):
            continue
        associated = index.get(run.name, set()) & jobs.keys()
        if not associated:
            continue  # Must be tied to exact manifest IDs, never name substring.
        names = {jobs[j] for j in associated}
        config = run / 'files' / 'config.yaml'
        if config.is_file() and not config.is_symlink():
            value = yaml.safe_load(config.read_text()) or {}
            name = value.get('wandb_run_name', {}) if isinstance(value, dict) else {}
            name = name.get('value') if isinstance(name, dict) else name
            if name and name not in names:
                raise RuntimeError('Run identity differs from its job log: ' + str(run))
        run_id = run.name.rsplit('-', 1)[-1]
        if (run / ('run-' + run_id + '.wandb')).is_file():
            selected.append((run, associated))
    return selected


def report(jobs, states):
    for job, name in sorted(jobs.items(), key=lambda x: int(x[0])):
        row = states.get(job, {})
        print('{} {:16} exit={} elapsed={} {}'.format(job, row.get('state', 'UNKNOWN'),
              row.get('exit_code', '?'), row.get('elapsed', '?'), name), flush=True)


def check_cloud_current(run):
    """Only exact original IDs; verify scalar progress, not full media completeness."""
    from ozstar_wandb_sync_verified import read_snapshot, verify_cloud
    snapshot = read_snapshot(run)
    if not snapshot.entity or not snapshot.project:
        raise RuntimeError('Original run destination unknown')
    verify_cloud(snapshot, snapshot.entity, snapshot.project, attempts=1, delay=0)


def sync_once(repo, runtime, python, ended_only=False, running_only=False,
              verify_existing=False, wanted_names=None):
    if ended_only and running_only:
        raise ValueError('ended_only and running_only are mutually exclusive')
    logs, wandb = runtime / 'ozstar_logs', runtime / 'wandb'
    result = dict(checked=0, current=0, uploaded=0, failed=0, missing_local=[], busy=False)
    with (wandb / '.gomarl-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another upload/cleanup holds the shared lock; retry next round', flush=True)
            result['busy'] = True
            return result
        jobs, states = selected_jobs(logs, repo, running_only=running_only, ended_only=ended_only)
        if wanted_names is not None:
            jobs = {job: name for job, name in jobs.items() if name in wanted_names}
            states = {job: states[job] for job in jobs if job in states}
        if not jobs:
            scope = 'RUNNING' if running_only else 'ENDED'
            print('NO {} JOBS in the configured manifests; nothing to upload'.format(scope), flush=True)
            return result
        report(jobs, states)
        ledger_path = logs / (GROUP + '_sync_ledger.json')
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        found = discover(wandb, logs, repo, jobs)
        seen = set()
        for run, associated in found:
            seen.update(associated)
            if any(j not in states for j in associated):
                print('SKIP unknown accounting state: ' + run.name, flush=True)
                continue
            if ended_only and not all(states[j]['state'] in TERMINAL for j in associated):
                continue
            before = fingerprint(run)
            signature = dict(fingerprint=before, job_states={j: states[j]['state'] for j in sorted(associated)},
                             verification_version=1)
            if verify_existing:
                result['checked'] += 1
                print('CHECK ORIGINAL CLOUD RUN: ' + run.name, flush=True)
                try:
                    check_cloud_current(run)
                except Exception as exc:
                    print('CLOUD NOT VERIFIED CURRENT: {} ({}); append original local records'.format(
                        run.name, type(exc).__name__), flush=True)
                else:
                    result['current'] += 1
                    print('CLOUD CURRENT (scalar progress only): ' + run.name, flush=True)
                    continue
            elif ledger.get(run.name) == signature:
                print('UNCHANGED: ' + run.name, flush=True)
                continue
            if running_only and set(running_jobs({j: jobs[j] for j in associated})) != associated:
                print('SKIP no longer RUNNING: ' + run.name, flush=True)
                continue
            print('UPLOAD: {} jobs={}'.format(run.name, ','.join(sorted(associated))), flush=True)
            try:
                completed = subprocess.run([python, str(Path(__file__).with_name('ozstar_wandb_sync_verified.py')),
                    str(run)],
                    timeout=int(os.environ.get('SYNC_TIMEOUT', '180')))
            except subprocess.TimeoutExpired:
                print('TIMEOUT; retry next round: ' + run.name, flush=True)
                result['failed'] += 1
                continue
            if completed.returncode:
                print('UPLOAD FAILED; retry next round: ' + run.name, flush=True)
                result['failed'] += 1
                continue
            result['uploaded'] += 1
            # Save PRE-upload signature. Changes written during upload force
            # another round rather than wrongly declaring new data synchronized.
            ledger[run.name] = signature
            temporary = ledger_path.with_suffix('.json.tmp')
            with temporary.open('w') as handle:
                json.dump(ledger, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(ledger_path)
        for job in sorted(jobs.keys() - seen):
            print('NO LOCAL RUN: {} {}'.format(job, jobs[job]), flush=True)
            result['missing_local'].append(dict(job=job, name=jobs[job]))
        print('Ended/backfill result: ' + json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'once', 'start', 'restart', 'loop'))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--ended-only', action='store_true', help='Upload terminal jobs only')
    scope.add_argument('--running-only', action='store_true', help='Upload only currently RUNNING jobs')
    scope.add_argument('--all-states', action='store_true', help='Opt into historical/backfill and live sync')
    args = parser.parse_args()
    running_only = args.running_only or (
        args.action in ('start', 'restart', 'loop') and not (args.all_states or args.ended_only))
    mode_flag = '--running-only' if running_only else ('--ended-only' if args.ended_only else '--all-states')
    repo = Path(os.environ.get('REPO_DIR', '/home/kyang/code/gomarl-dual-branch')).resolve()
    runtime = Path(os.environ.get('RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')).resolve()
    python = os.environ.get('PYTHON_BIN', '/home/kyang/.conda/envs/marl_cpu/bin/python')
    script = repo / 'scripts' / Path(__file__).name
    logs = runtime / 'ozstar_logs'
    if args.action == 'status':
        jobs, states = selected_jobs(logs, repo, running_only=running_only, ended_only=args.ended_only)
        report(jobs, states)
        if running_only and not jobs:
            print('NO RUNNING JOBS in the configured manifests', flush=True)
    elif args.action == 'once':
        sync_once(repo, runtime, python, ended_only=args.ended_only, running_only=running_only)
    elif args.action == 'loop':
        interval = int(os.environ.get('INTERVAL_SECONDS', '600'))
        if interval <= 0:
            raise ValueError('Interval must be positive')
        with (runtime / 'wandb' / '.gomarl-calibration-sync-loop.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                started = time.monotonic()
                print('\nSYNC ROUND {} mode={}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), mode_flag), flush=True)
                try:
                    sync_once(repo, runtime, python, ended_only=args.ended_only, running_only=running_only)
                except Exception:
                    traceback.print_exc()  # Keep loop alive after transient failures.
                time.sleep(max(1, interval - (time.monotonic() - started)))
    else:
        manifest_jobs(logs)  # Fail before creating a window with wrong scope.
        exists = subprocess.run(['tmux', 'has-session', '-t', '=' + SESSION], capture_output=True).returncode == 0
        if args.action == 'restart' and exists:
            subprocess.run(['tmux', 'kill-session', '-t', '=' + SESSION], check=True)
            exists = False  # Restart only our uploader, never any training job.
        if exists:
            dead = command(['tmux', 'display-message', '-p', '-t', SESSION + ':sync', '#{pane_dead}'])
            if dead == '0':
                print('Sync window already alive: {} (existing scope unchanged; use restart to change scope)'.format(SESSION))
                return
        else:
            subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, '-n', 'sync'], check=True)
        subprocess.run(['tmux', 'set-window-option', '-t', SESSION + ':sync', 'remain-on-exit', 'on'], check=True)
        log = logs / (SESSION + '.log')
        environment = ['env', 'REPO_DIR=' + str(repo), 'RUNTIME_ROOT=' + str(runtime),
                       'PYTHON_BIN=' + python, 'INTERVAL_SECONDS=' + os.environ.get('INTERVAL_SECONDS', '600'),
                       'SYNC_TIMEOUT=' + os.environ.get('SYNC_TIMEOUT', '180')]
        launched = 'exec ' + shlex.join(environment + [python, '-u', str(script), 'loop', mode_flag]) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
        subprocess.run(['tmux', 'respawn-pane', '-k', '-t', SESSION + ':sync', launched], check=True)
        time.sleep(1)
        if command(['tmux', 'display-message', '-p', '-t', SESSION + ':sync', '#{pane_dead}']) != '0':
            raise RuntimeError('Sync pane exited on startup; inspect ' + str(log))
        print('Started {}: mode={}, every {}s; log={}'.format(
            SESSION, mode_flag, os.environ.get('INTERVAL_SECONDS', '600'), log))
        print('View log: tail -f ' + str(log))


if __name__ == '__main__':
    main()
