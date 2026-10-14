#!/usr/bin/env python3
"""Status/backfill/live-sync for the exact calibration 18-job cohort only.

No deletion, cancellation, resubmission or historical project-wide scan.
Local ledger means successfully attempted upload, NOT cloud-history verification.
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
GROUPS = (GROUP, '5m6m_score_gradient_factorial_10m_6jobs_20261014')
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
    index = job_index(logs, wandb, repo)
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


def sync_once(repo, runtime, python, ended_only=False):
    logs, wandb = runtime / 'ozstar_logs', runtime / 'wandb'
    with (wandb / '.gomarl-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another upload/cleanup holds the shared lock; retry next round', flush=True)
            return
        jobs = manifest_jobs(logs)
        states = accounting(jobs, repo)
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
            signature = dict(fingerprint=before, job_states={j: states[j]['state'] for j in sorted(associated)})
            if ledger.get(run.name) == signature:
                print('UNCHANGED: ' + run.name, flush=True)
                continue
            print('UPLOAD: {} jobs={}'.format(run.name, ','.join(sorted(associated))), flush=True)
            try:
                completed = subprocess.run([python, '-m', 'wandb', 'sync', '--append',
                    '--include-offline', '--include-synced', '--no-mark-synced', '--skip-console', str(run)],
                    timeout=int(os.environ.get('SYNC_TIMEOUT', '180')))
            except subprocess.TimeoutExpired:
                print('TIMEOUT; retry next round: ' + run.name, flush=True)
                continue
            if completed.returncode:
                print('UPLOAD FAILED; retry next round: ' + run.name, flush=True)
                continue
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'once', 'start', 'restart', 'loop'))
    parser.add_argument('--ended-only', action='store_true')
    args = parser.parse_args()
    repo = Path(os.environ.get('REPO_DIR', '/home/kyang/code/gomarl-dual-branch')).resolve()
    runtime = Path(os.environ.get('RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')).resolve()
    python = os.environ.get('PYTHON_BIN', '/home/kyang/.conda/envs/marl_cpu/bin/python')
    script = repo / 'scripts' / Path(__file__).name
    logs = runtime / 'ozstar_logs'
    if args.action == 'status':
        jobs = manifest_jobs(logs)
        report(jobs, accounting(jobs, repo))
    elif args.action == 'once':
        sync_once(repo, runtime, python, args.ended_only)
    elif args.action == 'loop':
        interval = int(os.environ.get('INTERVAL_SECONDS', '600'))
        if interval <= 0:
            raise ValueError('Interval must be positive')
        with (runtime / 'wandb' / '.gomarl-calibration-sync-loop.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                started = time.monotonic()
                print('\nSYNC ROUND ' + time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
                try:
                    sync_once(repo, runtime, python)
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
                print('Sync window already alive: ' + SESSION)
                return
        else:
            subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, '-n', 'sync'], check=True)
        subprocess.run(['tmux', 'set-window-option', '-t', SESSION + ':sync', 'remain-on-exit', 'on'], check=True)
        log = logs / (SESSION + '.log')
        environment = ['env', 'REPO_DIR=' + str(repo), 'RUNTIME_ROOT=' + str(runtime),
                       'PYTHON_BIN=' + python, 'INTERVAL_SECONDS=' + os.environ.get('INTERVAL_SECONDS', '600'),
                       'SYNC_TIMEOUT=' + os.environ.get('SYNC_TIMEOUT', '180')]
        launched = 'exec ' + shlex.join(environment + [python, '-u', str(script), 'loop']) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
        subprocess.run(['tmux', 'respawn-pane', '-k', '-t', SESSION + ':sync', launched], check=True)
        time.sleep(1)
        if command(['tmux', 'display-message', '-p', '-t', SESSION + ':sync', '#{pane_dead}']) != '0':
            raise RuntimeError('Sync pane exited on startup; inspect ' + str(log))
        print('Started {}: backfill + live/final sync, every 600s by default; log={}'.format(SESSION, log))
        print('View log: tail -f ' + str(log))


if __name__ == '__main__':
    main()
