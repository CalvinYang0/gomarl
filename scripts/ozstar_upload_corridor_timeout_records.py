#!/usr/bin/env python3
"""Recover the current corridor TIMEOUT seeds, without a 10M requirement.

Prefer exact original W&B files. Otherwise publish an explicitly labelled
Sacred scalar snapshot as a separate recovered/partial run, never as training
completion. No resubmission, deletion, videos or fabricated mean3.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

import ozstar_upload_projected_fastrefresh_results as suite
from ozstar_wandb_sync_verified import read_snapshot, verify_cloud

GROUP = 'corridor_projected_fastrefresh_timeout_recovery_20261105'
RUN_PATH = re.compile(r'(?:/[^\s\"\x27<>]+/)?wandb/(?:offline-)?run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9]+')


def job_text(runtime, row):
    job, = row['job_ids']
    texts = []
    for suffix in ('out', 'err'):
        path = runtime/'ozstar_logs'/('{}_{}.{}'.format(row['name'], job, suffix))
        if path.is_file() and not path.is_symlink():
            texts.append(path.read_text(errors='replace'))
    return re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(texts))


def sources(repo, runtime, row):
    """Exact selected-job log pointers only; never choose a same-name old run."""
    text = job_text(runtime, row)
    native, ids = set(), set()
    allowed = {(runtime/'wandb').resolve(), (repo/'wandb').resolve()}
    for match in RUN_PATH.finditer(text):
        path = Path(match.group(0))
        path = (path if path.is_absolute() else repo/path).resolve()
        if path.parent not in allowed:
            continue
        ids.add(path.name.rsplit('-', 1)[-1])
        if (path/('run-'+path.name.rsplit('-', 1)[-1]+'.wandb')).is_file():
            native.add(path)
    roots = set(re.findall(r'Saving to FileStorageObserver in (.+?)\.\s*$', text, re.M))
    numbers = set(re.findall(r'Started run with ID [\"\x27]?(\d+)', text))
    sacred = None
    if len(roots) == len(numbers) == 1:
        candidate = (Path(next(iter(roots)))/next(iter(numbers))).resolve()
        if (runtime/'results/sacred').resolve() in candidate.parents and candidate.is_dir():
            sacred = candidate
    if len(native) > 1:
        raise RuntimeError('Multiple original W&B identities in this job; refusing to merge')
    return dict(native=list(native), ids=sorted(ids), sacred=sacred)


def scalar(value):
    while isinstance(value, dict) and 'value' in value:
        value = value['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Not a finite numeric scalar')
    return float(value)


def sacred_rows(path, plan, row):
    config = json.loads((path/'config.json').read_text())
    if (config.get('wandb_run_name') != plan['job_name'] or config.get('seed') != plan['seed'] or
            config.get('env_args', {}).get('map_name') != plan['map_name']):
        raise ValueError('Sacred name/seed/map does not match the selected job')
    # Numeric Sacred paths can be reused after cleanup; without immutable run
    # IDs or a time-zone-proven retry mapping, do not recover a same-name retry.
    if len(row['all_job_ids']) != 1:
        raise ValueError('Sacred-only recovery of repeated same-name attempts needs extra ownership proof')
    rows, omitted = {}, {}

    def add(key, steps, values):
        if not isinstance(steps, list) or not isinstance(values, list) or len(steps) != len(values):
            raise ValueError('Mismatched Sacred steps/values: '+key)
        pairs = {}
        for step, value in zip(steps, values):
            try:
                step, value = scalar(step), scalar(value)
                if step < 0 or step != int(step):
                    raise ValueError('Invalid environment step')
                if key == 'test_battle_won_mean' and not 0 <= value <= 1:
                    raise ValueError('Expected test-win fraction')
            except ValueError:
                omitted[key] = omitted.get(key, 0)+1
                continue
            pairs[int(step)] = value  # Last recorded duplicate within a source.
        for step, value in pairs.items():
            existing = rows.setdefault(step, {})
            if key in existing and not math.isclose(existing[key], value, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Conflicting info.json / metrics.json scalar: '+key)
            existing[key] = value

    info = path/'info.json'
    if info.is_file():
        data = json.loads(info.read_text())
        for key, values in data.items():
            if not key.endswith('_T') and key+'_T' in data:
                add(key, data[key+'_T'], values)
    metrics = path/'metrics.json'
    if metrics.is_file():
        for key, value in json.loads(metrics.read_text()).items():
            if isinstance(value, dict) and 'steps' in value and 'values' in value:
                add(key, value['steps'], value['values'])
    if not rows:
        raise ValueError('No saved finite scalar records for this timeout job')
    return [dict(env_steps=step, **value) for step, value in sorted(rows.items())], omitted


def recovery_identity(plan, row, path, rows, omitted, project):
    identity = dict(version=1, project=project, scene=plan['scene'], model=plan['label'], seed=plan['seed'],
        source_job=row['job_ids'][0], source_run_name=plan['job_name'], source_sacred_path=str(path),
        scheduler_state=row['state'], rows=rows, omitted_nonfinite_or_invalid=omitted)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return 'recovered'+digest[:20], digest, identity


def recovery_current(api, project, rid, digest, rows):
    run = api.run(project+'/'+rid)
    if (run.config.get('source_digest') != digest or not run.config.get('incomplete_training') or
            run.summary.get('recovery/source_end_step') != rows[-1]['env_steps'] or
            run.summary.get('recovery/source_row_count') != len(rows)):
        return False
    expected = [r for r in rows if 'test_battle_won_mean' in r]
    if expected:
        actual = list(run.scan_history(keys=['env_steps', 'test_battle_won_mean'], page_size=1000))
        if [(r['env_steps'], r['test_battle_won_mean']) for r in actual] != [
                (r['env_steps'], r['test_battle_won_mean']) for r in expected]:
            return False
    return True


def recover_sacred(plan, row, path, project, output):
    import wandb
    rows, omitted = sacred_rows(path, plan, row)
    rid, digest, identity = recovery_identity(plan, row, path, rows, omitted, project)
    print('SACRED PARTIAL: seed={} job={} scalar_rows={} last_env_step={} test_end={}'.format(
        plan['seed'], row['job_ids'][0], len(rows), rows[-1]['env_steps'],
        max((r['env_steps'] for r in rows if 'test_battle_won_mean' in r), default=None)), flush=True)
    try:
        current = recovery_current(wandb.Api(timeout=30), project, rid, digest, rows)
    except wandb.errors.CommError:
        current = False
    if current:
        return dict(seed=plan['seed'], job=row['job_ids'][0], run_id=rid,
            status='recovered scalar snapshot already current', end_step=rows[-1]['env_steps'])
    output.mkdir(parents=True, exist_ok=True)
    artifact = output/(rid+'.json')
    artifact.write_text(json.dumps(identity, indent=2, allow_nan=False))
    test_end = max((r['env_steps'] for r in rows if 'test_battle_won_mean' in r), default=None)
    config = {k: v for k, v in identity.items() if k != 'rows'}
    config.update(source_digest=digest, analysis_only=True, incomplete_training=True,
        recovery_scope='saved Sacred scalars only; not original W&B media or a checkpoint',
        source_end_step=rows[-1]['env_steps'], source_test_end_step=test_end, smoothing='none',
        note='Slurm TIMEOUT; actual recorded steps only, no extrapolation, no fabricated training completion')
    entity, name = project.split('/', 1)
    with wandb.init(entity=entity, project=name, id=rid, resume='allow', mode='online',
            name='corridor | '+plan['label']+' | s{} | TIMEOUT partial | {}'.format(plan['seed'], digest[:6]),
            group=GROUP, job_type='recovered-partial-history',
            tags=['timeout', 'partial', 'recovered-scalars', 'not-training', plan['scene']],
            config=config, dir=str(output)) as run:
        run.define_metric('env_steps')
        run.define_metric('*', step_metric='env_steps')
        for record in rows:
            if record['env_steps'] >= run.step:
                run.log(record, step=record['env_steps'])
        run.summary['recovery/source_end_step'] = rows[-1]['env_steps']
        run.summary['recovery/source_row_count'] = len(rows)
        data = wandb.Artifact(rid, type='recovered-partial-scalars')
        data.add_file(str(artifact))
        run.log_artifact(data)
        url = run.url
    if not recovery_current(wandb.Api(timeout=30), project, rid, digest, rows):
        raise RuntimeError('Recovered cloud snapshot/test history not verified')
    return dict(seed=plan['seed'], job=row['job_ids'][0], run_id=rid, run_url=url,
        status='uploaded and verified Sacred scalar recovery; original media not restored',
        end_step=rows[-1]['env_steps'], test_end_step=test_end, omitted=omitted)


def native_upload(path, project):
    snapshot = read_snapshot(path)
    if snapshot.entity+'/'+snapshot.project != project:
        raise ValueError('Original W&B destination differs; refusing to redirect')
    if not snapshot.history_count:
        raise ValueError('Original W&B has no complete scalar history; try saved Sacred records')
    try:
        verify_cloud(snapshot, snapshot.entity, snapshot.project, attempts=1, delay=0)
    except Exception:
        command = [sys.executable, str(Path(__file__).with_name('ozstar_wandb_sync_verified.py')), str(path)]
        subprocess.run(command, check=True, timeout=600)
        verify_cloud(snapshot, snapshot.entity, snapshot.project, attempts=1, delay=0)
        status = 'original partial W&B records synced and scalar progress verified'
    else:
        status = 'original partial W&B scalar progress already current'
    return dict(run_id=snapshot.run_id, status=status)


def cloud_original(ids, plan, project):
    """Only proven original IDs after missing local sources, not a name search."""
    import wandb
    api = wandb.Api(timeout=30)
    for rid in ids:
        try:
            run = api.run(project+'/'+rid)
        except wandb.errors.CommError:
            continue
        if run.name != plan['job_name']:
            raise ValueError('Original cloud run identity differs from job log')
        history = list(run.scan_history(keys=['_step', 'test_battle_won_mean'], page_size=1000))
        if history:
            return dict(run_id=rid, run_url=run.url, test_points=len(history),
                test_end_step=max(r['_step'] for r in history),
                status='original partial test history already in cloud; local files unavailable')
    return None


@contextmanager
def sync_lock(runtime):
    # Preserve the shared lock inode; never unlink another uploader's lock.
    with (runtime/'wandb/.gomarl-sync-once.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path('/home/kyang/gomarl-runtime/gomarl-dual-branch'))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args(argv)
    repo, runtime = Path(__file__).resolve().parents[1], args.runtime_root.resolve()
    plans = [p for p in suite.build_plans(repo) if p['scene'] == 'smac_corridor']
    up = suite.uploader
    chosen = {}
    with patch.multiple(up, SOURCES=(suite.cohort,), GROUP=GROUP):
        with up.source_scope():
            rows = suite.integrity.audit_latest(repo, runtime, plans, up.sync.manifest_jobs, chosen)
    targets = [(p, r) for p in plans for r in rows if r['name'] == p['job_name'] and r['state'] == 'TIMEOUT']
    report = dict(jobs=rows, results=[], errors=[])
    if any(r['state'].startswith(('UNKNOWN', 'MISSING', 'AMBIGUOUS')) for r in rows):
        report['errors'].append(dict(error='Some latest corridor job identities/accounting remain unverified'))
    if args.status_only:
        print('STATUS ONLY: {} latest corridor TIMEOUT jobs; no upload/writes'.format(len(targets)), flush=True)
        return report
    output = runtime/'figures'/GROUP
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/'upload_report.json'
    with sync_lock(runtime):
        report_path.write_text(json.dumps(report, indent=2))
        for plan, row in targets:
            try:
                found = sources(repo, runtime, row)
                result = None
                native_error = None
                if found['native']:
                    native, = found['native']
                    try:
                        result = native_upload(native, args.project)
                    except Exception as exc:
                        native_error = str(exc)
                        print('Native records unavailable; trying same-job Sacred scalars: '+native_error, flush=True)
                if result is None and found['sacred'] is not None:
                    try:
                        result = recover_sacred(plan, row, found['sacred'], args.project, output)
                    except Exception as exc:
                        native_error = (native_error+'; ' if native_error else '')+'Sacred recovery: '+str(exc)
                        print('Sacred recovery unavailable: '+str(exc), flush=True)
                if result is None and found['ids']:
                    result = cloud_original(found['ids'], plan, args.project)
                if result is None:
                    raise RuntimeError('No usable original W&B file or exact-job Sacred source; source data was not invented. '
                        +(native_error or ''))
                result.update(seed=plan['seed'], job=row['job_ids'][0])
                if native_error:
                    result['source_recovery_notes'] = native_error
                report['results'].append(result)
                print('TIMEOUT RECORD:', json.dumps(result), flush=True)
            except Exception as exc:
                report['errors'].append(dict(seed=plan['seed'], job=row['job_ids'][0], error=str(exc)))
                print('RECOVERY INCOMPLETE seed={}: {}'.format(plan['seed'], exc), flush=True)
            report_path.write_text(json.dumps(report, indent=2))
    print('Report: '+str(report_path), flush=True)
    if report['errors']:
        raise SystemExit('PARTIAL: some timeout source records remain unavailable; no deletion/training performed')
    return report


if __name__ == '__main__':
    main()
