#!/usr/bin/env python3
"""Nov13 SMAC transfer only: full mean3, then verify originals and remove local copies."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import ozstar_clean_uploaded_completed_runs as cleanup
import ozstar_submit_projected_faststages_six_maps_54jobs as cohort
import ozstar_upload_3s5z_refresh_warmup_results as integrity
import ozstar_wandb_sync_verified as verified_sync

up = integrity.uploader
PROJECT = 'hjh331-sjtu/gomarl'
GROUP = 'projected_faststages_smac_transfer_backfill_20261113'
SCENES = {key: (name, 'test_battle_won_mean') for key, name, env, *_ in cohort.SCENES if env == 'smac'}
LABELS = dict(zip(cohort.LABELS, [('QMIX live state', '#ff7f0e'), ('QMIX all ones', '#1f77b4'),
                                ('QMIX projected init1 / faststages8M', '#9467bd')]))


def snapshot_for(runtime, name):
    """Accept only verified SDK links to this exact run's gate JSONs."""
    original_snapshot = cleanup.snapshot

    def snapshot(directory):
        entries = original_snapshot(directory, allow_links=True)
        source = runtime/'results/mixer_soft_onefill_gate'
        for key, entry in list(entries.items()):
            if len(entry) != 4 or key == 'logs/debug-core.log':
                continue
            if not re.fullmatch(r'files/step_\d+_update_\d+\.json', key):
                raise RuntimeError('Unrecognized W&B link; retain: '+key)
            target = (directory/key).resolve(strict=True)
            if (target.parent.parent != source or not target.is_file() or
                    target.name != Path(key).name or json.loads(target.read_text()).get('run_name') != name):
                raise RuntimeError('Cannot prove saved-file link belongs to this run; retain: '+key)
            st = target.stat()
            entries[key] = (*entry, st.st_size, st.st_mtime_ns, st.st_ino)
        return entries
    return snapshot


def owned_directories(repo, runtime, jobs):
    result = {j: [] for j in jobs}
    root = runtime/'wandb'
    for name, owners in cleanup.job_index(runtime/'ozstar_logs', root, repo, job_ids=jobs).items():
        if len(owners) != 1:
            raise RuntimeError('Ambiguous job ownership: '+name)
        result[next(iter(owners))].append(root/name)
    return result


def exact_local_index(directories, jobs, wanted):
    result = {}
    for job, name in jobs.items():
        if name not in wanted:
            continue
        for directory in directories[job]:
            if directory.is_symlink():
                raise RuntimeError('Unexpected linked run directory: '+str(directory))
            rid = cleanup.RUN_RE.fullmatch(directory.name).group(2)
            path = directory/('run-'+rid+'.wandb')
            if path.is_symlink():
                raise RuntimeError('Unexpected linked W&B record: '+str(path))
            if path.is_file():
                result.setdefault(name, []).append(path)
    return result


def publish(record, output, api, wandb):
    rid, digest = up.scalar_identity(record, PROJECT)
    try:
        current = up.scalar_current(api, PROJECT, rid, digest)
    except wandb.errors.CommError:
        current = False
    if not current:
        up.upload_model(record, PROJECT, output/'mean3', wandb)
        if not up.scalar_current(wandb.Api(timeout=30), PROJECT, rid, digest):
            raise RuntimeError('Mean3 not verified on W&B: '+rid)
    return {'run_id': rid, 'url': 'https://wandb.ai/'+PROJECT+'/runs/'+rid}


def clean_original(directory, job, name, runtime, repo, api_factory):
    root = runtime/'wandb'
    if directory.is_symlink() or directory.resolve().parent != root or not directory.is_dir():
        raise RuntimeError('Unsafe local run path: '+str(directory))
    match = cleanup.RUN_RE.fullmatch(directory.name)
    if not match:
        raise RuntimeError('Not an offline W&B run directory: '+str(directory))
    data = directory/('run-'+match.group(2)+'.wandb')
    if not data.is_file() or data.is_symlink():
        raise RuntimeError('Missing original W&B record or unexpected symlink: '+str(data))
    dest, actual_name = cleanup.identity(data, match.group(2), *PROJECT.split('/'))
    if dest != PROJECT+'/'+match.group(2) or actual_name != name:
        raise RuntimeError('Original name/project mismatch: '+str(directory))
    scoped_snapshot = snapshot_for(runtime, name)
    before = scoped_snapshot(directory)
    if (job in cleanup.live_jobs() or not cleanup.job_finished(job, repo) or
            (before and time.time() - max(entry[1] for entry in before.values()) / 1e9 < 300)):
        print('RETAIN live/unknown or recently modified original: '+directory.name, flush=True)
        return 0
    # Avoid append when full history and files are already in the cloud.
    synced = False
    try:
        cleanup.verify_cloud(api_factory().run(dest), data, directory, name)
    except Exception as exc:
        print('Original needs final sync/verification: {} ({})'.format(dest, exc), flush=True)
        # The old raw CLI can report "done" while hiding a sender/worker
        # error. Reuse the compatibility uploader, with visibility retries,
        # before invoking the stricter deletion verification below.
        if job in cleanup.live_jobs() or not cleanup.job_finished(job, repo):
            raise RuntimeError('Job status changed before final sync; retain')
        subprocess.run([sys.executable, str(Path(verified_sync.__file__).resolve()), str(directory),
                        '-e', PROJECT.split('/')[0], '-p', PROJECT.split('/')[1]],
                       check=True, timeout=900)
        after = scoped_snapshot(directory)
        if not cleanup.unchanged(before, after, allow_sync_metadata=True):
            raise RuntimeError('Run payload changed during final sync; retain')
        synced = True
    args = SimpleNamespace(repo=repo, wandb_root=root, entity=PROJECT.split('/')[0],
        project=PROJECT.split('/')[1], apply=True, only_already_uploaded=True,
        # SDK metadata may have just been rewritten by our own sync. The age
        # check above preceded the sync; process still rechecks file stability.
        min_age=0 if synced else 300, sync_timeout=900)
    write_audit = cleanup.write_audit
    with patch.object(cleanup, 'snapshot', side_effect=scoped_snapshot), \
            patch.object(cleanup, 'write_audit', side_effect=lambda args, audit:
                         write_audit(args, dict(audit, final_sync_performed=synced))):
        return cleanup.process(directory, {job}, args, api_factory)


def save_report(path, report):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(report, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def upload_original(directory, job, name, runtime, repo, api_factory):
    root = runtime/'wandb'
    if directory.is_symlink() or directory.resolve().parent != root or not directory.is_dir():
        raise RuntimeError('Unsafe local run path: '+str(directory))
    match = cleanup.RUN_RE.fullmatch(directory.name)
    if not match:
        raise RuntimeError('Not an offline W&B run directory: '+str(directory))
    rid = match.group(2)
    data = directory/('run-'+rid+'.wandb')
    if data.is_symlink() or not data.is_file():
        raise RuntimeError('Missing or linked original W&B record: '+str(data))
    snapshot = verified_sync.read_snapshot(directory)
    if (snapshot.run_id != rid or snapshot.project != PROJECT.split('/')[1] or
            snapshot.entity not in ('', PROJECT.split('/')[0]) or not snapshot.history_count):
        raise RuntimeError('Unexpected original run destination or no readable history: '+str(directory))
    destination = PROJECT+'/'+rid

    def current():
        remote = api_factory().run(destination)
        if remote.id != rid or remote.name != name:
            raise ValueError('Cloud run ID/name differs from job ownership: '+destination)
        if remote.lastHistoryStep is None or remote.lastHistoryStep < snapshot.last_step:
            raise RuntimeError('Cloud history behind local snapshot: '+destination)

    try:
        current()
    except ValueError:
        raise
    except Exception:
        if job in cleanup.live_jobs() or not cleanup.job_finished(job, repo):
            raise RuntimeError('Job no longer confirmed ended: '+job)
        subprocess.run([sys.executable, str(Path(verified_sync.__file__).resolve()), str(directory),
                        '-e', PROJECT.split('/')[0], '-p', PROJECT.split('/')[1]],
                       check=True, timeout=900)
        current()
        status = 'synced; cloud history endpoint verified'
    else:
        status = 'already present; cloud history endpoint verified'
    print('ORIGINAL {}: {} ({}, step={})'.format(job, status, destination, snapshot.last_step), flush=True)
    return dict(run_id=rid, url='https://wandb.ai/'+PROJECT+'/runs/'+rid,
                last_step=snapshot.last_step, status=status)


def upload_ended(repo, runtime, rows, output, report, path, api_factory):
    live = cleanup.live_jobs()
    selected = {r['job_ids'][0]: r for r in rows if len(r['job_ids']) == 1 and
                r['state'] in cleanup.TERMINAL and r['job_ids'][0] not in live}
    jobs = {job: row['name'] for job, row in selected.items()}
    directories = owned_directories(repo, runtime, jobs)
    for row in rows:
        item = dict(name=row['name'], slurm_state=row['state'], jobs=row['job_ids'])
        report['originals'].append(item)
        try:
            ids = row['job_ids']
            if len(ids) != 1 or ids[0] not in selected:
                raise RuntimeError('Job not confirmed terminal')
            job = ids[0]
            targets = [d for d in directories[job] if d.exists() or d.is_symlink()]
            if len(targets) != 1:
                if targets:
                    raise RuntimeError('Multiple local originals; cannot select unambiguously')
                # A prior verified cleanup may have removed the local copy. Check
                # only the original ID established by this exact job's log.
                known = directories[job]
                if len(known) != 1:
                    raise RuntimeError('No local original or unique logged W&B ID')
                rid = cleanup.RUN_RE.fullmatch(known[0].name).group(2)
                remote = api_factory().run(PROJECT+'/'+rid)
                if remote.id != rid or remote.name != row['name'] or remote.lastHistoryStep is None:
                    raise RuntimeError('Original without local copy not confirmed on W&B')
                item.update(run_id=rid, last_step=remote.lastHistoryStep, status='cloud present; no local copy')
            else:
                item.update(upload_original(targets[0], job, row['name'], runtime, repo, api_factory))
        except Exception as exc:
            item.update(status='unverified', error=str(exc))
            report['errors'].append('{}: {}'.format(row['name'], exc))
            print('ORIGINAL UNVERIFIED {}: {}'.format(row['name'], exc), flush=True)
        save_report(path, report)


def run(repo, runtime, apply, upload_only=False):
    plans = cohort.build_plans(repo, 'smac')
    if len(plans) != 27 or {p['scene'] for p in plans} != set(SCENES):
        raise RuntimeError('Expected exactly the Nov13 SMAC transfer cohort')
    chosen = {}
    with patch.object(up.sync, 'GROUPS', (cohort.GROUP,)):
        rows = integrity.audit_latest(repo, runtime, plans, up.sync.manifest_jobs, chosen)
    by_name = {r['name']: r for r in rows}
    live = cleanup.live_jobs()
    ready, skipped = up.full_models(plans, [dict(r, state='LIVE_QUEUE') if live.intersection(r['job_ids'])
                                           else r for r in rows])
    print('Three COMPLETED seeds: {}/9 groups; test history must still reach 10M'.format(len(ready)), flush=True)
    if not (apply or upload_only) or (not ready and not upload_only):
        return
    output = runtime/'figures'/GROUP
    output.mkdir(parents=True, exist_ok=True)
    path = output/('upload_only_report.json' if upload_only else 'upload_report.json')
    report = dict(jobs=rows, skipped=skipped, groups=[], originals=[], errors=[])
    import wandb
    api = wandb.Api(timeout=30)
    api_factory = lambda: wandb.Api(timeout=90)
    with (runtime/'wandb'/'.gomarl-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('W&B shared sync lock busy; nothing uploaded or removed by this command')
        save_report(path, report)
        if upload_only:
            upload_ended(repo, runtime, rows, output, report, path, api_factory)
        for selected in ready:
            scene, label = selected[0]['scene'], selected[0]['label']
            group = dict(scene=scene, model=label, originals=[])
            report['groups'].append(group)
            jobs = {by_name[p['job_name']]['job_ids'][0]: p['job_name'] for p in selected}
            try:
                if cleanup.live_jobs().intersection(jobs) or not all(cleanup.job_finished(j, repo) for j in jobs):
                    raise RuntimeError('A selected job is no longer confirmed terminal')
                directories = owned_directories(repo, runtime, jobs)
                with patch.multiple(up, GROUP=GROUP, SCENES=SCENES, LABELS=LABELS), \
                        patch.multiple(up.runner.study.charts, ROOT=repo, local_run_index=lambda _, wanted:
                                       exact_local_index(directories, jobs, wanted)):
                    record, inventory = integrity.strict_render(runtime, PROJECT, output/(scene+'__'+label),
                                                               selected, jobs, up.render_model)
                group.update(mean3=publish(record, output, api, wandb), inventory=inventory)
                print('MEAN3 VERIFIED: '+group['mean3']['url'], flush=True)
                save_report(path, report)
            except (Exception, SystemExit) as exc:
                report['errors'].append('{} {}: {}'.format(scene, label, exc))
                print('RETAIN GROUP: {} {}: {}'.format(scene, label, exc), flush=True)
                save_report(path, report)
                continue
            if upload_only:
                continue
            for job, name in jobs.items():
                targets = [d for d in directories[job] if d.exists() or d.is_symlink()]
                item = dict(job=job, name=name, paths=list(map(str, targets)))
                group['originals'].append(item)
                try:
                    if not targets:
                        item['status'] = 'no local copy to remove; original full backup not reverified'
                    elif len(targets) != 1:
                        raise RuntimeError('Multiple local originals; retain all')
                    else:
                        freed = clean_original(targets[0], job, name, runtime, repo, api_factory)
                        item.update(status='verified then removed' if not targets[0].exists() else 'retained',
                                    estimated_bytes=freed)
                except Exception as exc:
                    item.update(status='retained', error=str(exc))
                    report['errors'].append('{}: {}'.format(job, exc))
                    print('RETAIN ORIGINAL {}: {}'.format(job, exc), flush=True)
                save_report(path, report)
    removed = [item for group in report['groups'] for item in group['originals']
               if item['status'] == 'verified then removed']
    print('Mean3 verified: {}; local copies removed: {}; report: {}'.format(
        sum('mean3' in group for group in report['groups']), len(removed), path), flush=True)
    if upload_only:
        print('Original run visibility/history endpoint checks: {}/{}; no cleanup or submission'.format(
            sum(item['status'] != 'unverified' for item in report['originals']), len(rows)), flush=True)
    if report['errors']:
        raise SystemExit('PARTIAL: see report; unverified local copies retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Upload and remove only fully verified local W&B copies')
    mode.add_argument('--upload-only', action='store_true', help='All ended originals and complete mean3; never delete')
    parser.add_argument('--job-repo', type=Path, default=Path('/home/kyang/code/gomarl-sixmap-9b8d3a2'))
    parser.add_argument('--runtime-root', type=Path, default=Path('/home/kyang/gomarl-runtime/gomarl-dual-branch'))
    args = parser.parse_args()
    runtime, repo = args.runtime_root, args.job_repo.resolve(strict=True)
    if (runtime.resolve(strict=True) != runtime or Path('/home/kyang') not in runtime.parents or
            any(not p.is_dir() or p.is_symlink() for p in (runtime/'wandb', runtime/'ozstar_logs'))):
        raise SystemExit('Unsafe runtime/W&B/log root')
    run(repo, runtime, args.apply, args.upload_only)


if __name__ == '__main__':
    main()
