#!/usr/bin/env python3
"""Exact Nov22 cohort: ended whole groups only; scalars + video byte audits.

No training, deletion, environment changes or whole-project history scans.
"""
import base64
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path('/home/kyang/gomarl-runtime/gomarl-dual-branch')
REPO = Path('/home/kyang/code/gomarl-sixmap-9b8d3a2')
GROUP = 'projected_microtrick_init03_init07_freeze5m_video_18jobs_20261122'
RECEIPT = GROUP+'_20261122_050042_264434.json'
PROJECT = 'hjh331-sjtu/gomarl'


def video_payloads(rows, directory, last_step):
    """One actual logged video per observed 1M milestone; exact file hashes."""
    clips = {}
    milestones = []
    prefix = 'test_battle_video/episode_01'
    seen = set()
    for row in rows:
        value = row.get(prefix)
        # W&B 0.18.7 can encode Video metadata as nested_key components.
        # cleanup.histories preserves them as dot-separated keys, not a dict.
        if not isinstance(value, dict):
            value = {k[len(prefix)+1:]: v for k, v in row.items()
                     if k.startswith(prefix+'.')}
        if not isinstance(value, dict) or value.get('_type') != 'video-file':
            continue
        relative = value['path']
        if not isinstance(relative, str):
            raise ValueError('Invalid video path metadata')
        path = directory/'files'/relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts or
                (directory/'files').resolve() not in path.resolve().parents):
            raise ValueError('Unexpected video path')
        # Native training logger uses the environment step as W&B's _step.
        step = row.get('_step', row.get('env_steps'))
        if isinstance(step, bool) or not isinstance(step, (int, float)) or int(step) != step:
            raise ValueError('Video has no valid environment step')
        identity = (int(step), relative)
        if identity in seen:
            continue
        seen.add(identity)
        before = path.stat()
        digest = hashlib.md5()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1048576), b''):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Video changed during audit')
        if before.st_size <= 0:
            raise RuntimeError('Empty local video')
        if relative in clips:
            raise RuntimeError('One video file reused at different training steps')
        clips[relative] = (before.st_size, {digest.hexdigest(), base64.b64encode(digest.digest()).decode()})
        milestones.append(int(step)//1000000)
    expected = set(range(1, min(10, int(last_step)//1000000)+1))
    if not expected or set(milestones) != expected or len(milestones) != len(expected):
        raise RuntimeError('Expected one video at each milestone {}; found {}'.format(sorted(expected), milestones))
    return clips


def main():
    import wandb
    import ozstar_submit_projected_microtrick_video_18jobs as suite
    manifest = json.loads((ROOT/'ozstar_logs'/RECEIPT).read_text())
    plans = manifest['plans']
    names = dict(manifest['retained'], **manifest['submitted'])
    jobs = {str(j): n for n, j in names.items()}
    expected = suite.build_plans(REPO)
    signature = lambda rows: {(p['job_name'], p['scene'], p['label'], p['seed']) for p in rows}
    if not (manifest['group'] == GROUP and len(plans) == len(jobs) == 18 and
            set(jobs) == {str(j) for j in range(21354580, 21354598)} and
            signature(plans) == signature(expected) and set(names) == {p['job_name'] for p in plans}):
        raise RuntimeError('Nov22 manifest/job/model identity mismatch; nothing uploaded')
    if any(p.is_symlink() or not p.is_dir() for p in (ROOT, ROOT/'wandb', ROOT/'ozstar_logs')):
        raise RuntimeError('Unsafe runtime path')
    h = ModuleType('microtrick_exact_upload_helper')
    source = subprocess.check_output(['git', 'show',
        '7f91834:scripts/ozstar_upload_clean_sixmap_smac.py'], cwd=REPO, timeout=20)
    exec(compile(source, 'verified_upload_helper', 'exec'), h.__dict__)
    raw = subprocess.check_output(['sacct', '-S', '2026-11-22', '-X', '-n', '-P',
        '-j', ','.join(jobs), '--format=JobIDRaw,JobName%256,State,ExitCode,WorkDir%4096'],
        text=True, timeout=40)
    states = {}
    for line in raw.splitlines():
        job, name, state, code, work = line.strip().split('|')
        if job in jobs and name == jobs[job] and Path(work).resolve() == REPO:
            value = (state.split()[0].rstrip('+'), code)
            assert job not in states or states[job] == value
            states[job] = value
    active = h.cleanup.live_jobs()
    groups = defaultdict(list)
    for p in plans:
        groups[p['scene'], p['label']].append(p)
    ready = []
    for key, selected in groups.items():
        ids = {str(names[p['job_name']]) for p in selected}
        if not ids.intersection(active) and all(states.get(j) in {
                ('COMPLETED', '0:0'), ('TIMEOUT', '0:0')} for j in ids):
            ready.append(selected)
        else:
            print('DEFER WHOLE GROUP:', key, {j: states.get(j) for j in ids}, flush=True)
    print('READY: {} groups / {} single seeds'.format(len(ready), 3*len(ready)), flush=True)
    output = ROOT/'figures'/GROUP/'upload'
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/'upload_report.json'
    report = dict(states=states, originals=[], groups=[], errors=[])
    api = lambda: wandb.Api(timeout=20)
    checks = []

    def cloud_original(item):
        remote = api().run(PROJECT+'/'+item['run_id'])
        if (remote.name != item['name'] or remote.lastHistoryStep is None or
                remote.lastHistoryStep < item['last_step'] or
                remote.id != item['run_id'] or remote.summary.get('test_battle_won_mean') is None):
            raise RuntimeError('Original identity/scalar endpoint pending')
        item['scalar_status'] = 'verified'
        if item.get('video_error'):
            raise RuntimeError(item['video_error'])
        files = {f.name: f for f in remote.files()}
        for name, (size, hashes) in item['_clips'].items():
            f = files.get(name)
            if f is None or f.size != size or f.md5 not in hashes:
                raise RuntimeError('Video missing or different: '+name)
        item['video_status'] = 'verified'
        return True

    def save():
        public = dict(report, originals=[{k: v for k, v in row.items() if k != '_clips'}
                                        for row in report['originals']])
        h.save_report(report_path, public)

    with (ROOT/'wandb/.gomarl-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Upload lock busy; no upload/deletion/submission by this command')
        selected_jobs = {str(names[p['job_name']]): p['job_name'] for g in ready for p in g}
        directories = h.owned_directories(REPO, ROOT, selected_jobs)
        for job, name in selected_jobs.items():
            item = dict(job=job, name=name, status='pending', video_count=0)
            report['originals'].append(item)
            try:
                targets = [p for p in directories[job] if p.is_dir()]
                if len(targets) != 1:
                    raise RuntimeError('Requires one uniquely owned local original')
                directory = targets[0]
                if directory.is_symlink() or directory.resolve().parent != ROOT/'wandb':
                    raise RuntimeError('Original is linked or outside exact W&B root')
                snap = h.verified_sync.read_snapshot(directory)
                data = directory/('run-'+snap.run_id+'.wandb')
                destination, actual = h.cleanup.identity(data, snap.run_id, *PROJECT.split('/'))
                if actual != name or destination != PROJECT+'/'+snap.run_id:
                    raise RuntimeError('Original name/project differs from the manifest')
                item.update(run_id=snap.run_id, last_step=snap.last_step)
                try:
                    item['_clips'] = video_payloads(h.cleanup.histories(data), directory, snap.last_step)
                    item['video_count'] = len(item['_clips'])
                except Exception as exc:
                    item['_clips'], item['video_error'] = {}, str(exc)
                try:
                    cloud_original(item)
                except Exception:
                    if job in h.cleanup.live_jobs() or not h.cleanup.job_finished(job, REPO):
                        raise RuntimeError('Job no longer confirmed ended; not syncing')
                    process = subprocess.run([sys.executable, str(Path(h.verified_sync.__file__).resolve()),
                        str(directory), '-e', PROJECT.split('/')[0], '-p', PROJECT.split('/')[1]], timeout=900)
                    item['sync_exit_code'] = process.returncode
                checks.append(item)
                print('SINGLE UPLOAD/CHECK:', job, snap.run_id, 'local videos=', item['video_count'], flush=True)
            except Exception as exc:
                item['error'] = str(exc)
                print('SINGLE ERROR:', job, str(exc), flush=True)
            save()

        up, charts = h.up, h.up.runner.study.charts
        for selected in ready:
            scene, label = selected[0]['scene'], selected[0]['label']
            try:
                chosen = {str(names[p['job_name']]): p['job_name'] for p in selected}
                assert not set(chosen).intersection(h.cleanup.live_jobs())
                local = h.exact_local_index(directories, chosen, set(chosen.values()))
                dest = output/(scene+'__'+label)
                with patch.multiple(up, GROUP=GROUP,
                        SCENES={scene: (selected[0]['map_name'], 'test_battle_won_mean')},
                        LABELS={label: (label, '#1f77b4')}), \
                     patch.multiple(charts, ROOT=REPO, local_run_index=lambda *_: local):
                    record, inventory = h.integrity.strict_render(ROOT, PROJECT, dest, selected,
                                                                 chosen, up.render_model)
                rid, digest = up.scalar_identity(record, PROJECT)
                item = dict(scene=scene, model=label, run_id=rid, digest=digest,
                    end_step=record['rows'][-1]['env_steps'], inventory=inventory, status='pending')
                report['groups'].append(item)
                try:
                    current = up.scalar_current(api(), PROJECT, rid, digest)
                except Exception:
                    current = False
                if not current:
                    up.upload_model(record, PROJECT, output/'mean3', wandb)
                print('MEAN3 UPLOAD/CHECK:', scene, label, rid, flush=True)
            except Exception as exc:
                report['errors'].append('{} {}: {}'.format(scene, label, exc))
                print('GROUP ERROR:', scene, label, str(exc), flush=True)
            save()

        for item in checks:
            try:
                cloud_original(item)
                item['status'] = 'verified'
                print('SINGLE+VIDEO VERIFIED:', item['job'], item['run_id'], item['video_count'], flush=True)
            except Exception as exc:
                item['video_status'] = 'pending'
                item['verification_error'] = str(exc)
                print('SINGLE+VIDEO PENDING:', item['job'], item['run_id'], str(exc), flush=True)
        for item in report['groups']:
            try:
                if up.scalar_current(api(), PROJECT, item['run_id'], item['digest']):
                    item['status'] = 'verified'
                print('MEAN3', item['status'].upper(), item['scene'], item['model'],
                      'https://wandb.ai/'+PROJECT+'/runs/'+item['run_id'], flush=True)
            except Exception as exc:
                print('MEAN3 CHECK PENDING:', item['run_id'], str(exc), flush=True)
        save()
    print('REPORT:', report_path, flush=True)
    print('Verified single+video: {}/{}; mean3: {}/{}'.format(
        sum(i['status'] == 'verified' for i in report['originals']), len(report['originals']),
        sum(i['status'] == 'verified' for i in report['groups']), len(report['groups'])), flush=True)
    print('No training submitted; no data deleted.', flush=True)


if __name__ == '__main__':
    main()
