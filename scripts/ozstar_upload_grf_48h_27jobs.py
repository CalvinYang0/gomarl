#!/usr/bin/env python3
"""Exact GRF Nov23/24 48h cohort: status or upload only, never train/delete."""
import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

GROUP = 'grf_projected_init1_fast8m_vs_ones_qmix_10m_48h_27jobs_20261123'
METRIC = 'test_game_win_mean'
TARGET = 10000000
JOB_REPO = Path('/home/kyang/code/gomarl-sixmap-9b8d3a2')
RUNTIME = Path('/home/kyang/gomarl-runtime/gomarl-dual-branch')
MODELS = ('qmix', 'qmix_state_ones', 'qmix_projected_init1_faststages8m')
MAPS = (('grf_counter', 'academy_counterattack_easy'),
        ('grf_pass', 'academy_pass_and_shoot_with_keeper'),
        ('grf_3v1', 'academy_3_vs_1_with_keeper'))


def expected_plans():
    return [dict(job_name='{}_{}_10m_s{}_grf10m48h20261123'.format(scene, model, seed),
                 scene=scene, map_name=name, label=model, seed=seed, win_metric=METRIC)
            for scene, name in MAPS for model in MODELS for seed in (1, 2, 3)]


def expected_jobs():
    return {str(21445821+i): p['job_name'] for i, p in enumerate(expected_plans())}


def load_helpers():
    # The legacy helper's unused Nov13 cohort import is not shipped in the
    # pinned worktree. Supply metadata only, without importing any submitter.
    stub = ModuleType('ozstar_submit_projected_faststages_six_maps_54jobs')
    stub.SCENES, stub.LABELS = (), MODELS
    with patch.dict(sys.modules, {stub.__name__: stub}):
        import ozstar_upload_clean_sixmap_smac as h
    return h


def accounting(repo):
    jobs = expected_jobs()
    text = subprocess.check_output(['sacct', '-u', 'kyang', '-S', '2026-11-24',
        '-X', '-n', '-P', '-j', ','.join(jobs),
        '--format=JobIDRaw,JobName%256,State,ExitCode,WorkDir%4096'], text=True, timeout=45)
    rows = {}
    for line in text.splitlines():
        parts = line.strip().rstrip('|').split('|')
        if len(parts) != 5 or parts[0] not in jobs:
            continue
        job, name, state, code, work = parts
        if name != jobs[job] or Path(work).resolve() != repo.resolve():
            raise RuntimeError('Exact job name/workdir differs: '+job)
        row = dict(name=name, state=state.split()[0].rstrip('+'), exit_code=code, job_ids=[job])
        if job in rows and rows[job] != row:
            raise RuntimeError('Conflicting accounting: '+job)
        rows[job] = row
    return [rows.get(job, dict(name=name, state='UNKNOWN_ACCOUNTING',
            exit_code='?', job_ids=[job])) for job, name in jobs.items()]


def grf_mean3(selected, curves, output, label):
    import numpy as np

    selected = sorted(selected, key=lambda p: p['seed'])
    if [p['seed'] for p in selected] != [1, 2, 3]:
        raise ValueError('Expected exactly seeds 1/2/3')
    if any(p['win_metric'] != METRIC for p in selected):
        raise ValueError('Expected the GRF test win metric for every seed')
    results = [curves[p['job_name']] for p in selected]
    if any(x[-1] < TARGET for (x, _), _ in results):
        raise ValueError('At least one seed test history ends below 10M')
    start = max(x[0] for (x, _), _ in results)
    grid = np.arange(int(np.ceil(start / 10000)) * 10000, TARGET + 1, 10000)
    if not grid.size:
        raise ValueError('No shared test step grid through 10M')
    aligned = np.stack([np.interp(grid, x, y) for (x, y), _ in results])
    means, stds = aligned.mean(axis=0), aligned.std(axis=0, ddof=1)
    rows = [dict(env_steps=int(step), seed_count=3, **{
        METRIC: float(mean), METRIC+'_std': float(std),
        METRIC+'_lower': float(max(0, mean-std)), METRIC+'_upper': float(min(1, mean+std)),
        **{f'seed_{j+1}/{METRIC}': float(aligned[j, i]) for j in range(3)},
    }) for i, (step, mean, std) in enumerate(zip(grid, means, stds))]
    output.mkdir(parents=True, exist_ok=True)
    inventory = [dict(seed=p['seed'], run_name=p['job_name'], run_id=d['run_id'],
                      attempt_start=d['attempt_start'], source=d['source'],
                      source_start_step=float(x[0]), source_end_step=float(x[-1]), points=len(x))
                 for p, ((x, _), d) in zip(selected, results)]
    raw = output/'seed_curves.csv'
    with raw.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'run_name', 'run_id', 'env_steps', METRIC])
        for item, ((x, y), _) in zip(inventory, results):
            writer.writerows((item['seed'], item['run_name'], item['run_id'], s, v) for s, v in zip(x, y))
    with (output/'seed_inventory.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    # Digest must be stable when a retried backfill reads the same run from cloud
    # instead of disk; original ID and actual scalar data identify the snapshot.
    sources = [{k: v for k, v in item.items() if k not in ('source', 'attempt_start')}
               for item in inventory]
    record = dict(scene=selected[0]['scene'], model=selected[0]['label'], label=label,
                  sources=sources, rows=rows, source_csv=str(raw))
    return record, inventory


def selected_curves(h, repo, runtime, jobs, selected, api_factory):
    import plot_wandb_seed_aggregate as plotter

    charts = h.up.runner.study.charts
    owners = h.integrity.run_ownership(repo, runtime, jobs)
    dirs = h.owned_directories(repo, runtime, jobs)
    local = h.exact_local_index(dirs, jobs, set(jobs.values()))
    sacred_root = runtime/'results/sacred'
    sacred = {}
    import json
    for name, owner in owners.items():
        for raw_path in owner['sacred']:
            path = Path(raw_path)
            if path.is_dir() and not path.is_symlink():
                config = json.loads((path/'config.json').read_text())
                if config.get('wandb_run_name') != name:
                    raise ValueError('Sacred identity differs: '+name)
                stamp = json.loads((path/'run.json').read_text()).get('start_time')
                sacred.setdefault(name, []).append((charts.timestamp(stamp), path))
    result = {}
    for plan in selected:
        name = plan['job_name']
        ids = owners[name]['ids']
        if len(ids) > 1:
            raise ValueError('Multiple W&B attempts within selected job: '+name)
        sacred_exact = {name: [(t, p) for t, p in sacred.get(name, [])
                              if str(p.resolve()) in owners[name]['sacred'] and not p.is_symlink()]}
        curve, details, note = charts.select_latest(plan, METRIC, local, {}, {}, runtime/'wandb', plotter)
        if curve is None or curve[0][-1] < TARGET:
            recovered, meta, info = charts.select_latest(plan, METRIC, {}, {}, sacred_exact, runtime/'wandb', plotter)
            if recovered is not None and (curve is None or recovered[0][-1] > curve[0][-1]):
                curve, details, note = recovered, meta, info
        if curve is None or curve[0][-1] < TARGET:
            try:
                cloud = {name: []}
                for rid in ids:
                    remote = api_factory().run(h.PROJECT+'/'+rid)
                    if remote.id != rid or remote.name != name:
                        raise ValueError('W&B name/ID mismatch: '+name)
                    cloud[name].append(remote)
                recovered, meta, info = charts.select_latest(plan, METRIC, {}, cloud, {}, runtime/'wandb', plotter)
                if recovered is not None and (curve is None or recovered[0][-1] > curve[0][-1]):
                    curve, details, note = recovered, meta, info
            except ValueError:
                raise
            except Exception as exc:
                note = f'{note}; cloud: {exc}'
        if curve is None or curve[0][-1] < TARGET:
            raise ValueError(f'{name}: test end={None if curve is None else curve[0][-1]}; {note}')
        result[name] = curve, details
    return result


def current(api_factory, project, item):
    remote = api_factory().run(project+'/'+item['run_id'])
    summary = remote.summary
    return (remote.config.get('source_digest') == item['digest'] and
            summary.get('env_steps') == TARGET and summary.get('seed_count') == 3 and
            summary.get(METRIC) is not None and
            all(summary.get('seed_{}/{}'.format(i, METRIC)) is not None for i in (1,2,3)))


def run(repo, runtime, upload, h):
    plans = expected_plans()
    rows = accounting(repo)
    active = h.cleanup.live_jobs()
    for row in rows:
        if active.intersection(row['job_ids']):
            row['state'] = 'LIVE_QUEUE'
        print(row['job_ids'][0], row['state'], row['exit_code'], row['name'], flush=True)
    ready = [r for r in rows if r['state'] == 'COMPLETED' and r['exit_code'] == '0:0']
    print('GRF STATUS:', dict(Counter(r['state'] for r in rows)), flush=True)
    if not upload:
        return
    import wandb
    output = runtime/'figures'/(GROUP+'_upload')
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/'upload_report.json'
    report = dict(group=GROUP, jobs=rows, originals=[], groups=[], errors=[])
    api_factory = lambda: wandb.Api(timeout=30)
    def save():
        h.save_report(report_path, report)
    fd = os.open(runtime/'wandb/.gomarl-sync-once.lock',
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Upload lock busy; no changes made by this command')
        save()
        h.upload_ended(repo, runtime, ready, output, report, report_path, api_factory)
        ended = {r['name']: r['job_ids'][0] for r in ready}
        for scene, _ in MAPS:
            for model in MODELS:
                selected = [p for p in plans if p['scene'] == scene and p['label'] == model]
                if any(p['job_name'] not in ended for p in selected):
                    print('DEFER MEAN3:', scene, model, flush=True)
                    continue
                try:
                    jobs = {ended[p['job_name']]: p['job_name'] for p in selected}
                    if h.cleanup.live_jobs().intersection(jobs):
                        raise RuntimeError('Selected job became active')
                    curves = selected_curves(h, repo, runtime, jobs, selected, api_factory)
                    record, inventory = grf_mean3(selected, curves, output/(scene+'__'+model), model)
                    rid, digest = h.up.scalar_identity(record, h.PROJECT)
                    item = dict(scene=scene, model=model, run_id=rid, digest=digest,
                                end_step=TARGET, status='pending', inventory=inventory)
                    report['groups'].append(item)
                    try:
                        present = current(api_factory, h.PROJECT, item)
                    except Exception:
                        present = False
                    if not present:
                        h.up.upload_model(record, h.PROJECT, output/'mean3', wandb)
                    print('MEAN3 UPLOAD/CHECK:', scene, model, rid, flush=True)
                except Exception as exc:
                    report['errors'].append(scene+' '+model+': '+str(exc))
                    print('GROUP ERROR:', scene, model, str(exc), flush=True)
                save()
        # One final independent read after all uploads, not immediate repeated writes.
        for item in report['groups']:
            try:
                if current(api_factory, h.PROJECT, item):
                    item['status'] = 'verified'
                print('MEAN3', item['status'].upper(), item['scene'], item['model'],
                      'https://wandb.ai/'+h.PROJECT+'/runs/'+item['run_id'], flush=True)
            except Exception as exc:
                item['verification_error'] = str(exc)
            save()
        for item in report['originals']:
            if item.get('status') == 'unverified' and item.get('run_id'):
                try:
                    remote = api_factory().run(h.PROJECT+'/'+item['run_id'])
                    if (remote.name == item['name'] and remote.lastHistoryStep is not None and
                            remote.lastHistoryStep >= item['last_step']):
                        item['status'] = 'cloud history endpoint verified on final check'
                except Exception:
                    pass
        save()
    originals = sum(i.get('status') != 'unverified' for i in report['originals'])
    means = sum(i['status'] == 'verified' for i in report['groups'])
    print('RESULT originals={}/27 mean3={}/9; REPORT: {}'.format(originals, means, report_path), flush=True)
    print('No training submitted; no data deleted.', flush=True)
    if originals != 27 or means != 9 or report['errors']:
        raise SystemExit('PARTIAL: preserve all local data; inspect report; do not assume upload verified')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--job-repo', type=Path, default=JOB_REPO)
    parser.add_argument('--runtime-root', type=Path, default=RUNTIME)
    args = parser.parse_args(argv)
    if (args.runtime_root.resolve(strict=True) != args.runtime_root or
            any(p.is_symlink() or not p.is_dir() for p in
                (args.runtime_root/'wandb', args.runtime_root/'ozstar_logs'))):
        raise SystemExit('Unsafe runtime')
    if args.upload and not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run --upload in a compute allocation; status-only is permitted on login nodes')
    run(args.job_repo.resolve(strict=True), args.runtime_root, args.upload, load_helpers())


if __name__ == '__main__':
    main()
