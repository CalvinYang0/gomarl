#!/usr/bin/env python3
"""Audit the three recent cohorts (21 jobs); backfill originals and full mean3.

No training, cleanup, polling, tmux changes or full-project cloud scan.
--status-only reads receipts and sacct, without writing files or contacting W&B.
Failed terminal jobs upload available original records only, never full mean3.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import ozstar_submit_5m6m_mixer_state_refresh_slow_6jobs as slow
import ozstar_submit_qtot_sum_four_maps_12jobs as transfer
import ozstar_submit_corridor_ones_3jobs as ones
import ozstar_upload_5m6m_onefill_results as runner
import ozstar_calibration_suite_sync as sync
from wandb_three_seed_scalar_runs import aggregate_models, upload_model

GROUP = 'recent_21jobs_backfill_20261028'
SOURCES = (slow, transfer, ones)
SCENES = {
    'smac_5m6m': ('5 Marines vs. 6 Marines', 'test_battle_won_mean'),
    'smac_8m9m': ('8 Marines vs. 9 Marines', 'test_battle_won_mean'),
    'smac_corridor': ('Corridor', 'test_battle_won_mean'),
    'smac_mmm2': ('MMM2', 'test_battle_won_mean'),
    'smac_3s5z': ('3 Stalkers + 5 Zealots vs. 3 Stalkers + 6 Zealots', 'test_battle_won_mean'),
}
LABELS = {
    'qmix_state_refresh_5000k': ('QMIX mixer state refresh / stage=5M (10M)', '#d95f02'),
    'qmix_state_refresh_10000k': ('QMIX mixer state refresh / stage=10M (10M)', '#7570b3'),
    'qmix_qtot_sum_p10': ('QMIX Qtot sum / all-ones main + one-fill p0=0.1 secondary (10M)', '#e7298a'),
    'qmix_state_ones': ('QMIX all-ones mixer condition / historical fixed head (10M)', '#17becf'),
}


@contextmanager
def source_scope():
    previous = sync.GROUP, sync.GROUPS
    sync.GROUP, sync.GROUPS = GROUP, tuple(s.GROUP for s in SOURCES)
    try:
        yield
    finally:
        sync.GROUP, sync.GROUPS = previous


def build_plans(repo):
    plans = [p for source in SOURCES for p in source.build_plans(repo)]
    if len(plans) != 21 or len({p['job_name'] for p in plans}) != 21:
        raise RuntimeError('Expected exactly the three authorized cohorts / 21 names')
    return plans


def audit_jobs(repo, runtime, plans):
    expected = {p['job_name'] for p in plans}
    try:
        jobs = {j: n for j, n in sync.manifest_jobs(runtime/'ozstar_logs').items() if n in expected}
    except RuntimeError as exc:
        if 'No submitted/retained job IDs' not in str(exc):
            raise
        jobs = {}
    states = sync.accounting(jobs, repo) if jobs else {}
    rows = []
    for p in plans:
        ids = sorted((j for j, n in jobs.items() if n == p['job_name']), key=int)
        row = dict(name=p['job_name'], scene=p['scene'], model=p['label'], seed=p['seed'], job_ids=ids)
        if len(ids) != 1:
            row.update(state='MISSING_RECEIPT' if not ids else 'AMBIGUOUS_RECEIPTS', exit_code='?')
        else:
            row.update(states.get(ids[0], dict(state='UNKNOWN_ACCOUNTING_OR_WORKDIR', exit_code='?')))
        rows.append(row)
        print('{} {:30} exit={} {}'.format(','.join(ids) or '-', row['state'], row['exit_code'], row['name']), flush=True)
    return rows


def full_models(plans, rows):
    by_name = {r['name']: r for r in rows}
    ready, skipped = [], []
    keys = list(dict.fromkeys((p['scene'], p['label']) for p in plans))
    for scene, label in keys:
        selected = [p for p in plans if (p['scene'], p['label']) == (scene, label)]
        if {p['seed'] for p in selected} == {1, 2, 3} and all(
                by_name[p['job_name']]['state'] == 'COMPLETED' and
                by_name[p['job_name']]['exit_code'] == '0:0' for p in selected):
            ready.append(selected)
        else:
            skipped.append(dict(scene=scene, model=label, reason='requires three COMPLETED / 0:0 receipts'))
    return ready, skipped


def render_model(runtime, project, output, plans):
    study = runner.study
    previous = (runner.cohort, runner.RESULTS_TITLE, study.LABELS, study.SCENE_MODELS, study.configure)
    configure = study.configure
    scene, label = plans[0]['scene'], plans[0]['label']

    def configure_maps():
        configure()
        study.charts.SCENES = {scene: SCENES[scene]}

    try:
        runner.cohort = SimpleNamespace(GROUP=GROUP, LABELS=(label,), build_plans=lambda repo: plans)
        runner.RESULTS_TITLE = 'Recent completed jobs: ' + LABELS[label][0]
        study.LABELS, study.SCENE_MODELS = {label: LABELS[label]}, {scene: (label,)}
        study.configure = configure_maps
        selected, inventory, _ = runner.local_figures(runtime, project, output)
        if len(inventory) != 3 or any(r['coverage'] != 'complete' for r in inventory):
            raise RuntimeError('Three successful jobs but local/retained test histories are not all 10M; no full mean3 published')
        records, skipped = aggregate_models(output/'seed_curves.csv', selected, study.charts.SCENES)
        if skipped or len(records) != 1 or records[0]['rows'][-1]['env_steps'] != 10000000:
            raise RuntimeError('No complete three-seed shared 10M interval; refusing to fabricate mean3')
        return records[0], inventory
    finally:
        (runner.cohort, runner.RESULTS_TITLE, study.LABELS, study.SCENE_MODELS, study.configure) = previous


def scalar_identity(record, project):
    value = dict(version=1, project=project, scene=record['scene'], model=record['model'],
                 sources=record['sources'], rows=record['rows'])
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return 'mean3' + digest[:20], digest


def scalar_current(api, project, rid, digest):
    run = api.run(project + '/' + rid)
    return (run.config.get('source_digest') == digest and
            run.summary.get('env_steps', -1) >= 10000000 and run.summary.get('seed_count') == 3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get(
        'RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args(argv)
    repo, runtime = Path(__file__).resolve().parents[1], args.runtime_root.resolve()
    plans = build_plans(repo)
    with source_scope():
        rows = audit_jobs(repo, runtime, plans)
        ready, skipped = full_models(plans, rows)
        counts = {}
        for row in rows:
            counts[row['state']] = counts.get(row['state'], 0) + 1
        print('JOB STATUS COUNTS: ' + json.dumps(counts, sort_keys=True), flush=True)
        report = dict(jobs=rows, state_counts=counts, skipped_mean3=skipped, originals=None, mean3=[], errors=[])
        if args.status_only:
            print('STATUS ONLY: {} / {} models have three successful jobs; no uploads'.format(
                len(ready), len(ready)+len(skipped)), flush=True)
            return report
        output = runtime/'figures'/GROUP
        output.mkdir(parents=True, exist_ok=True)
        report_path = output/'upload_report.json'
        save = lambda: report_path.write_text(json.dumps(report, indent=2))
        save()
        unknown = [r for r in rows if r['state'].startswith(('UNKNOWN', 'AMBIGUOUS', 'MISSING'))]
        if unknown:
            report['errors'].append('Some jobs have missing/ambiguous receipts or unverified accounting; see jobs')
        ended = {r['name'] for r in rows if len(r['job_ids']) == 1 and r['state'] in sync.TERMINAL}
        if ended:
            os.environ.setdefault('SYNC_TIMEOUT', '600')
            result = runner.sync_once(repo, runtime, sys.executable, ended_only=True,
                                      verify_existing=True, wanted_names=ended)
            report['originals'] = result
            save()
            if result['busy']:
                raise SystemExit('Shared upload/cleanup lock busy; no upload by this command; retry later')
            if (result['failed'] or result['missing_local'] or
                    result['current'] + result['uploaded'] < len(ended)):
                report['errors'].append('Original backfill incomplete; missing local source is NOT proof of missing cloud data')
        else:
            print('No confirmed terminal jobs to backfill', flush=True)
        # Exact model groups only; partial/failed groups never become a complete mean3.
        import wandb
        api = wandb.Api(timeout=30) if ready else None
        for selected in ready:
            scene, model = selected[0]['scene'], selected[0]['label']
            try:
                record, inventory = render_model(runtime, args.project, output/(scene+'__'+model), selected)
                rid, digest = scalar_identity(record, args.project)
                try:
                    current = scalar_current(api, args.project, rid, digest)
                except wandb.errors.CommError:
                    current = False
                if current:
                    result = dict(scene=scene, model=model, run_id=rid, status='cloud current; skipped')
                    print('MEAN3 CURRENT: ' + args.project+'/'+rid, flush=True)
                else:
                    result = upload_model(record, args.project, output/'mean3', wandb)
                    if not scalar_current(wandb.Api(timeout=30), args.project, rid, digest):
                        raise RuntimeError('Mean3 finish returned but cloud scalar/digest verification failed')
                    result['status'] = 'uploaded and verified'
                report['mean3'].append(dict(**result, inventory=inventory))
            except Exception as exc:
                report['errors'].append('{} {}: {}'.format(scene, model, exc))
                print('INCOMPLETE {} {}: {}'.format(scene, model, exc), flush=True)
            save()
        print('Report: ' + str(report_path), flush=True)
        print('Mean3 verified/current: {}; not ready: {}; errors: {}'.format(
            len(report['mean3']), len(skipped), len(report['errors'])), flush=True)
        if report['errors']:
            raise SystemExit('PARTIAL: see report; no training or data deletion performed')
        return report


if __name__ == '__main__':
    main()
