#!/usr/bin/env python3
"""One pass for ONLY the fifteen 5m6m one-fill jobs: raw runs, figures, mean3."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import ozstar_submit_5m6m_onefill_gate_15jobs as cohort
import plot_5m6m_8m9m_qmix_variants_3seeds as study
from ozstar_calibration_suite_sync import sync_once, manifest_jobs, accounting, TERMINAL
from wandb_three_seed_scalar_runs import aggregate_models, upload_model

SUBDIR = '5m6m_onefill_20261019_3seeds'
RESULTS_TITLE = 'One-fill gate ablations (October 19 cohort only)'


def original_scope(repo, runtime, expected):
    jobs = {job: name for job, name in manifest_jobs(runtime / 'ozstar_logs').items() if name in expected}
    if len(jobs) != len(expected) or set(jobs.values()) != expected:
        raise RuntimeError('Submission manifests do not identify exactly these {} jobs; refusing to guess IDs'.format(len(expected)))
    states = accounting(jobs, repo)
    if set(states) != set(jobs) or any(row['state'] not in TERMINAL for row in states.values()):
        raise RuntimeError('Some requested jobs lack confirmed terminal accounting in this checkout')
    return {job: dict(name=name, **states[job]) for job, name in jobs.items()}


def local_figures(runtime, project, output):
    study.configure()
    c = study.charts
    plans = cohort.build_plans(c.ROOT)
    c.build_plans = lambda root: plans
    selected = {p['scene'] for p in plans}
    c.SCENES = {scene: settings for scene, settings in c.SCENES.items() if scene in selected}
    c.SCENE_MODELS = {scene: tuple(label for label in cohort.LABELS
        if any(p['scene'] == scene and p['label'] == label for p in plans)) for scene in c.SCENES}
    if set(c.SCENES) != selected:
        raise RuntimeError('Cohort contains an unsupported scene')
    c.GROUP = cohort.GROUP + '_results'
    c.RESULTS_TITLE = RESULTS_TITLE
    previous = sys.argv
    try:
        sys.argv = [__file__, '--runtime-root', str(runtime), '--project', project,
            '--output-dir', str(output), '--retained-figure-root', str(runtime / 'figures'),
            '--local-only', '--no-upload', '--mean-window', '10']
        c.main()
    finally:
        sys.argv = previous
    with (output / 'seed_inventory.csv').open() as handle:
        inventory = list(csv.DictReader(handle))
    if {row['run_name'] for row in inventory} != {p['job_name'] for p in plans}:
        raise RuntimeError('Figure inventory does not exactly match the requested cohort')
    outputs = {scene: [output / (scene + '_' + suffix) for suffix in
        ('three_seed.png', 'three_seed.pdf', 'individual_seeds.png')] for scene in c.SCENES}
    return plans, inventory, outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=RESULTS_TITLE + ': original runs, figures and unsmoothed mean3')
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get(
        'RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--no-upload', action='store_true', help='Local plots/inventory only; no cloud or scheduler calls')
    parser.add_argument('--skip-original-sync', action='store_true', help='Only after original-run backfill has been verified')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    runtime = args.runtime_root.resolve()
    output = runtime / 'figures' / SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    planned = cohort.build_plans(repo)
    expected = {p['job_name'] for p in planned}
    expected_models = {(p['scene'], p['label']) for p in planned}
    model_count = len(expected_models)
    report = dict(cohort=cohort.GROUP, expected_runs=sorted(expected), dry_run=args.no_upload,
                  original_sync=None, figures=None, mean3=[], skipped=[], errors=[])
    if not args.no_upload and not args.skip_original_sync:
        os.environ.setdefault('SYNC_TIMEOUT', '600')
        report['original_scope'] = original_scope(repo, runtime, expected)
        print('STAGE 1/3: audit/backfill original W&B IDs for ONLY these {} ended jobs'.format(len(expected)), flush=True)
        audit = sync_once(repo, runtime, sys.executable, ended_only=True,
                          verify_existing=True, wanted_names=expected)
        report['original_sync'] = audit
        (output / 'upload_report.json').write_text(json.dumps(report, indent=2))
        if audit['busy']:
            raise SystemExit('Shared sync lock busy; nothing uploaded by this command. Retry after that upload finishes.')
        if audit['failed'] or audit['missing_local']:
            report['errors'].append('Some original-run uploads failed or have no local source; see original_sync')
        if audit['checked'] < len(expected):
            report['errors'].append('Fewer than {} original local runs audited; do not assume every job was uploaded'.format(len(expected)))
    print('STAGE 2/3: local-only single-seed panels and three-seed figures for {} map/model groups'.format(model_count), flush=True)
    plans, inventory, outputs = local_figures(runtime, args.project, output)
    report['inventory'] = inventory
    incomplete = [row['run_name'] for row in inventory if row['coverage'] != 'complete']
    if incomplete:
        report['incomplete_test_histories'] = incomplete
        report['errors'].append('Some test curves are missing or below 10M despite scheduler completion')
    if not args.no_upload:
        try:
            study.upload(args.project, output, outputs, inventory, 10)
            report['figures'] = json.loads((output / 'model_panel_upload.json').read_text())
        except Exception as exc:
            # Keep scalar uploads independent of a workspace/layout failure.
            report['errors'].append('Figure/layout upload: ' + str(exc))
            metadata = output / 'model_panel_upload.json'
            if metadata.exists():
                report['figures'] = json.loads(metadata.read_text())
            print('Figure/layout stage incomplete; still attempting mean3: ' + str(exc), flush=True)
    print('STAGE 3/3: selectable UNSMOOTHED mean3 runs ({} map/model groups, seeds 1/2/3)'.format(model_count), flush=True)
    records, skipped = aggregate_models(output / 'seed_curves.csv', plans, study.charts.SCENES)
    report['skipped'] = skipped
    if {(r['scene'], r['model']) for r in records} != expected_models:
        report['errors'].append('Not all {} map/model groups have readable seeds 1/2/3; missing mean3 models are not fabricated'.format(model_count))
    for record in records:
        if not args.no_upload:
            try:
                import wandb
                report['mean3'].append(upload_model(record, args.project, output / 'mean3', wandb))
            except Exception as exc:
                report['errors'].append('Mean3 {}: {}'.format(record['model'], exc))
    report['mean3_plan'] = [dict(scene=r['scene'], model=r['model'], seeds=[s['seed'] for s in r['sources']],
                               end_step=r['rows'][-1]['env_steps']) for r in records]
    path = output / 'upload_report.json'
    path.write_text(json.dumps(report, indent=2))
    print('Original audit: ' + json.dumps(report['original_sync']), flush=True)
    print('Mean3: {} uploaded; {} prepared; {} skipped'.format(len(report['mean3']), len(records), len(skipped)), flush=True)
    print('Detailed report: ' + str(path), flush=True)
    if report['errors']:
        raise SystemExit('PARTIAL/INCOMPLETE: ' + '; '.join(report['errors']))
    return report


if __name__ == '__main__':
    main()
