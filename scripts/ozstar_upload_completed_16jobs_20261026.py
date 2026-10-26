#!/usr/bin/env python3
"""Backfill exactly 16 successful jobs: 15 originals/5 mean3 + 1 raw seed.

No training, deletion, tmux changes, running/pending jobs or historical mixing.
--no-upload generates local figures only and makes no cloud/scheduler calls.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import ozstar_submit_qmix_ones_five_maps_30jobs as paired
import ozstar_submit_corridor_qmix_3jobs as corridor
import ozstar_submit_5m6m_mixer_state_refresh_slow_6jobs as slow
import ozstar_upload_5m6m_onefill_results as runner
import ozstar_calibration_suite_sync as sync

GROUP = 'completed_qmix_baselines_15jobs_20261026'
SUBDIR = 'completed_qmix_baselines_20261026_3seeds'
TITLE = 'QMIX and constant-input baselines: October 26 completed jobs only'
SOURCE_GROUPS = tuple(dict.fromkeys((paired.SMAC_GROUP, paired.GROUP, corridor.GROUP, slow.GROUP)))
LABELS = {
    'qmix': ('QMIX / state-conditioned mixer / historical fixed head (10M)', '#7f7f7f'),
    'qmix_state_ones': ('QMIX / all-ones mixer condition / historical fixed head (10M)', '#d627a5'),
}
SCENES = {
    'smac_mmm2': ('MMM2', 'test_battle_won_mean'),
    'smac_3s5z': ('3 Stalkers + 5 Zealots vs. 3 Stalkers + 6 Zealots', 'test_battle_won_mean'),
    'smac_corridor': ('Corridor', 'test_battle_won_mean'),
}
# Exact IDs/names from the user's sacct output, not inferred from squeue order.
EXPECTED_JOBS = {
    '19185607': 'smac_mmm2_qmix_10m_s1_stateones20261025',
    '19185608': 'smac_mmm2_qmix_10m_s2_stateones20261025',
    '19185609': 'smac_mmm2_qmix_10m_s3_stateones20261025',
    '19185610': 'smac_mmm2_qmix_state_ones_10m_s1_stateones20261025',
    '19185611': 'smac_mmm2_qmix_state_ones_10m_s2_stateones20261025',
    '19185612': 'smac_mmm2_qmix_state_ones_10m_s3_stateones20261025',
    '19185613': 'smac_3s5z_qmix_10m_s1_stateones20261025',
    '19185614': 'smac_3s5z_qmix_10m_s2_stateones20261025',
    '19185615': 'smac_3s5z_qmix_10m_s3_stateones20261025',
    '19185616': 'smac_3s5z_qmix_state_ones_10m_s1_stateones20261025',
    '19185617': 'smac_3s5z_qmix_state_ones_10m_s2_stateones20261025',
    '19185618': 'smac_3s5z_qmix_state_ones_10m_s3_stateones20261025',
    '19217303': 'smac_corridor_qmix_baseline_10m_s1_valuediag',
    '19217304': 'smac_corridor_qmix_baseline_10m_s2_valuediag',
    '19217305': 'smac_corridor_qmix_baseline_10m_s3_valuediag',
    '19246004': 'smac_5m6m_qmix_state_refresh_5000k_10m_s1_refresh20261026',
}
EXTRA_JOB = '19246004'


def build_plans(repo):
    plans = paired.build_plans(repo, 'smac') + corridor.build_plans(repo)
    expected = set(EXPECTED_JOBS.values()) - {EXPECTED_JOBS[EXTRA_JOB]}
    if len(plans) != 15 or {p['job_name'] for p in plans} != expected:
        raise RuntimeError('Current baseline plans differ from the authorized fifteen jobs')
    if len({(p['scene'], p['label']) for p in plans}) != 5:
        raise RuntimeError('Expected exactly five three-seed map/model groups')
    return plans


def completed_scope(repo, runtime, expected):
    expected = set(expected)
    wanted = {job: name for job, name in EXPECTED_JOBS.items() if name in expected}
    if set(wanted.values()) != expected:
        raise RuntimeError('Requested jobs fall outside the authorized sixteen-job whitelist')
    manifested = {job: name for job, name in sync.manifest_jobs(runtime/'ozstar_logs').items()
                  if name in expected}
    if manifested != wanted:
        raise RuntimeError('Submission receipts differ from the supplied exact job IDs; refusing to guess')
    states = sync.accounting(wanted, repo)
    if set(states) != set(wanted) or any(
            row['state'] != 'COMPLETED' or row['exit_code'] != '0:0' for row in states.values()):
        raise RuntimeError('Requested jobs must still be COMPLETED / 0:0 in this checkout')
    return {job: dict(name=name, **states[job]) for job, name in wanted.items()}


def extra_original(repo, runtime):
    expected = {EXPECTED_JOBS[EXTRA_JOB]}
    scope = completed_scope(repo, runtime, expected)
    os.environ.setdefault('SYNC_TIMEOUT', '600')
    audit = runner.sync_once(repo, runtime, sys.executable, ended_only=True,
                             verify_existing=True, wanted_names=expected)
    if (audit['busy'] or audit['failed'] or audit['missing_local'] or
            audit['checked'] < 1 or audit['current'] + audit['uploaded'] < 1):
        raise RuntimeError('Refresh seed 1 original backfill incomplete: ' + json.dumps(audit))
    return dict(scope=scope, audit=audit, mean3='NOT CREATED: only seed 1 requested')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get(
        'RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args(argv)
    repo, runtime = Path(__file__).resolve().parents[1], args.runtime_root.resolve()
    output = runtime/'figures'/SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    plans = build_plans(repo)
    extras = [p for p in slow.build_plans(repo) if p['job_name'] == EXPECTED_JOBS[EXTRA_JOB]]
    if len(extras) != 1:
        raise RuntimeError('Authorized extra refresh seed no longer matches its training plan')
    models = {scene: tuple(dict.fromkeys(p['label'] for p in plans if p['scene'] == scene))
              for scene in SCENES}
    study = runner.study
    previous = (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS, study.configure)
    configure = study.configure

    def configure_maps():
        configure()
        study.charts.SCENES = dict(SCENES)

    report = dict(expected_jobs=EXPECTED_JOBS, dry_run=args.no_upload,
                  baselines=None, refresh_seed1=None, errors=[])
    forwarded = ['--runtime-root', str(runtime), '--project', args.project]
    if args.no_upload:
        forwarded.append('--no-upload')
    try:
        runner.cohort = SimpleNamespace(GROUP=GROUP, LABELS=tuple(LABELS), build_plans=build_plans)
        runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope = SUBDIR, TITLE, completed_scope
        sync.GROUP, sync.GROUPS = GROUP, SOURCE_GROUPS
        study.LABELS, study.SCENE_MODELS, study.configure = LABELS, models, configure_maps
        print('BATCH 1/2: fifteen exact completed originals and five mean3', flush=True)
        try:
            report['baselines'] = runner.main(forwarded)
        except (Exception, SystemExit) as exc:
            report['errors'].append('Baseline batch: ' + str(exc))
            print('BASELINE BATCH INCOMPLETE: ' + str(exc), flush=True)
        print('BATCH 2/2: refresh stage=5M, seed=1, job=19246004; ORIGINAL ONLY', flush=True)
        if args.no_upload:
            report['refresh_seed1'] = dict(job=EXTRA_JOB, status='preview only; no cloud/scheduler calls',
                                          mean3='NOT CREATED: only seed 1 requested')
        else:
            try:
                report['refresh_seed1'] = extra_original(repo, runtime)
            except Exception as exc:
                report['errors'].append('Refresh seed 1: ' + str(exc))
                print('EXTRA ORIGINAL INCOMPLETE: ' + str(exc), flush=True)
    finally:
        (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
         sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS, study.configure) = previous
    path = output/'completed16_upload_report.json'
    path.write_text(json.dumps(report, indent=2))
    print('Sixteen-job report: ' + str(path), flush=True)
    if report['errors']:
        raise SystemExit('PARTIAL/INCOMPLETE: ' + '; '.join(report['errors']))
    print('LOCAL VALIDATION COMPLETE' if args.no_upload else
          'COMPLETE: 16 original backfills verified (scalar progress); 5 mean3; no refresh mean3', flush=True)
    return report


if __name__ == '__main__':
    main()
