#!/usr/bin/env python3
"""One pass: ended original-run audit/backfill, two-map figures, selectable mean3 runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get(
        'RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--skip-ended-sync', action='store_true')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    report = {}
    if not args.skip_ended_sync and not args.no_upload:
        import plot_5m6m_8m9m_qmix_variants_3seeds as study
        from ozstar_calibration_suite_sync import sync_once
        wanted = {p['job_name'] for p in study.build_plans(repo)}
        os.environ.setdefault('SYNC_TIMEOUT', '600')
        print('STAGE 1: check terminal original runs and backfill missing/behind records', flush=True)
        report = sync_once(repo.resolve(), args.runtime_root.resolve(), sys.executable,
                           ended_only=True, verify_existing=True, wanted_names=wanted)
        audit_path = args.runtime_root / 'ozstar_logs/last_ended_cloud_audit.json'
        audit_path.write_text(json.dumps(report, indent=2))
        if report['busy']:
            raise SystemExit('Shared sync lock busy; no backfill performed. Retry after current upload round.')
    for map_name in ('5m6m', '8m9m'):
        print('STAGE 2: refresh local {} figures and available seed CSVs'.format(map_name), flush=True)
        command = [sys.executable, str(repo / 'scripts' / ('plot_' + map_name + '_local_available_3seeds.py')),
                   '--runtime-root', str(args.runtime_root), '--project', args.project]
        if args.no_upload:
            command.append('--no-upload')
        subprocess.run(command, check=True)
    print('STAGE 3: upload one selectable, unsmoothed mean3 analysis run per model', flush=True)
    from wandb_three_seed_scalar_runs import main as scalar_main
    command = ['--runtime-root', str(args.runtime_root), '--project', args.project]
    if args.no_upload:
        command.append('--no-upload')
    scalar_main(command)
    if report:
        print('Original-run audit summary: ' + json.dumps(report, sort_keys=True), flush=True)
        if report['failed'] or report['missing_local']:
            print('WARNING: some original runs remain unverified or have no local source; '
                  'aggregate figures do not restore missing training logs.', flush=True)
        if report['failed']:
            raise SystemExit('Aggregate uploads finished, but some original-run backfills failed; see audit report')


if __name__ == '__main__':
    main()
