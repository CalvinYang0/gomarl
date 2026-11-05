#!/usr/bin/env python3
"""Synthetic exact12 retry/curve fixtures and mock Slurm/cloud; no real services."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import ozstar_upload_projected_fastrefresh_results as suite


def main():
    up, integrity = suite.uploader, suite.integrity
    repo = Path(__file__).resolve().parents[1]
    plans = suite.build_plans(repo)
    assert len(plans) == 12 and len(suite.LABELS) == 1
    original = up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-projected-fast-upload-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        (root/'wandb').mkdir()
        old = {p['job_name']: str(71000000+i) for i, p in enumerate(plans)}
        new = {p['job_name']: str(81000000+i) for i, p in enumerate(plans)}
        for tag, jobs in (('old', old), ('new', new)):
            (logs/(suite.cohort.GROUP+'_'+tag+'.json')).write_text(json.dumps(dict(
                group=suite.cohort.GROUP, plans=plans, submitted=jobs)))
        # Parent model receipts must not enter current cohort selection.
        (logs/(integrity.refresh.GROUP+'_unrelated.json')).write_text(json.dumps(dict(
            group=integrity.refresh.GROUP, plans=[dict(job_name='parent')], submitted={'parent': '1'})))
        states = {}
        for p in plans:
            name = p['job_name']
            states[old[name]] = dict(state='COMPLETED', exit_code='0:0', elapsed='15:00:00',
                submitted='2026-11-04T02:00:00', started='2026-11-04T02:01:00', ended='2026-11-04T17:01:00')
            states[new[name]] = dict(state='COMPLETED', exit_code='0:0', elapsed='15:00:00',
                submitted='2026-11-04T20:00:00', started='2026-11-04T20:01:00', ended='2026-11-05T11:01:00')
        def account(jobs, checkout):
            assert checkout == repo and set(jobs) == set(old.values()) | set(new.values())
            return copy.deepcopy(states)
        # Empty live queue cannot distinguish these statuses; audit exact accounting.
        states[new[plans[-1]['job_name']]].update(state='FAILED', exit_code='1:0')
        states[new[plans[6]['job_name']]].update(state='RUNNING')
        states[new[plans[7]['job_name']]].update(state='RUNNING')
        with redirect_stdout(io.StringIO()), patch.object(integrity, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once') as raw, patch.object(up, 'upload_model') as mean, patch('wandb.Api') as api:
            report = suite.main(['--runtime-root', str(root), '--status-only'])
            raw.assert_not_called(); mean.assert_not_called(); api.assert_not_called()
        assert report['state_counts'] == dict(COMPLETED=9, RUNNING=2, FAILED=1)
        assert len(report['skipped_mean3']) == 2 and not (root/'figures').exists()
        assert all(r['job_ids'] == [new[r['name']]] for r in report['jobs'])
        # Later state only, never select an older successful attempt over a newer failure.
        for p in plans:
            states[new[p['job_name']]].update(state='COMPLETED', exit_code='0:0')
            for tag, win in (('current', .7), ('old', .99)):
                path = root/'results/sacred'/p['map_name']/(p['label']+tag)/str(p['seed'])
                path.mkdir(parents=True)
                (path/'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
                (path/'run.json').write_text(json.dumps(dict(start_time='2026-11-04T20:01:00Z'
                    if tag == 'current' else '2026-11-05T20:01:00Z')))
                (path/'info.json').write_text(json.dumps(dict(
                    test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                    test_battle_won_mean=[.01, .1, .4, win, win])))
                if tag == 'current':
                    rid = 'current'+new[p['job_name']]
                    (logs/('{}_{}.err'.format(p['job_name'], new[p['job_name']]))).write_text(
                        'Saving to FileStorageObserver in {}.\nStarted run with ID "{}"\n'
                        'W&B run data saved in {}\n'.format(path.parent, p['seed'],
                            root/'wandb'/('offline-run-20261104_200100-'+rid)))
        def raw_sync(*args, **kwargs):
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names={p['job_name'] for p in plans})
            assert up.sync.manifest_jobs(logs) == {new[p['job_name']]: p['job_name'] for p in plans}
            return dict(busy=False, checked=12, current=4, uploaded=8, failed=0, missing_local=[])
        def upload(record, project, output, wandb):
            assert record['model'] == suite.cohort.LABEL and record['rows'][-1]['env_steps'] == 10000000
            assert record['rows'][-1]['test_battle_won_mean'] < .8, 'Wrong attempt leaked into mean3'
            assert all(s['run_id'].startswith('sacred:') for s in record['sources'])
            return dict(scene=record['scene'], model=record['model'], run_id='SYNTHETIC')
        with redirect_stdout(io.StringIO()) as output, patch.object(integrity, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(up, 'upload_model', side_effect=upload) as mean, \
                patch.object(up, 'scalar_current', side_effect=[False, True]*4), patch('wandb.Api'), \
                patch.object(up.runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No project scan')):
            try: report = suite.main(['--runtime-root', str(root)])
            except SystemExit:
                print(output.getvalue(), file=sys.stderr)
                raise
        assert mean.call_count == 4 and len(report['mean3']) == 4 and not report['errors']
        assert not report['skipped_mean3']
        assert len(list((root/'figures'/suite.GROUP).rglob('*three_seed.png'))) == 4
        assert len(list((root/'figures'/suite.GROUP).rglob('seed_inventory.csv'))) == 4
        # Idempotent mean3: current cloud identity/digest skips writes.
        with redirect_stdout(io.StringIO()), patch.object(integrity, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(up, 'upload_model') as mean, patch.object(up, 'scalar_current', return_value=True), \
                patch('wandb.Api'):
            report = suite.main(['--runtime-root', str(root)])
            mean.assert_not_called()
        assert len(report['mean3']) == 4
        # Missing latest-job test endpoint blocks THAT map, not the other three.
        selected = plans[0]
        path = root/'results/sacred'/selected['map_name']/(selected['label']+'current')/str(selected['seed'])/'info.json'
        path.write_text(json.dumps(dict(test_battle_won_mean_T=[10000, 1000000], test_battle_won_mean=[.01, .1])))
        with redirect_stdout(io.StringIO()), patch.object(integrity, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(up, 'upload_model') as mean, patch.object(up, 'scalar_current', return_value=True), patch('wandb.Api'):
            try: suite.main(['--runtime-root', str(root)])
            except SystemExit as exc: assert 'PARTIAL' in str(exc)
            else: raise AssertionError('Short seed was called full10M')
            mean.assert_not_called()
        report = json.loads((root/'figures'/suite.GROUP/'upload_report.json').read_text())
        assert len(report['mean3']) == 3 and len(report['errors']) == 1
        assert original == (up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs)
    print('PASS: exact12 / four maps, latest retry not best result, parent exclusion, read-only status, '
          'real synthetic four-map figures/CSV, full10M gating, original scope, idempotent mean3, no cloud/jobs/deletion')


if __name__ == '__main__':
    main()
