#!/usr/bin/env python3
"""Synthetic files and mocked Slurm/cloud only; no real network or jobs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import ozstar_upload_recent_21jobs_20261028 as suite


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = sorted(suite.build_plans(repo), key=lambda p: (p['scene'], p['label'], p['seed']))
    assert len(plans) == 21
    assert {p['seed'] for p in plans} == {1, 2, 3}
    before = suite.sync.GROUP, suite.sync.GROUPS
    study = suite.runner.study
    old_render = (suite.runner.cohort, study.LABELS, study.SCENE_MODELS, study.configure)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-recent21-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        ids = {p['job_name']: str(80000000+i) for i, p in enumerate(plans)}
        for source in suite.SOURCES:
            selected = source.build_plans(repo)
            (logs/(source.GROUP+'_SYNTHETIC.json')).write_text(json.dumps(dict(
                group=source.GROUP, plans=selected, submitted={p['job_name']: ids[p['job_name']] for p in selected})))
        for plan in plans[:3]:
            path = root/'results/sacred'/plan['map_name']/plan['label']/str(plan['seed'])
            path.mkdir(parents=True)
            (path/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
            (path/'run.json').write_text(json.dumps(dict(start_time='2026-10-27T00:00:00Z')))
            (path/'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                test_battle_won_mean=[.01, .3, .7, .80+plan['seed']*.01, .82])))
        states = {ids[p['job_name']]: dict(state='PENDING', exit_code='0:0', elapsed='00:00:00') for p in plans}
        for p in plans[:8]:
            states[ids[p['job_name']]].update(state='COMPLETED', elapsed='10:00:00')
        states[ids[plans[3]['job_name']]].update(state='FAILED', exit_code='1:0')
        states[ids[plans[8]['job_name']]].update(state='RUNNING')

        def account(jobs, checkout):
            assert checkout == repo
            assert set(jobs) == set(ids.values())
            return states

        with redirect_stdout(io.StringIO()), patch.object(suite.sync, 'accounting', side_effect=account), \
                patch.object(suite.runner, 'sync_once') as raw, patch.object(suite, 'upload_model') as mean:
            report = suite.main(['--runtime-root', str(root), '--status-only'])
            raw.assert_not_called(); mean.assert_not_called()
            assert not (root/'figures').exists()
        assert len(report['jobs']) == 21 and len(report['skipped_mean3']) == 6
        ready, _ = suite.full_models(plans, report['jobs'])
        assert len(ready) == 1 and {p['job_name'] for p in ready[0]} == {p['job_name'] for p in plans[:3]}

        # Real renderer and shared aggregation, but synthetic local data only.
        out = root/'synthetic_figures'
        with redirect_stdout(io.StringIO()), patch.object(study.charts, 'cloud_run_index', side_effect=AssertionError('No cloud')):
            record, inventory = suite.render_model(root, 'SYNTHETIC/project', out, plans[:3])
        assert len(inventory) == 3 and record['rows'][-1]['env_steps'] == 10000000
        assert {s['seed'] for s in record['sources']} == {1, 2, 3}
        assert list(out.glob('*three_seed.png')) and list(out.glob('*seeds.png'))
        assert old_render == (suite.runner.cohort, study.LABELS, study.SCENE_MODELS, study.configure)

        ended = {p['job_name'] for p in plans[:8]}
        def raw_sync(*args, **kwargs):
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=ended)
            assert suite.sync.GROUPS == tuple(s.GROUP for s in suite.SOURCES)
            return dict(busy=False, checked=8, uploaded=7, current=1, failed=0, missing_local=[])
        fake_result = dict(scene=record['scene'], model=record['model'], run_id='SYNTHETIC', run_url='SYNTHETIC')
        with redirect_stdout(io.StringIO()), patch.object(suite.sync, 'accounting', side_effect=account), \
                patch.object(suite.runner, 'sync_once', side_effect=raw_sync) as raw, \
                patch.object(suite, 'render_model', return_value=(record, inventory)), \
                patch.object(suite, 'scalar_current', side_effect=[False, True]) as current, \
                patch.object(suite, 'upload_model', return_value=fake_result) as mean, patch('wandb.Api'):
            report = suite.main(['--runtime-root', str(root)])
        assert len(report['mean3']) == 1 and not report['errors'] and len(report['skipped_mean3']) == 6
        raw.assert_called_once(); mean.assert_called_once(); assert current.call_count == 2
        assert report['mean3'][0]['status'] == 'uploaded and verified'

        # Current cloud scalars are skipped, never re-uploaded from a local marker.
        with redirect_stdout(io.StringIO()), patch.object(suite.sync, 'accounting', side_effect=account), \
                patch.object(suite.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(suite, 'render_model', return_value=(record, inventory)), \
                patch.object(suite, 'scalar_current', return_value=True), \
                patch.object(suite, 'upload_model') as mean, patch('wandb.Api'):
            report = suite.main(['--runtime-root', str(root)])
            mean.assert_not_called()
        assert report['mean3'][0]['status'] == 'cloud current; skipped'

        with redirect_stdout(io.StringIO()), patch.object(suite.sync, 'accounting', side_effect=account), \
                patch.object(suite.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(suite, 'render_model', side_effect=RuntimeError('SYNTHETIC missing history')), \
                patch.object(suite, 'upload_model') as mean, patch('wandb.Api'):
            try: suite.main(['--runtime-root', str(root)])
            except SystemExit as exc: assert 'PARTIAL' in str(exc)
            else: raise AssertionError('Incomplete history hidden')
            mean.assert_not_called()
        # Ambiguous retry IDs / missing receipts / unknown accounting fail closed.
        with suite.source_scope(), redirect_stdout(io.StringIO()), \
                patch.object(suite.sync, 'manifest_jobs', return_value={'80000001': plans[0]['job_name'], '80000002': plans[0]['job_name']}), \
                patch.object(suite.sync, 'accounting', return_value={}):
            bad = suite.audit_jobs(repo, root, plans)
            assert bad[0]['state'] == 'AMBIGUOUS_RECEIPTS' and bad[1]['state'] == 'MISSING_RECEIPT'
            assert not suite.full_models(plans, bad)[0]
        # Reject a local 2/3 inventory before scalar aggregation/cloud writes.
        with patch.object(suite.runner, 'local_figures', return_value=(plans[:3], inventory[:2], {})):
            try: suite.render_model(root, 'SYNTHETIC/project', out, plans[:3])
            except RuntimeError as exc: assert 'not all 10M' in str(exc)
            else: raise AssertionError('Incomplete seed coverage accepted')
        assert (suite.sync.GROUP, suite.sync.GROUPS) == before
        assert old_render == (suite.runner.cohort, study.LABELS, study.SCENE_MODELS, study.configure)
    print('PASS: exact 21-job scope, real synthetic three-seed rendering/aggregation, status-only read-only, live/failed/incomplete exclusions, original backfill, cloud-current skip and post-upload verification (services mocked)')


if __name__ == '__main__':
    main()
