#!/usr/bin/env python3
"""Synthetic eighteen-job fixtures, mocked accounting/W&B; no real services."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import ozstar_upload_soft_onefill_refresh_results as suite


def main():
    uploader = suite.uploader
    repo = Path(__file__).resolve().parents[1]
    plans = sorted(suite.build_plans(repo), key=lambda p: (p['scene'], p['label'], p['seed']))
    assert len(plans) == 18
    assert {p['map_name'] for p in plans} == {'3s5z_vs_3s6z', 'MMM2', 'corridor'}
    before = uploader.GROUP, uploader.SOURCES, uploader.LABELS, uploader.build_plans
    before_sync = uploader.sync.GROUP, uploader.sync.GROUPS
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-soft-refresh-upload-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        ids = {p['job_name']: str(85000000+i) for i, p in enumerate(plans)}
        (logs/(suite.source.GROUP+'_SYNTHETIC.json')).write_text(json.dumps(dict(
            group=suite.source.GROUP, plans=plans, submitted=ids)))
        # An unrelated historical cohort must never enter accounting or uploads.
        old_group = 'qtot_sum_p10_four_maps_10m_12jobs_20261026'
        (logs/(old_group+'_SYNTHETIC.json')).write_text(json.dumps(dict(group=old_group,
            plans=[dict(job_name='UNRELATED')], submitted={'UNRELATED': '1'})))
        states = {j: dict(state='PENDING', exit_code='0:0', elapsed='00:00:00') for j in ids.values()}
        for p in plans[:3]:
            states[ids[p['job_name']]].update(state='COMPLETED', elapsed='10:00:00')
        for p in plans[3:5]:
            states[ids[p['job_name']]].update(state='RUNNING', elapsed='01:00:00')

        def account(jobs, checkout):
            assert checkout == repo and set(jobs) == set(ids.values())
            assert uploader.sync.GROUPS == (suite.source.GROUP,)
            return states

        with redirect_stdout(io.StringIO()) as output, patch.object(uploader.sync, 'accounting', side_effect=account), \
                patch.object(uploader.runner, 'sync_once') as raw, patch.object(uploader, 'upload_model') as mean:
            report = suite.main(['--runtime-root', str(root), '--status-only'])
            raw.assert_not_called(); mean.assert_not_called()
        assert report['state_counts'] == dict(COMPLETED=3, RUNNING=2, PENDING=13)
        assert len(report['skipped_mean3']) == 5 and '1 / 6 models' in output.getvalue()
        assert not (root/'figures').exists()

        for plan in plans[:3]:
            path = root/'results/sacred'/plan['map_name']/plan['label']/str(plan['seed'])
            path.mkdir(parents=True)
            (path/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
            (path/'run.json').write_text(json.dumps(dict(start_time='2026-10-28T12:00:00Z')))
            (path/'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                test_battle_won_mean=[.01, .2, .5, .7+plan['seed']*.01, .72])))
        ended = {p['job_name'] for p in plans[:3]}
        def raw_sync(*args, **kwargs):
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=ended)
            assert uploader.sync.GROUPS == (suite.source.GROUP,)
            return dict(busy=False, checked=3, uploaded=2, current=1, failed=0, missing_local=[])
        def upload(record, project, output, wandb):
            assert record['model'] in suite.LABELS
            assert record['rows'][-1]['env_steps'] == 10000000
            assert {s['run_name'] for s in record['sources']} == ended
            return dict(scene=record['scene'], model=record['model'], run_id='SYNTHETIC')

        with redirect_stdout(io.StringIO()), patch.object(uploader.sync, 'accounting', side_effect=account), \
                patch.object(uploader.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(uploader, 'upload_model', side_effect=upload) as mean, \
                patch.object(uploader, 'scalar_current', side_effect=[False, True]), patch('wandb.Api'), \
                patch.object(uploader.runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No broad cloud scan')):
            report = suite.main(['--runtime-root', str(root)])
        mean.assert_called_once()
        assert not report['errors'] and len(report['mean3']) == 1 and len(report['skipped_mean3']) == 5
        out = root/'figures'/suite.GROUP
        assert (out/'upload_report.json').exists() and list(out.rglob('*three_seed.png'))
        assert {r['run_name'] for r in report['mean3'][0]['inventory']} == ended

        # Failed job's partial originals are still backfilled, but no full mean3.
        states[ids[plans[0]['job_name']]].update(state='FAILED', exit_code='1:0')
        with redirect_stdout(io.StringIO()), patch.object(uploader.sync, 'accounting', side_effect=account), \
                patch.object(uploader.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(uploader, 'upload_model') as mean, patch('wandb.Api') as api:
            report = suite.main(['--runtime-root', str(root)])
            mean.assert_not_called(); api.assert_not_called()
        assert report['state_counts']['FAILED'] == 1 and len(report['skipped_mean3']) == 6
        assert report['mean3'] == []
        assert before == (uploader.GROUP, uploader.SOURCES, uploader.LABELS, uploader.build_plans)
        assert before_sync == (uploader.sync.GROUP, uploader.sync.GROUPS)
    print('PASS: only latest18/three maps/two models, exact receipt IDs, read-only status counts, real synthetic rendering/mean3, completed originals and failed partials, live/pending exclusion and restored scope (services mocked)')


if __name__ == '__main__':
    main()
