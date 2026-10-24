#!/usr/bin/env python3
"""Synthetic-only refresh upload QA; no scheduler, cloud or real experiment data."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import ozstar_upload_mixer_state_refresh_results as upload
from smoke_test_three_seed_scalar_runs import FakeWandb


def main():
    runner, sync = upload.runner, upload.sync
    prior = runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope, sync.GROUPS
    repo = Path(__file__).resolve().parents[1]
    plans = upload.cohort.build_plans(repo)
    wanted = {p['job_name'] for p in plans}
    assert len(wanted) == 9 and {p['label'] for p in plans} == set(upload.LABELS)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-refresh-upload-') as temp:
        root = Path(temp)
        for plan in plans:
            path = root / 'results/sacred' / plan['map_name'] / plan['label'] / str(plan['seed'])
            path.mkdir(parents=True)
            (path/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
            (path/'run.json').write_text(json.dumps(dict(start_time='2026-10-23T00:00:00Z')))
            (path/'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                test_battle_won_mean=[.01, .3, .7, .8+plan['seed']*.01, .82])))
        other = root/'results/sacred/unrelated'
        other.mkdir()
        (other/'config.json').write_text(json.dumps(dict(wandb_run_name='unrelated_old_qmix')))
        with redirect_stdout(io.StringIO()), patch.object(runner, 'sync_once') as raw_sync, \
                patch.object(runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No cloud scans')), \
                patch.object(runner.study, 'upload') as figure_upload:
            report = upload.main(['--runtime-root', str(root), '--no-upload'])
            raw_sync.assert_not_called(); figure_upload.assert_not_called()
        assert not report['errors'] and len(report['inventory']) == 9
        assert {r['run_name'] for r in report['inventory']} == wanted
        assert len(report['mean3_plan']) == 3
        assert all(r['seeds'] == [1, 2, 3] and r['end_step'] == 10000000 for r in report['mean3_plan'])
        assert {p['section'] for p in runner.study._PANELS} == {'5m6m'}
        assert len(runner.study._PANELS) == 3
        output = root/'figures'/upload.SUBDIR
        if os.environ.get('REFRESH_UPLOAD_QA_OUTPUT'):
            import shutil
            shutil.copytree(output, os.environ['REFRESH_UPLOAD_QA_OUTPUT'], dirs_exist_ok=True)
        # Exercise actual manifest/accounting scope as well as mocked raw upload.
        jobs = {str(19086819+i): plan['job_name'] for i, plan in enumerate(plans)}
        logs = root/'ozstar_logs'
        logs.mkdir()
        (logs/(upload.cohort.GROUP+'_test.json')).write_text(json.dumps(dict(
            group=upload.cohort.GROUP, plans=plans,
            submitted={name: job for job, name in jobs.items()})))
        old = dict(group='unrelated', plans=[dict(job_name='old')], submitted={'old': '1'})
        (logs/'unrelated_test.json').write_text(json.dumps(old))
        states = {job: dict(state='COMPLETED', exit_code='0:0', elapsed='10:00:00') for job in jobs}
        def fake_figures(project, out, outputs, inventory, window):
            assert set(outputs) == {'smac_5m6m'} and len(inventory) == 9
            (out/'model_panel_upload.json').write_text(json.dumps(dict(run_id='SYNTHETIC_FIGURE_ID')))
        def fake_sync(*args, **kwargs):
            assert sync.GROUPS == (upload.cohort.GROUP,)
            assert sync.manifest_jobs(logs) == jobs
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=wanted)
            return dict(busy=False, checked=9, uploaded=9, current=0, failed=0, missing_local=[])
        fake = FakeWandb()
        import wandb
        with redirect_stdout(io.StringIO()), patch.object(runner, 'accounting', return_value=states), \
                patch.object(runner, 'sync_once', side_effect=fake_sync), \
                patch.object(runner.study, 'upload', side_effect=fake_figures), \
                patch.object(wandb, 'init', side_effect=fake.init), \
                patch.object(wandb, 'Artifact', side_effect=fake.Artifact):
            report = upload.main(['--runtime-root', str(root)])
        assert not report['errors'] and len(report['mean3']) == len(fake.runs) == 3
        assert all(r.kwargs['config']['smoothing'] == 'none' for r in fake.runs)
        incomplete = copy.deepcopy(report['inventory'])
        incomplete[0]['coverage'] = 'partial'
        with redirect_stdout(io.StringIO()), patch.object(runner, 'local_figures',
                return_value=(plans, incomplete, {'smac_5m6m': []})):
            try:
                upload.main(['--runtime-root', str(root), '--no-upload'])
            except SystemExit as exc:
                assert 'PARTIAL/INCOMPLETE' in str(exc)
            else:
                raise AssertionError('Incomplete history reported as complete')
        broken = copy.deepcopy(states)
        broken[next(iter(broken))].update(state='FAILED', exit_code='1:0')
        with patch.object(runner, 'accounting', return_value=broken), patch.object(runner, 'sync_once') as raw_sync:
            try:
                upload.main(['--runtime-root', str(root)])
            except RuntimeError as exc:
                assert 'COMPLETED / 0:0' in str(exc)
            else:
                raise AssertionError('Failed job accepted as complete')
            raw_sync.assert_not_called()
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope, sync.GROUPS) == prior
    print('PASS: 9 exact originals, 3 mean3, only 5m6m; isolated manifests; completed guard; no real uploads')


if __name__ == '__main__':
    main()
