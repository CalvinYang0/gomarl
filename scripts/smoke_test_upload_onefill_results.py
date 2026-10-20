#!/usr/bin/env python3
"""Synthetic scoped uploader QA; no actual Slurm or cloud access."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import ozstar_upload_5m6m_onefill_results as upload
from smoke_test_three_seed_scalar_runs import FakeWandb


def fixture(root):
    plans = upload.cohort.build_plans(Path(__file__).resolve().parents[1])
    for p in plans:
        path = root / 'results/sacred/5m_vs_6m' / p['label'] / str(p['seed'])
        path.mkdir(parents=True)
        (path / 'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
        (path / 'run.json').write_text(json.dumps(dict(start_time='2026-10-19T00:00:00Z')))
        (path / 'info.json').write_text(json.dumps(dict(test_battle_won_mean_T=[10000, 1000000, 10000000, 10010000],
            test_battle_won_mean=[.1, .5, .8+p['seed']*.01, .8+p['seed']*.01])))
    # An unrelated historic run must never be requested or drawn.
    other = root / 'results/sacred/8m_vs_9m/qmix/old'
    other.mkdir(parents=True)
    (other / 'config.json').write_text(json.dumps(dict(wandb_run_name='unrelated')))
    return plans


def main():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-onefill-upload-QA-') as temp:
        root = Path(temp)
        plans = fixture(root)
        with redirect_stdout(io.StringIO()), patch.object(upload, 'sync_once') as sync, \
                patch.object(upload.study.charts, 'cloud_run_index', side_effect=AssertionError('No cloud histories')), \
                patch.object(upload.study, 'upload') as figures:
            report = upload.main(['--runtime-root', str(root), '--no-upload'])
            sync.assert_not_called()
            figures.assert_not_called()
            assert len(report['inventory']) == 15
            assert len(report['mean3_plan']) == 5
            assert not report['errors'] and all(r['end_step'] == 10000000 for r in report['mean3_plan'])
            assert {row['scene'] for row in report['inventory']} == {'smac_5m6m'}
            assert len(upload.study._PANELS) == 5
            assert all(r['seeds'] == [1, 2, 3] and r['section'] == '5m6m' for r in upload.study._PANELS)
            assert all(r['end_steps'] == [10000000]*3 for r in upload.study._PANELS)
        output = root / 'figures' / upload.SUBDIR
        # Reuse the real local output to test upload orchestration without
        # rendering another full set of figures.
        inventory, outputs = report['inventory'], {'smac_5m6m': []}
        def figures(*args):
            (output / 'model_panel_upload.json').write_text(json.dumps(dict(run_id='SYNTHETIC_FIGURE_ID')))
        fake = FakeWandb()
        import wandb
        with redirect_stdout(io.StringIO()), \
                patch.object(upload, 'local_figures', return_value=(plans, inventory, outputs)), \
                patch.object(upload, 'original_scope', return_value={p['job_name']: 'SYNTHETIC_JOB' for p in plans}), \
                patch.object(upload, 'sync_once', return_value=dict(busy=False, checked=15, current=12,
                    uploaded=3, failed=0, missing_local=[])) as sync, \
                patch.object(upload.study, 'upload', side_effect=figures), \
                patch.object(wandb, 'init', side_effect=fake.init), \
                patch.object(wandb, 'Artifact', side_effect=fake.Artifact):
            report = upload.main(['--runtime-root', str(root)])
        assert sync.call_args.kwargs == dict(ended_only=True, verify_existing=True,
            wanted_names={p['job_name'] for p in plans})
        assert len(report['mean3']) == len(fake.runs) == 5
        assert not report['errors']
        assert all(run.kwargs['config']['smoothing'] == 'none' for run in fake.runs)
        with patch.object(upload, 'original_scope', return_value={}), \
                patch.object(upload, 'sync_once', return_value=dict(busy=True)) as sync, \
                patch.object(upload, 'local_figures') as render:
            try:
                upload.main(['--runtime-root', str(root)])
            except SystemExit as exc:
                assert 'lock busy' in str(exc)
            else:
                raise AssertionError('Busy lock falsely treated as successful upload')
            render.assert_not_called()
        jobs = {str(i): p['job_name'] for i, p in enumerate(plans)}
        expected = set(jobs.values())
        with patch.object(upload, 'manifest_jobs', return_value=jobs), \
                patch.object(upload, 'accounting', return_value={job: dict(state='COMPLETED') for job in jobs}):
            assert len(upload.original_scope(root, root, expected)) == 15
        with patch.object(upload, 'manifest_jobs', return_value={}), \
                patch.object(upload, 'accounting') as account:
            try:
                upload.original_scope(root, root, expected)
            except RuntimeError as exc:
                assert 'refusing to guess' in str(exc)
            else:
                raise AssertionError('Missing manifest was not rejected')
            account.assert_not_called()
    print('PASS: fifteen exact original IDs scoped; five 5m6m panels/mean3; no historic/8m9m histories; local-only dry run; shared-lock busy stops; cloud/scheduler mocked')


if __name__ == '__main__':
    main()
