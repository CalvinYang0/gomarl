#!/usr/bin/env python3
"""Synthetic two-map cohort upload QA: all cloud/scheduler writes are mocked."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import ozstar_upload_weight_composition_results as upload
from smoke_test_three_seed_scalar_runs import FakeWandb


def main():
    runner = upload.runner
    old = runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE
    plans = upload.cohort.build_plans(Path(__file__).resolve().parents[1])
    wanted = {p['job_name'] for p in plans}
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-weight-results-upload-') as temp:
        root = Path(temp)
        for p in plans:
            path = root / 'results/sacred' / p['map_name'] / p['label'] / str(p['seed'])
            path.mkdir(parents=True)
            (path / 'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
            (path / 'run.json').write_text(json.dumps(dict(start_time='2026-10-20T00:00:00Z')))
            (path / 'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 10000000, 10010000],
                test_battle_won_mean=[.1, .6, .8+p['seed']*.01, .8+p['seed']*.01])))
        # Out-of-scope historical data cannot be included by name/map proximity.
        other = root / 'results/sacred/5m_vs_6m/qmix_onefill_p10/old'
        other.mkdir(parents=True)
        (other / 'config.json').write_text(json.dumps(dict(wandb_run_name='unrelated')))
        with redirect_stdout(io.StringIO()), patch.object(runner, 'sync_once') as sync, \
                patch.object(runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No cloud history scans')), \
                patch.object(runner.study, 'upload') as figures:
            report = upload.main(['--runtime-root', str(root), '--no-upload'])
            sync.assert_not_called(); figures.assert_not_called()
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE) == old
        inventory = report['inventory']
        assert len(inventory) == 15 and {r['run_name'] for r in inventory} == wanted
        assert sum(r['scene'] == 'smac_5m6m' for r in inventory) == 12
        assert sum(r['scene'] == 'smac_8m9m' for r in inventory) == 3
        assert not report['errors'] and len(report['mean3_plan']) == 5
        assert all(r['end_step'] == 10000000 and r['seeds'] == [1, 2, 3] for r in report['mean3_plan'])
        panels = runner.study._PANELS
        assert len(panels) == 5
        assert sum(p['section'] == '5m6m' for p in panels) == 4
        assert sum(p['section'] == '8m9m' for p in panels) == 1
        assert all(p['seeds'] == [1, 2, 3] for p in panels)
        output = root / 'figures' / upload.SUBDIR
        for scene in ('smac_5m6m', 'smac_8m9m'):
            for suffix in ('three_seed.png', 'three_seed.pdf', 'individual_seeds.png'):
                assert (output / (scene + '_' + suffix)).is_file()
        if os.environ.get('WEIGHT_UPLOAD_QA_OUTPUT'):
            import shutil
            shutil.copytree(output, os.environ['WEIGHT_UPLOAD_QA_OUTPUT'], dirs_exist_ok=True)
        outputs = {scene: [] for scene in ('smac_5m6m', 'smac_8m9m')}
        def fake_figures(*args):
            assert set(args[2]) == set(outputs)
            (output / 'model_panel_upload.json').write_text(json.dumps(dict(run_id='SYNTHETIC_FIGURE_ID')))
        fake = FakeWandb()
        import wandb
        with redirect_stdout(io.StringIO()), \
                patch.object(runner, 'local_figures', return_value=(plans, inventory, outputs)), \
                patch.object(runner, 'original_scope', return_value={p['job_name']: 'SYNTHETIC_JOB' for p in plans}), \
                patch.object(runner, 'sync_once', return_value=dict(busy=False, checked=15, current=12,
                    uploaded=3, failed=0, missing_local=[])) as sync, \
                patch.object(runner.study, 'upload', side_effect=fake_figures), \
                patch.object(wandb, 'init', side_effect=fake.init), \
                patch.object(wandb, 'Artifact', side_effect=fake.Artifact):
            report = upload.main(['--runtime-root', str(root)])
        assert sync.call_args.kwargs == dict(ended_only=True, verify_existing=True, wanted_names=wanted)
        assert not report['errors'] and len(report['mean3']) == len(fake.runs) == 5
        assert all(run.kwargs['config']['smoothing'] == 'none' for run in fake.runs)
        assert {run.kwargs['config']['scene'] for run in fake.runs} == set(outputs)
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE) == old
        missing = copy.deepcopy(inventory)
        missing[0]['coverage'] = 'partial'
        with redirect_stdout(io.StringIO()), patch.object(runner, 'local_figures', return_value=(plans, missing, outputs)):
            try:
                upload.main(['--runtime-root', str(root), '--no-upload'])
            except SystemExit as exc:
                assert 'PARTIAL/INCOMPLETE' in str(exc)
            else:
                raise AssertionError('Incomplete training falsely reported as complete')
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE) == old
        with patch.object(runner, 'original_scope', return_value={}), \
                patch.object(runner, 'sync_once', return_value=dict(busy=True)), \
                patch.object(runner, 'local_figures') as render:
            try:
                upload.main(['--runtime-root', str(root)])
            except SystemExit as exc:
                assert 'lock busy' in str(exc)
            else:
                raise AssertionError('Busy upload lock ignored')
            render.assert_not_called()
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE) == old
    print('PASS: 15 exact original IDs; 4+1 model panels/5 mean3; both-map scope; partial/busy errors; no cloud/scheduler writes')


if __name__ == '__main__':
    main()
