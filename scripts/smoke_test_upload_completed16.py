#!/usr/bin/env python3
"""Synthetic local figures and mocked original/mean3 uploads; no real services."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as SN
from unittest.mock import patch

import ozstar_upload_completed_16jobs_20261026 as suite
from smoke_test_three_seed_scalar_runs import FakeWandb
from wandb_seed_figure_sections import make_workspace


def layout_checks(records):
    package, ws, reports, wr = [ModuleType(n) for n in (
        'wandb_workspaces', 'wandb_workspaces.workspaces',
        'wandb_workspaces.reports', 'wandb_workspaces.reports.v2')]
    ws.Section = ws.SectionLayoutSettings = ws.Workspace = ws.RunsetSettings = lambda **kwargs: SN(**kwargs)
    wr.MediaBrowser = lambda **kwargs: SN(**kwargs)
    class Metric:
        def __init__(self, name): self.name = name
        def __eq__(self, value): return (self.name, value)
    package.expr = SN(Metric=Metric)
    package.workspaces, package.reports, reports.v2 = ws, reports, wr
    with patch.dict(sys.modules, {m.__name__: m for m in (package, ws, reports, wr)}):
        layout = make_workspace('SYNTHETIC', 'project', 'ID', records, [('summary', 'Media/mean')])
        assert [s.name for s in layout.sections] == ['Media', 'mmm2', 'corridor', '3s5z']
        assert [len(s.panels) for s in layout.sections] == [1, 2, 1, 2]
        assert all(s.layout_settings.columns == 1 for s in layout.sections)
        assert layout.runset_settings.filters == [('id', 'ID')]
        for section in layout.sections[1:]:
            assert all(p.media_keys[0].startswith(section.name+'/') for p in section.panels)


def main():
    repo, runner, sync = Path(__file__).resolve().parents[1], suite.runner, suite.sync
    before = (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
              sync.GROUP, sync.GROUPS, runner.study.LABELS, runner.study.SCENE_MODELS, runner.study.configure)
    plans = suite.build_plans(repo)
    assert len(plans) == 15 and len(suite.EXPECTED_JOBS) == 16
    wanted = {p['job_name'] for p in plans}
    assert suite.EXPECTED_JOBS['19185607'] == 'smac_mmm2_qmix_10m_s1_stateones20261025'
    assert suite.EXPECTED_JOBS['19185618'] == 'smac_3s5z_qmix_state_ones_10m_s3_stateones20261025'
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-completed16-upload-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        all_plans = plans + suite.slow.build_plans(repo)
        job_by_name = {name: job for job, name in suite.EXPECTED_JOBS.items()}
        slow_extra = [p for p in all_plans if p['job_name'] not in job_by_name]
        for job, plan in zip(range(19246005, 19246010), slow_extra):
            job_by_name[plan['job_name']] = str(job)
        assert len(job_by_name) == 21
        for group in (suite.paired.SMAC_GROUP, suite.corridor.GROUP, suite.slow.GROUP):
            selected = [p for p in all_plans if p['exports']['GROUP_NAME'] == group]
            (logs/(group+'_SYNTHETIC.json')).write_text(json.dumps(dict(
                group=group, plans=selected, submitted={p['job_name']: job_by_name[p['job_name']] for p in selected})))
        for plan in all_plans:
            path = root/'results/sacred'/plan['map_name']/plan['label']/str(plan['seed'])
            path.mkdir(parents=True, exist_ok=True)
            (path/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
            (path/'run.json').write_text(json.dumps(dict(start_time='2026-10-26T00:00:00Z')))
            (path/'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                test_battle_won_mean=[.01, .3, .7, .8+plan['seed']*.01, .82])))
        states = {job: dict(state='COMPLETED' if job in suite.EXPECTED_JOBS else 'PENDING',
                           exit_code='0:0', elapsed='10:00:00') for job in job_by_name.values()}
        with redirect_stdout(io.StringIO()), patch.object(runner, 'sync_once') as raw, \
                patch.object(sync, 'accounting', side_effect=AssertionError('Dry run must not query Slurm')), \
                patch.object(runner.study, 'upload') as figures, \
                patch.object(runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No cloud scans')):
            report = suite.main(['--runtime-root', str(root), '--no-upload'])
            raw.assert_not_called(); figures.assert_not_called()
        baseline = report['baselines']
        assert not report['errors'] and len(baseline['inventory']) == 15
        assert {r['run_name'] for r in baseline['inventory']} == wanted
        assert len(baseline['mean3_plan']) == 5
        assert all(r['seeds'] == [1, 2, 3] and r['end_step'] == 10000000 for r in baseline['mean3_plan'])
        records = list(runner.study._PANELS)
        assert len(records) == 5 and {r['section'] for r in records} == {'mmm2', 'corridor', '3s5z'}
        layout_checks(records)
        output = root/'figures'/suite.SUBDIR
        inventory = baseline['inventory']
        outputs = {scene: [output/(scene+'_three_seed.png')] for scene in suite.SCENES}
        fake, sync_calls, queried = FakeWandb(), [], []
        def fake_accounting(jobs, checkout):
            assert checkout == repo and set(jobs) <= set(suite.EXPECTED_JOBS)
            queried.append(set(jobs))
            return {j: states[j] for j in jobs}
        def fake_sync(*args, **kwargs):
            names = kwargs['wanted_names']
            assert names in (wanted, {suite.EXPECTED_JOBS[suite.EXTRA_JOB]})
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=names)
            assert sync.GROUPS == suite.SOURCE_GROUPS
            assert set(sync.manifest_jobs(logs)) == set(job_by_name.values())
            sync_calls.append(set(names))
            return dict(busy=False, checked=len(names), uploaded=len(names), current=0, failed=0, missing_local=[])
        def fake_figures(project, out, image_outputs, rows, window):
            assert set(image_outputs) == set(suite.SCENES) and len(rows) == 15 and window == 10
            (out/'model_panel_upload.json').write_text(json.dumps(dict(run_id='SYNTHETIC_FIGURES')))
        import wandb
        with redirect_stdout(io.StringIO()), \
                patch.object(runner, 'local_figures', return_value=(plans, inventory, outputs)), \
                patch.object(sync, 'accounting', side_effect=fake_accounting), \
                patch.object(runner, 'sync_once', side_effect=fake_sync), \
                patch.object(runner.study, 'upload', side_effect=fake_figures), \
                patch.object(wandb, 'init', side_effect=fake.init), patch.object(wandb, 'Artifact', side_effect=fake.Artifact):
            uploaded = suite.main(['--runtime-root', str(root)])
        assert not uploaded['errors'] and len(uploaded['baselines']['mean3']) == len(fake.runs) == 5
        assert [len(v) for v in sync_calls] == [15, 1]
        assert set.union(*queried) == set(suite.EXPECTED_JOBS)
        assert all(r.kwargs['config']['smoothing'] == 'none' for r in fake.runs)
        assert all(r.kwargs['config']['scene'] != 'smac_5m6m' for r in fake.runs)
        assert 'NOT CREATED' in uploaded['refresh_seed1']['mean3']
        with patch.object(sync, 'manifest_jobs', return_value={'WRONG': next(iter(wanted))}), \
                patch.object(sync, 'accounting') as account:
            try: suite.completed_scope(repo, root, wanted)
            except RuntimeError as exc: assert 'refusing to guess' in str(exc)
            else: raise AssertionError('Mismatched job receipts accepted')
            account.assert_not_called()
        with patch.object(sync, 'manifest_jobs', return_value=suite.EXPECTED_JOBS), \
                patch.object(sync, 'accounting', return_value={j: dict(state='RUNNING', exit_code='0:0')
                    for j in suite.EXPECTED_JOBS if j != suite.EXTRA_JOB}):
            try: suite.completed_scope(repo, root, wanted)
            except RuntimeError as exc: assert 'COMPLETED / 0:0' in str(exc)
            else: raise AssertionError('Live jobs accepted')
        with redirect_stdout(io.StringIO()), patch.object(runner, 'main', side_effect=RuntimeError('SYNTHETIC batch failure')), \
                patch.object(suite, 'extra_original', return_value={}) as extra:
            try: suite.main(['--runtime-root', str(root)])
            except SystemExit as exc: assert 'PARTIAL/INCOMPLETE' in str(exc)
            else: raise AssertionError('Partial upload failure hidden')
            extra.assert_called_once()
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                sync.GROUP, sync.GROUPS, runner.study.LABELS, runner.study.SCENE_MODELS, runner.study.configure) == before
        if os.environ.get('COMPLETED16_QA_OUTPUT'):
            import shutil
            shutil.copytree(output, os.environ['COMPLETED16_QA_OUTPUT'], dirs_exist_ok=True)
    print('PASS: exact completed IDs, 15 originals/5 mean3 plus one raw seed, pending exclusion, local-only mode, three-map layouts, independent errors; services mocked')


if __name__ == '__main__':
    main()
