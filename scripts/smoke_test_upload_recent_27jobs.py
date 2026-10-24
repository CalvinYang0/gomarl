#!/usr/bin/env python3
"""Synthetic cohort/map/layout QA; every scheduler and W&B operation mocked."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as SN, ModuleType
from unittest.mock import patch

import ozstar_upload_recent_completed_27jobs as combined
import ozstar_calibration_suite_sync as sync
from smoke_test_three_seed_scalar_runs import FakeWandb
from wandb_seed_figure_sections import make_workspace


def layout_checks(records):
    # Exercise actual layout routing against a tiny SDK-shaped local fake.
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
        result = make_workspace('SYNTHETIC', 'project', 'ID', records, [('summary', 'Media/mean')])
        assert [s.name for s in result.sections] == ['Media', '5m6m', 'mmm2', 'corridor']
        assert [len(s.panels) for s in result.sections] == [1, 4, 1, 1]
        assert all(s.layout_settings.columns == 1 for s in result.sections)
        for section in result.sections[1:]:
            assert all(p.media_keys[0].startswith(section.name+'/') for p in section.panels)
        assert result.runset_settings.filters == [('id', 'ID')]
        two_maps = [dict(section=s, title=s, key=s+'/model') for s in ('5m6m', '8m9m')]
        old_view = make_workspace('SYNTHETIC', 'project', 'ID', two_maps, [])
        assert [s.name for s in old_view.sections] == ['Media', '5m6m', '8m9m']
        try:
            make_workspace('SYNTHETIC', 'project', 'ID', [dict(section='unknown')], [])
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown map silently omitted from workspace')


def main():
    repo = Path(__file__).resolve().parents[1]
    runner = combined.rerun.runner
    original_state = (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                      sync.GROUP, sync.GROUPS, runner.study.configure, runner.study.LABELS)
    modules = (combined.rerun, combined.refresh)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-recent27-upload-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        all_jobs, all_names, states = {}, set(), {}
        counts = [18, 9]
        for module, count in zip(modules, counts):
            plans = module.cohort.build_plans(repo)
            assert len(plans) == count
            jobs = {str(90000+len(all_names)+i): p['job_name'] for i, p in enumerate(plans)}
            all_jobs[module.cohort.GROUP] = jobs
            all_names.update(jobs.values())
            states.update({job: dict(state='COMPLETED', exit_code='0:0', elapsed='10:00:00') for job in jobs})
            (logs/(module.cohort.GROUP+'_SYNTHETIC.json')).write_text(json.dumps(dict(
                group=module.cohort.GROUP, plans=plans, submitted={n: j for j, n in jobs.items()})))
            for plan in plans:
                path = root/'results/sacred'/plan['map_name']/plan['label']/(str(plan['seed'])+'_'+module.SUBDIR)
                path.mkdir(parents=True)
                (path/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
                (path/'run.json').write_text(json.dumps(dict(start_time='2026-10-23T00:00:00Z')))
                (path/'info.json').write_text(json.dumps(dict(
                    test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                    test_battle_won_mean=[.01, .3, .7, .8+plan['seed']*.01, .82])))
        # Old failed attempts exist but cannot leak into rerun means.
        old_plans = combined.rerun.source.source_plans(repo)
        old_group = old_plans[0]['exports']['GROUP_NAME']
        (logs/(old_group+'_SYNTHETIC.json')).write_text(json.dumps(dict(
            group=old_group, plans=old_plans, submitted={p['job_name']: str(i) for i, p in enumerate(old_plans)})))
        old_path = root/'results/sacred/5m_vs_6m/old/1'
        old_path.mkdir(parents=True)
        (old_path/'config.json').write_text(json.dumps(dict(wandb_run_name=old_plans[0]['job_name'])))
        (old_path/'info.json').write_text(json.dumps(dict(test_battle_won_mean_T=[10000, 10000000], test_battle_won_mean=[1., 1.])))
        with redirect_stdout(io.StringIO()), patch.object(runner, 'sync_once') as uploader, \
                patch.object(runner.study, 'upload') as figures, \
                patch.object(runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No broad cloud reads')):
            reports = combined.main(['--runtime-root', str(root), '--no-upload'])
            uploader.assert_not_called(); figures.assert_not_called()
        assert len(reports) == 2 and all(not r['errors'] for r in reports)
        assert [len(r['inventory']) for r in reports] == [18, 9]
        assert [len(r['mean3_plan']) for r in reports] == [6, 3]
        assert {r['run_name'] for report in reports for r in report['inventory']} == all_names
        assert all(r['seeds'] == [1, 2, 3] and r['end_step'] == 10000000 for report in reports for r in report['mean3_plan'])
        fake = FakeWandb()
        seen_panels = []
        def fake_figures(project, out, outputs, inventory, window):
            records = list(runner.study._PANELS)
            seen_panels.append(records)
            if len(inventory) == 18:
                assert set(outputs) == {'smac_5m6m', 'smac_mmm2', 'smac_corridor'}
                assert len(records) == 6
                layout_checks(records)
            else:
                assert len(records) == 3 and set(outputs) == {'smac_5m6m'}
            (out/'model_panel_upload.json').write_text(json.dumps(dict(run_id='SYNTHETIC_FIGURES')))
        def fake_sync(*args, **kwargs):
            jobs = all_jobs[sync.GROUP]
            assert sync.GROUPS == (sync.GROUP,)
            assert sync.manifest_jobs(logs) == jobs
            assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=set(jobs.values()))
            return dict(checked=len(jobs), uploaded=len(jobs), current=0, failed=0, busy=False, missing_local=[])
        import wandb
        with redirect_stdout(io.StringIO()), patch.object(runner, 'accounting', return_value=states) as accounting, \
                patch.object(runner, 'sync_once', side_effect=fake_sync), \
                patch.object(runner.study, 'upload', side_effect=fake_figures), \
                patch.object(wandb, 'init', side_effect=fake.init), patch.object(wandb, 'Artifact', side_effect=fake.Artifact):
            accounting.side_effect = lambda jobs, repo: {job: states[job] for job in jobs}
            reports = combined.main(['--runtime-root', str(root)])
        assert [len(r['mean3']) for r in reports] == [6, 3] and len(fake.runs) == 9
        assert all(r.kwargs['config']['smoothing'] == 'none' for r in fake.runs)
        assert len(seen_panels) == 2
        if os.environ.get('RECENT27_QA_OUTPUT'):
            import shutil
            shutil.copytree(root/'figures', os.environ['RECENT27_QA_OUTPUT'], dirs_exist_ok=True)
        # One batch failure must not suppress the other, nor report success.
        with redirect_stdout(io.StringIO()), patch.object(combined.rerun, 'main', side_effect=RuntimeError('SYNTHETIC missing source')), \
                patch.object(combined.refresh, 'main', return_value={}) as second:
            try: combined.main([])
            except SystemExit as exc: assert 'PARTIAL/INCOMPLETE' in str(exc)
            else: raise AssertionError('Batch failure hidden')
            second.assert_called_once()
        assert (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                sync.GROUP, sync.GROUPS, runner.study.configure, runner.study.LABELS) == original_state
    print('PASS: isolated 18+9 originals / 6+3 mean3; failed-attempt exclusion; MMM2/corridor layouts; independent errors; no real uploads')


if __name__ == '__main__':
    main()
