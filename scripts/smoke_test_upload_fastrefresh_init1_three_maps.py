#!/usr/bin/env python3
"""Synthetic six-group inventory, mixed states/retries, no Slurm/W&B writes."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import ozstar_upload_fastrefresh_init1_three_maps_results as suite


def main():
    repo = Path(__file__).resolve().parents[1]
    up = suite.uploader
    plans = suite.build_plans(repo)
    assert len(plans) == 18
    assert len({(p['scene'], p['label']) for p in plans}) == 6
    old = {p['job_name']: str(73000000+i) for i, p in enumerate(plans)}
    new = {p['job_name']: str(83000000+i) for i, p in enumerate(plans)}
    states = {}
    for p in plans:
        name = p['job_name']
        states[old[name]] = dict(state='COMPLETED', exit_code='0:0', elapsed='15:00:00',
            submitted='2026-11-06T04:00:00', started='2026-11-06T05:00:00', ended='2026-11-06T20:00:00')
        states[new[name]] = dict(state='COMPLETED', exit_code='0:0', elapsed='15:00:00',
            submitted='2026-11-07T04:00:00', started='2026-11-07T05:00:00', ended='2026-11-07T20:00:00')
        if p['scene'] == 'smac_corridor' and p['label'] == suite.cohort.VARIANTS['faststages'] and p['seed'] != 2:
            states[new[name]].update(state='TIMEOUT', elapsed='2-00:00:11')
        if p['scene'] == 'smac_mmm2' and p['label'] == suite.cohort.VARIANTS['init1'] and p['seed'] == 3:
            states[new[name]].update(state='RUNNING')
    def account(jobs, checkout):
        assert checkout == repo and jobs == {j: n for d in (old, new) for n, j in d.items()}
        return copy.deepcopy(states)
    def raw_sync(*args, **kwargs):
        ended = {name for name, job in new.items() if states[job]['state'] != 'RUNNING'}
        assert kwargs == dict(ended_only=True, verify_existing=True, wanted_names=ended)
        assert up.sync.manifest_jobs(args[1]/'ozstar_logs') == {job: name for name, job in new.items()}
        return dict(busy=False, checked=len(ended), current=0, uploaded=len(ended), failed=0, missing_local=[])
    original = up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-parents-upload-') as temp:
        root = Path(temp).resolve()
        logs = root/'ozstar_logs'
        logs.mkdir()
        (root/'wandb').mkdir()
        for tag, ids in (('old', old), ('new', new)):
            (logs/(suite.cohort.GROUP+'_'+tag+'.json')).write_text(json.dumps(dict(
                group=suite.cohort.GROUP, plans=plans, submitted=ids)))
        from ozstar_submit_projected_fastrefresh_four_maps_12jobs import GROUP as unrelated
        (logs/(unrelated+'_different.json')).write_text(json.dumps(dict(group=unrelated,
            plans=[dict(job_name='other-batch')], submitted={'other-batch':'1'})))
        for plan in plans:
            name, seed = plan['job_name'], plan['seed']
            for attempt, ids, end in (('old', old, 10000000), ('new', new, 10000000)):
                if attempt == 'new' and states[ids[name]]['state'] != 'COMPLETED':
                    end = 8700000
                path = root/'results/sacred'/plan['map_name']/(plan['label']+'-'+attempt)/str(seed)
                path.mkdir(parents=True)
                (path/'config.json').write_text(json.dumps(dict(wandb_run_name=name)))
                # Old file timestamps intentionally look newer than the true selected job.
                (path/'run.json').write_text(json.dumps(dict(start_time='2026-11-08T05:00:00Z'
                    if attempt == 'old' else '2026-11-07T05:00:00Z')))
                win = .99 if attempt == 'old' else .7
                (path/'info.json').write_text(json.dumps(dict(test_battle_won_mean_T=[10000, 1000000, end],
                    test_battle_won_mean=[0., .2, win])))
                (logs/('{}_{}.err'.format(name, ids[name]))).write_text(
                    'Saving to FileStorageObserver in {}.\nStarted run with ID "{}"\n'.format(path.parent, seed))
        # Fake W&B: real plot/CSV paths are exercised, with no installed SDK required.
        fake_wandb = SN(Api=lambda **kw: SN(), errors=SN(CommError=type('CommError', (Exception,), {})))
        def upload(record, project, output, wandb):
            assert record['rows'][-1]['env_steps'] == 10000000
            assert record['rows'][-1]['test_battle_won_mean'] < .8
            assert all('-new/' in s['run_id'] for s in record['sources'])
            return dict(scene=record['scene'], model=record['model'], run_id='SYNTHETIC')
        with redirect_stdout(io.StringIO()) as output, patch.dict(sys.modules, wandb=fake_wandb), \
                patch.object(suite.integrity, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once', side_effect=raw_sync) as raw, \
                patch.object(up, 'upload_model', side_effect=upload) as mean, \
                patch.object(up, 'scalar_current', side_effect=[False, True]*4), \
                patch.object(up.runner.study.charts, 'local_run_index', return_value={}), \
                patch.object(up.runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No project scan')):
            status = suite.main(['--runtime-root', str(root), '--status-only'])
            raw.assert_not_called(); mean.assert_not_called()
            assert not (root/'figures').exists()
            assert status['state_counts'] == dict(COMPLETED=15, TIMEOUT=2, RUNNING=1)
            assert all(r['job_ids'] == [new[r['name']]] for r in status['jobs'])
            try: report = suite.main(['--runtime-root', str(root)])
            except SystemExit:
                print(output.getvalue(), file=sys.stderr)
                raise
            raw.assert_called_once()
            assert mean.call_count == 4
        assert len(report['mean3']) == 4 and len(report['skipped_mean3']) == 2 and not report['errors']
        assert report['originals']['uploaded'] == 17
        assert len(list((root/'figures'/suite.GROUP).rglob('*three_seed.png'))) == 4
        with patch.object(suite.integrity, 'strict_render', side_effect=SystemExit('No test-win histories')), \
                patch.object(up, 'main', side_effect=lambda argv: up.render_model(
                    root, 'hjh331-sjtu/gomarl', root/'unused', plans[:3])):
            try: suite.main([])
            except RuntimeError as exc: assert 'No test-win histories' in str(exc)
            else: raise AssertionError('A missing group must be a recoverable per-group failure')
        assert original == (up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs)
    print('PASS: Nov6 exact18, six map/model groups, latest attempts only, timeout originals included, '
          'running job excluded, four complete mean3, real synthetic plots, no external uploads')


if __name__ == '__main__':
    main()
