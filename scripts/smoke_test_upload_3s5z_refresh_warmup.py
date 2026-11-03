#!/usr/bin/env python3
"""Synthetic retry receipts/Sacred histories and mock Slurm/cloud; no services."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as SN
from unittest.mock import patch

import ozstar_upload_3s5z_refresh_warmup_results as suite


def main():
    up = suite.uploader
    repo = Path(__file__).resolve().parents[1]
    plans = suite.build_plans(repo)
    assert len(plans) == 18 and len(suite.LABELS) == 6
    model = plans[0]['label']
    complete = [p for p in plans if p['label'] == model]
    ended_names = {p['job_name'] for p in complete}
    original = up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-3s5z-retry-upload-') as temp:
        root = Path(temp)
        logs = root/'ozstar_logs'
        logs.mkdir()
        (root/'wandb').mkdir()
        old = {p['job_name']: str(77000000+i) for i, p in enumerate(plans)}
        new = {p['job_name']: str(88000000+i) for i, p in enumerate(plans)}
        for source in (suite.refresh, suite.warmup):
            scoped = [p for p in plans if p['exports']['GROUP_NAME'] == source.GROUP]
            for tag, ids in (('old', old), ('new', new)):
                (logs/(source.GROUP+'_'+tag+'.json')).write_text(json.dumps(dict(
                    group=source.GROUP, plans=scoped,
                    submitted={p['job_name']: ids[p['job_name']] for p in scoped})))
        (logs/'unrelated.json').write_text(json.dumps(dict(group='unrelated', plans=[], submitted={'unrelated':'1'})))
        states = {}
        for p in plans:
            name = p['job_name']
            states[old[name]] = dict(state='FAILED', exit_code='1:0', elapsed='00:00:10',
                submitted='2026-11-02T05:04:00', started='2026-11-02T05:05:00', ended='2026-11-02T05:05:10')
            states[new[name]] = dict(state='COMPLETED' if name in ended_names else 'RUNNING',
                exit_code='0:0', elapsed='15:00:00', submitted='2026-11-02T06:00:00',
                started='2026-11-02T06:01:00', ended='2026-11-03T03:00:00')
        def account(jobs, checkout):
            assert checkout == repo and set(jobs) == set(old.values()) | set(new.values())
            return copy.deepcopy(states)
        with redirect_stdout(io.StringIO()), patch.object(suite, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once') as raw, patch.object(up, 'upload_model') as mean, patch('wandb.Api') as api:
            report = suite.main(['--runtime-root', str(root), '--status-only'])
            raw.assert_not_called(); mean.assert_not_called(); api.assert_not_called()
        assert report['state_counts'] == dict(COMPLETED=3, RUNNING=15)
        assert not (root/'figures').exists()
        assert all(r['job_ids'] == [new[r['name']]] and len(r['older_attempts']) == 1 for r in report['jobs'])
        # Verify parser checks Submit, exact name and WorkDir, not numeric IDs alone.
        p = plans[0]
        jid, name = new[p['job_name']], p['job_name']
        good = '|'.join((jid, 'COMPLETED', '0:0', '15:00:00', str(repo), '2026-11-02T06:00:00',
            '2026-11-02T06:01:00', '2026-11-03T03:00:00', name))
        with patch.object(up.sync, 'command', return_value=good):
            assert suite.detailed_accounting({jid: name}, repo)[jid]['state'] == 'COMPLETED'
        with patch.object(up.sync, 'command', return_value=good.replace(str(repo), '/wrong/checkout')):
            assert not suite.detailed_accounting({jid: name}, repo)

        for p in complete:
            name = p['job_name']
            for tag, win in (('current', .7), ('old', .99)):
                path = root/'results/sacred'/p['map_name']/(p['label']+tag)/str(p['seed'])
                path.mkdir(parents=True)
                (path/'config.json').write_text(json.dumps(dict(wandb_run_name=name)))
                # Make the wrong old source look newer: only job ownership, NOT
                # newest-looking filenames or best curves, is allowed to select.
                (path/'run.json').write_text(json.dumps(dict(start_time='2026-11-02T06:01:00Z'
                    if tag == 'current' else '2026-11-03T06:00:00Z')))
                (path/'info.json').write_text(json.dumps(dict(
                    test_battle_won_mean_T=[10000, 1000000, 5000000, 10000000, 10010000],
                    test_battle_won_mean=[.01, .1, .4, win, win])))
                if tag == 'current':
                    dirname = 'offline-run-20261102_060100-current{}'.format(p['seed'])
                    (logs/('{}_{}.err'.format(name, new[name]))).write_text(
                        'Saving to FileStorageObserver in {}.\nStarted run with ID "{}"\n'
                        'W&B run data saved in {}\n'.format(path.parent, p['seed'], root/'wandb'/dirname))
        def raw_sync(*args, **kw):
            assert kw == dict(ended_only=True, verify_existing=True, wanted_names=ended_names)
            filtered = up.sync.manifest_jobs(logs)
            assert filtered == {new[p['job_name']]: p['job_name'] for p in plans}
            return dict(busy=False, checked=3, current=1, uploaded=2, failed=0, missing_local=[])
        def upload(record, project, output, wandb):
            assert record['model'] == model and record['rows'][-1]['env_steps'] == 10000000
            assert record['rows'][-1]['test_battle_won_mean'] < .8, 'An old/best curve leaked into mean3'
            assert all('current' in s['run_id'] for s in record['sources'])
            return dict(scene='smac_3s5z', model=model, run_id='SYNTHETIC')
        with redirect_stdout(io.StringIO()) as output, patch.object(suite, 'detailed_accounting', side_effect=account), \
                patch.object(up.runner, 'sync_once', side_effect=raw_sync), \
                patch.object(up, 'upload_model', side_effect=upload) as mean, \
                patch.object(up, 'scalar_current', side_effect=[False, True]), patch('wandb.Api'), \
                patch.object(up.runner.study.charts, 'cloud_run_index', side_effect=AssertionError('No project scan')):
            try:
                report = suite.main(['--runtime-root', str(root)])
            except SystemExit:
                print(output.getvalue(), file=sys.stderr)
                raise
        mean.assert_called_once()
        assert not report['errors'] and len(report['mean3']) == 1 and len(report['skipped_mean3']) == 5
        assert (root/'figures'/suite.GROUP/'upload_report.json').is_file()
        assert list((root/'figures'/suite.GROUP).rglob('*three_seed.png'))
        assert original == (up.GROUP, up.SOURCES, up.LABELS, up.audit_jobs, up.render_model, up.sync.manifest_jobs)

        # After local cleanup, exact original IDs can restore curves from cloud.
        owned = {p['job_name']: dict(ids={'current'+str(p['seed'])}, sacred=set()) for p in complete}
        remote = {'current'+str(p['seed']): SN(id='current'+str(p['seed']), name=p['job_name'],
            created_at='2026-11-02T06:01:00Z', scan_history=lambda **kw: iter([
                dict(_step=10000, test_battle_won_mean=.1), dict(_step=10000000, test_battle_won_mean=.7)]))
            for p in complete}
        charts = up.runner.study.charts
        def cloud_render(runtime, project, output, selected):
            inventory = []
            for p in selected:
                curve, details, note = charts.select_latest(p, 'test_battle_won_mean', {}, {}, {},
                    runtime/'wandb', SN(collapse_duplicate_steps=lambda x, y: (x, y)),
                    {p['job_name']: [(9999999999, 'old-unrelated', 'retained_csv', {})]})
                assert curve[0][-1] == 10000000
                inventory.append(dict(run_name=p['job_name'], run_id=details['run_id']))
            return dict(cloud_fallback=True), inventory
        with patch.object(suite, 'run_ownership', return_value=owned), patch('wandb.Api') as api:
            api.return_value.run.side_effect = lambda path: remote[path.rsplit('/', 1)[-1]]
            record, inventory = suite.strict_render(root.resolve(), 'hjh331-sjtu/gomarl', root/'unused',
                complete, {new[p['job_name']]: p['job_name'] for p in complete}, cloud_render)
        assert record['cloud_fallback'] and len(inventory) == 3
        assert api.return_value.run.call_count == 3

        # A latest failed/running attempt must never be replaced by old success.
        name = complete[0]['job_name']
        states[old[name]].update(state='COMPLETED', exit_code='0:0')
        states[new[name]].update(state='FAILED', exit_code='1:0')
        with redirect_stdout(io.StringIO()), patch.object(suite, 'detailed_accounting', side_effect=account):
            report = suite.main(['--runtime-root', str(root), '--status-only'])
        row = next(r for r in report['jobs'] if r['name'] == name)
        assert row['state'] == 'FAILED' and row['job_ids'] == [new[name]]
        assert len(report['skipped_mean3']) == 6
        # Missing ANY receipt candidate's accounting makes latest selection unknown.
        states.pop(old[name])
        with redirect_stdout(io.StringIO()), patch.object(suite, 'detailed_accounting', side_effect=account):
            report = suite.main(['--runtime-root', str(root), '--status-only'])
        assert next(r for r in report['jobs'] if r['name'] == name)['state'].startswith('UNKNOWN')
    print('PASS: exact latest18/6models, retry selection by Submit, full job-owned curves only, old attempts excluded, '
          'real synthetic figures/mean3, original sync and cloud verification mocked, read-only status, no broad scan/cleanup/training')


if __name__ == '__main__':
    main()
