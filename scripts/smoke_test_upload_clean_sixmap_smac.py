#!/usr/bin/env python3
"""Fake scheduler/cloud integration; no network or real result directories."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import ozstar_upload_clean_sixmap_smac as suite


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = suite.cohort.build_plans(repo, 'smac')
    with tempfile.TemporaryDirectory(prefix='synthetic-sixmap-clean-') as temp:
        runtime = Path(temp).resolve()
        (runtime/'wandb').mkdir()
        (runtime/'ozstar_logs').mkdir()
        group = plans[:3]
        originals = {}
        for job, plan in enumerate(group, 101):
            rid = 'demo'+str(job)
            name = plan['job_name']
            directory = runtime/'wandb'/('offline-run-20261113_070000-'+rid)
            (directory/'files').mkdir(parents=True)
            (directory/('run-'+rid+'.wandb')).touch()
            source = runtime/'results/mixer_soft_onefill_gate'/name/'step_000010000_update_0000001.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'run_name': name}))
            (directory/'files'/source.name).symlink_to(source)
            (runtime/'ozstar_logs'/f'{name}_{job}.out').write_text(str(directory)+'\n')
            originals[str(job)] = directory
        unowned = runtime/'wandb/offline-run-20261113_070000-unowned'
        unowned.mkdir()
        rows = [dict(name=p['job_name'], state='COMPLETED' if i < 6 else 'FAILED', exit_code='0:0',
                     job_ids=[str(101+i)]) for i, p in enumerate(plans)]
        order = []

        def render(runtime, project, output, selected, jobs, base):
            assert selected == group and set(jobs) == set(originals)
            order.append('render')
            return {'scene': group[0]['scene'], 'model': group[0]['label']}, []

        def publish(record, output, api, wandb):
            order.append('publish')
            return {'run_id': 'synthetic', 'url': 'synthetic'}

        def verify(remote, data, directory, name):
            assert 'publish' in order
            if directory == originals['103']:
                raise RuntimeError('synthetic missing cloud data')
            return (10, 1)

        fake_wandb = SimpleNamespace(Api=lambda **kwargs: SimpleNamespace(run=lambda dest: dest))
        with patch.dict('sys.modules', {'wandb': fake_wandb}), \
                patch.object(suite.integrity, 'audit_latest', return_value=rows), \
                patch.object(suite.cleanup, 'live_jobs', return_value={'104'}), \
                patch.object(suite.cleanup, 'job_finished', return_value=True), \
                patch.object(suite.integrity, 'strict_render', side_effect=render), \
                patch.object(suite, 'publish', side_effect=publish), \
                patch.object(suite.cleanup, 'identity', side_effect=lambda data, rid, *args: (
                    suite.PROJECT+'/'+rid, group[int(rid[4:])-101]['job_name'])), \
                patch.object(suite.cleanup, 'verify_cloud', side_effect=verify), \
                patch.object(suite.cleanup.subprocess, 'run') as sync, \
                patch.object(suite.cleanup.time, 'time', return_value=10**12):
            suite.run(repo, runtime, False)
            assert not order and all(d.is_dir() for d in originals.values())
            try:
                suite.run(repo, runtime, True)
            except SystemExit as exc:
                assert 'PARTIAL' in str(exc)
            else:
                raise AssertionError('Expected one unverified original to be retained')
            sync.assert_called_once()
            assert Path(sync.call_args.args[0][1]).name == 'ozstar_wandb_sync_verified.py'
            assert sync.call_args.args[0][2:] == [str(originals['103']), '-e', 'hjh331-sjtu', '-p', 'gomarl']
            assert sync.call_args.kwargs == dict(check=True, timeout=900)
        assert order == ['render', 'publish']
        assert not originals['101'].exists() and not originals['102'].exists()
        assert originals['103'].is_dir() and unowned.is_dir()
        assert len(list((runtime/'results').rglob('*.json'))) == 3
        entries = (runtime/'wandb/completed_cleanup_audit.jsonl').read_text().splitlines()
        assert len(entries) == 2 and all(json.loads(x)['cloud_verified'] for x in entries)
        rows[2]['state'] = 'TIMEOUT'
        known = dict(id='demo103', name=group[2]['job_name'], lastHistoryStep=8123000)
        upload_api = SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=lambda dest: SimpleNamespace(**known)))
        snapshot = SimpleNamespace(run_id='demo103', entity='hjh331-sjtu', project='gomarl',
                                   history_count=10, last_step=8123000)
        with patch.dict('sys.modules', {'wandb': upload_api}), \
                patch.object(suite.integrity, 'audit_latest', return_value=rows), \
                patch.object(suite.cleanup, 'live_jobs', return_value={'104'}), \
                patch.object(suite.verified_sync, 'read_snapshot', return_value=snapshot), \
                patch.object(suite.cleanup.subprocess, 'run') as sync, \
                patch.object(suite, 'clean_original', side_effect=AssertionError('Upload-only cannot delete')):
            try:
                suite.run(repo, runtime, False, upload_only=True)
            except SystemExit as exc:
                assert 'PARTIAL' in str(exc)
            sync.assert_not_called()
        report = json.loads((runtime/'figures'/suite.GROUP/'upload_only_report.json').read_text())
        assert report['originals'][2]['last_step'] == 8123000
        assert report['originals'][2]['slurm_state'] == 'TIMEOUT'
        assert report['originals'][3]['status'] == 'unverified' and not report['groups']
        assert originals['103'].is_dir() and unowned.is_dir()
        name = group[2]['job_name']
        link = originals['103']/'files/step_000010000_update_0000001.json'
        link.resolve().write_text(json.dumps({'run_name': 'another job'}))
        try:
            suite.snapshot_for(runtime, name)(originals['103'])
        except RuntimeError as exc:
            assert 'belongs to this run' in str(exc)
        else:
            raise AssertionError('Foreign link target accepted')
        with tempfile.TemporaryDirectory(dir=runtime, prefix='render-') as temp_output:
            root = runtime/'results/sacred/sc2/synthetic'
            for seed, plan in enumerate(group, 1):
                run = root/str(seed)
                run.mkdir(parents=True)
                (run/'config.json').write_text(json.dumps(dict(wandb_run_name=plan['job_name'])))
                (run/'run.json').write_text(json.dumps(dict(start_time='2026-11-13T07:00:00')))
                (run/'info.json').write_text(json.dumps(dict(test_battle_won_mean_T=[0, 10000000],
                                                           test_battle_won_mean=[0, seed/10])))
                name = plan['job_name']
                log = runtime/'ozstar_logs'/f'{name}_{100+seed}.out'
                with log.open('a') as handle:
                    handle.write(f'Saving to FileStorageObserver in {root}.\nStarted run with ID "{seed}"\n')
            output = Path(temp_output)
            jobs = {str(100+seed): plan['job_name'] for seed, plan in enumerate(group, 1)}
            with patch.multiple(suite.up, GROUP=suite.GROUP, SCENES=suite.SCENES, LABELS=suite.LABELS), \
                    patch.object(suite.up.runner.study.charts, 'local_run_index', return_value={}):
                record, inventory = suite.integrity.strict_render(runtime, suite.PROJECT, output,
                                                                  group, jobs, suite.up.render_model)
            assert len(inventory) == 3 and all(r['coverage'] == 'complete' for r in inventory)
            assert record['rows'][-1]['env_steps'] == 10000000
            assert abs(record['rows'][-1]['test_battle_won_mean']-.2) < 1e-9
            assert (output/'seed_curves.csv').is_file()
    print('PASS: exact SMAC scope, live group skipped, mean3 first, closed cloud-verified copies only, source links preserved')


if __name__ == '__main__':
    main()
