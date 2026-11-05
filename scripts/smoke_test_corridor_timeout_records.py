#!/usr/bin/env python3
"""Synthetic timeout sources and fake W&B; never contacts scheduler/cloud."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import ozstar_upload_corridor_timeout_records as recovery


class CommError(Exception):
    pass


class FakeRun:
    def __init__(self, store, **kwargs):
        self.id, self.name, self.config = kwargs['id'], kwargs['name'], kwargs['config']
        self.url = 'SYNTHETIC_NOT_A_REAL_URL'
        self.summary, self.rows, self.step = {}, [], 0
        store[self.id] = self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def define_metric(self, *args, **kwargs): pass
    def log_artifact(self, *args): pass
    def log(self, row, step):
        assert step >= self.step
        self.rows.append(copy.deepcopy(row))
        self.step = step+1
    def scan_history(self, keys, **kwargs):
        return iter([{key: r[key] for key in keys} for r in self.rows if all(key in r for key in keys)])


def main():
    repo = Path(__file__).resolve().parents[1]
    up = recovery.suite.uploader
    plans = [p for p in recovery.suite.build_plans(repo) if p['scene'] == 'smac_corridor']
    assert [p['seed'] for p in plans] == [1, 2, 3]
    jobs = {str(99000000+i): p['job_name'] for i, p in enumerate(plans)}
    states = {job: dict(state='COMPLETED' if i == 1 else 'TIMEOUT', exit_code='0:0', elapsed='1-00:00:11',
        submitted='2026-11-04T04:59:52', started='2026-11-04T19:52:44', ended='2026-11-05T19:52:55')
        for i, job in enumerate(jobs)}
    def account(found, checkout):
        assert found == jobs and checkout == repo
        return copy.deepcopy(states)
    store, created = {}, []
    def api_run(path):
        rid = path.rsplit('/', 1)[-1]
        if rid not in store: raise CommError('Synthetic absent run')
        return store[rid]
    def init(**kwargs):
        assert 'TIMEOUT partial' in kwargs['name']
        assert kwargs['job_type'] == 'recovered-partial-history'
        assert kwargs['config']['incomplete_training'] and kwargs['config']['scheduler_state'] == 'TIMEOUT'
        created.append(kwargs['id'])
        return FakeRun(store, **kwargs)
    fake = SN(errors=SN(CommError=CommError), init=init, Api=lambda **kw: SN(run=api_run),
        Artifact=lambda *a, **kw: SN(add_file=lambda path: None))
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-timeout-recovery-') as temp:
        root = Path(temp).resolve()
        logs = root/'ozstar_logs'
        logs.mkdir()
        (root/'wandb').mkdir()
        (logs/(recovery.suite.cohort.GROUP+'_test.json')).write_text(json.dumps(dict(
            group=recovery.suite.cohort.GROUP, plans=plans, submitted={name: job for job, name in jobs.items()})))
        paths = {}
        for (job, name), plan in zip(jobs.items(), plans):
            source = root/'results/sacred/corridor'/recovery.suite.cohort.GROUP/str(plan['seed'])
            source.mkdir(parents=True)
            paths[plan['seed']] = source
            (source/'config.json').write_text(json.dumps(dict(wandb_run_name=name, seed=plan['seed'],
                env_args=dict(map_name='corridor'))))
            end = (10000000 if plan['seed'] == 2 else 9500000 if plan['seed'] == 1 else 8700000)
            info = dict(test_battle_won_mean_T=[10000, 1000000, end], test_battle_won_mean=[.0, .1, .8],
                td_error_abs_T=[10000, 1000000, end], td_error_abs=[1., .5, .1])
            (source/'info.json').write_text(json.dumps(info))
            (source/'metrics.json').write_text(json.dumps(dict(test_battle_won_mean=dict(
                steps=info['test_battle_won_mean_T'], values=info['test_battle_won_mean']))))
            native = root/'wandb'/('run-20261104_195244-native'+str(plan['seed']))
            if plan['seed'] == 1:
                native.mkdir()
                (native/('run-native'+str(plan['seed'])+'.wandb')).write_bytes(b'SYNTHETIC_BROKEN_BINARY')
            (logs/(name+'_'+job+'.err')).write_text('\x1b[34mSaving to FileStorageObserver in {}.\x1b[0m\n'
                'Started run with ID "{}"\nW&B run data saved in {}\n'.format(source.parent, plan['seed'], native))
        source_hashes = {p: p.read_bytes() for p in root.rglob('*.json')}
        with redirect_stdout(io.StringIO()), patch.object(recovery.suite.integrity, 'detailed_accounting', side_effect=account), \
                patch.object(recovery, 'native_upload', side_effect=ValueError('Synthetic torn native file')), \
                patch.dict(sys.modules, wandb=fake):
            report = recovery.main(['--runtime-root', str(root), '--status-only'])
            assert not created and not (root/'figures').exists()
            report = recovery.main(['--runtime-root', str(root)])
            assert not report['errors'] and {r['seed'] for r in report['results']} == {1, 3}
            assert {r['end_step'] for r in report['results']} == {9500000, 8700000}
            assert len(created) == 2
            for run in store.values():
                assert run.rows[-1]['env_steps'] < 10000000
                assert all('seed_2' not in key for row in run.rows for key in row)
            # Repeating the operation must not create duplicate recovered runs.
            recovery.main(['--runtime-root', str(root)])
            assert len(created) == 2
        assert all(p.read_bytes() == data for p, data in source_hashes.items())
        result = recovery.sources(repo, root, dict(name=plans[0]['job_name'], job_ids=[next(iter(jobs))]))
        assert len(result['native']) == 1 and result['sacred'] == paths[1]
        # Conflicting scalar pairs, wrong identity, and ambiguous retries are rejected.
        plan = plans[0]
        row = next(r for r in report['jobs'] if r['seed'] == 1)
        wrong = copy.deepcopy(row)
        wrong['all_job_ids'].append('99000009')
        try: recovery.sacred_rows(paths[1], plan, wrong)
        except ValueError: pass
        else: raise AssertionError('Same-name retry guessed from Sacred numeric path')
        metric = paths[1]/'metrics.json'
        old_metric = metric.read_text()
        metric.write_text(json.dumps(dict(test_battle_won_mean=dict(steps=[10000], values=[.99]))))
        try: recovery.sacred_rows(paths[1], plan, row)
        except ValueError: pass
        else: raise AssertionError('Conflicting scalar sources silently merged')
        metric.write_text(old_metric)
        # Current cloud snapshot must contain the actual test curve, not just summary progress.
        rid, run = next(iter(store.items()))
        saved = copy.deepcopy(run.rows)
        run.rows[0]['test_battle_won_mean'] = .99
        assert not recovery.recovery_current(fake.Api(), 'hjh331-sjtu/gomarl', rid,
            run.config['source_digest'], saved)
        run.rows = saved
        # Exact original cloud ID can still prove saved data after local files disappear.
        store['original1'] = SN(name=plan['job_name'], url='SYNTHETIC_NOT_A_REAL_URL',
            scan_history=lambda **kw: iter([dict(_step=8000000, test_battle_won_mean=.7)]))
        with patch.dict(sys.modules, wandb=fake):
            assert recovery.cloud_original(['original1'], plan, 'hjh331-sjtu/gomarl')['test_end_step'] == 8000000
        # Native successful upload and already-current paths retain original ID.
        snap = SN(run_id='original1', entity='hjh331-sjtu', project='gomarl', history_count=3)
        with patch.object(recovery, 'read_snapshot', return_value=snap), patch.object(recovery, 'verify_cloud'), \
                patch.object(recovery.subprocess, 'run') as sync:
            assert recovery.native_upload(Path('/SYNTHETIC'), 'hjh331-sjtu/gomarl')['run_id'] == 'original1'
            sync.assert_not_called()
        with patch.object(recovery, 'read_snapshot', return_value=snap), \
                patch.object(recovery, 'verify_cloud', side_effect=[RuntimeError('not current'), None]), \
                patch.object(recovery.subprocess, 'run') as sync:
            recovery.native_upload(Path('/SYNTHETIC'), 'hjh331-sjtu/gomarl')
            sync.assert_called_once()
    print('PASS: only latest corridor TIMEOUT seeds1/3, native ID preservation/fallback, exact Sacred ownership, '
          'real partial steps/all saved scalars, conflicting-source refusal, idempotent recovery/test-history verification; '
          'no real scheduler/cloud/training/deletion or fake full mean3')


if __name__ == '__main__':
    main()
