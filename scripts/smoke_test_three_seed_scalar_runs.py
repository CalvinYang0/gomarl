#!/usr/bin/env python3
"""Synthetic scalar aggregates and exact-ended backfill; never real cloud writes."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import wandb_three_seed_scalar_runs as scalars
import ozstar_calibration_suite_sync as sync
import ozstar_update_two_map_three_seed as update
from smoke_test_retained_seed_curves import write_csv


class FakeRun:
    def __init__(self, kwargs):
        self.kwargs, self.step, self.rows, self.metrics = kwargs, 0, [], []
        self.url = 'https://synthetic.invalid/' + kwargs['id']

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, row, step):
        self.rows.append((step, row))

    def log_artifact(self, artifact):
        pass


class FakeWandb:
    def __init__(self):
        self.runs = []

    def init(self, **kwargs):
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run

    def Artifact(self, *args, **kwargs):
        return SimpleNamespace(add_file=lambda path: None)


def scalar_test(root):
    scalars.study.configure()
    all_plans = scalars.study.build_plans(scalars.study.charts.ROOT)
    plans = [p for p in all_plans if p['scene'] == 'smac_8m9m' and p['label'] in {'qmix', 'vdn'}]
    metadata, rows = [], []
    for p in plans:
        if p['label'] == 'vdn' and p['seed'] == 3:
            continue
        rid = 'synthetic' + p['label'] + str(p['seed'])
        metadata.append(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                             run_name=p['job_name'], run_id=rid,
                             attempt_start='2026-10-14T00:00:00Z'))
        # Seed 3 ends early. The aggregate must stop there, not switch to n=2.
        steps = [10000, 20000, 30000] if p['seed'] != 3 else [10000, 20000]
        rows.extend(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                         run_id=rid, step=step, win_fraction=p['seed'] * .1 + i * .05)
                    for i, step in enumerate(steps))
    source = root / 'source'
    source.mkdir()
    write_csv(source / 'seed_inventory.csv', list(metadata[0]), metadata)
    write_csv(source / 'seed_curves.csv', list(rows[0]), rows)
    records, skipped = scalars.aggregate_models(source / 'seed_curves.csv', plans, scalars.study.charts.SCENES)
    assert len(records) == 1 and records[0]['model'] == 'qmix'
    assert len(skipped) == 1 and skipped[0]['available_seeds'] == [1, 2]
    data = records[0]['rows']
    assert [row['env_steps'] for row in data] == [10000, 20000]
    np.testing.assert_allclose([row['test_battle_won_mean'] for row in data], [.2, .25])
    np.testing.assert_allclose([row['test_battle_won_mean_std'] for row in data], [.1, .1])
    assert all(row['seed_count'] == 3 for row in data)
    wandb = FakeWandb()
    first = scalars.upload_model(records[0], 'synthetic/project', root / 'output', wandb)
    second = scalars.upload_model(records[0], 'synthetic/project', root / 'output', wandb)
    assert first['run_id'] == second['run_id']
    assert wandb.runs[0].kwargs['job_type'] == 'three-seed-aggregate'
    assert wandb.runs[0].kwargs['config']['smoothing'] == 'none'
    assert len(wandb.runs[0].kwargs['config']['sources']) == 3
    assert wandb.runs[0].rows == [(row['env_steps'], row) for row in data]
    changed = dict(records[0], rows=[dict(row, test_battle_won_mean=.8) for row in data])
    third = scalars.upload_model(changed, 'synthetic/project', root / 'output', wandb)
    assert third['run_id'] != first['run_id'], 'Revised history must not append over an old snapshot'


def backfill_test(root):
    runtime = root / 'runtime'
    logs, wandb = runtime / 'ozstar_logs', runtime / 'wandb'
    logs.mkdir(parents=True)
    wandb.mkdir()
    runs = []
    for i in (1, 2):
        run = wandb / ('offline-run-20261018_120000-fixture' + str(i))
        (run / 'files').mkdir(parents=True)
        (run / ('run-fixture' + str(i) + '.wandb')).write_bytes(b'synthetic')
        runs.append((run, {str(i)}))
    jobs = {'1': 'current', '2': 'behind', '3': 'missing_source'}
    states = {job: dict(state='COMPLETED') for job in jobs}
    (logs / (sync.GROUP + '_sync_ledger.json')).write_text('{}')
    def cloud_check(run):
        if run == runs[1][0]:
            raise RuntimeError('Synthetic missing/behind cloud run')
    with redirect_stdout(io.StringIO()), patch.object(sync, 'selected_jobs', return_value=(jobs, states)), \
            patch.object(sync, 'discover', return_value=runs), \
            patch.object(sync, 'check_cloud_current', side_effect=cloud_check) as checks, \
            patch.object(sync.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as upload:
        result = sync.sync_once(root, runtime, 'synthetic_python', ended_only=True, verify_existing=True)
        assert checks.call_count == 2 and upload.call_count == 1
        assert upload.call_args.args[0][-1] == str(runs[1][0])
        assert result['checked'] == 2 and result['current'] == 1 and result['uploaded'] == 1
        assert result['missing_local'] == [dict(job='3', name='missing_source')]
    with redirect_stdout(io.StringIO()), patch.object(sync, 'selected_jobs', return_value=(jobs, states)), \
            patch.object(sync, 'discover', return_value=runs), \
            patch.object(sync, 'check_cloud_current', side_effect=cloud_check), \
            patch.object(sync.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
        result = sync.sync_once(root, runtime, 'synthetic_python', ended_only=True, verify_existing=True)
        assert result['failed'] == 1 and result['uploaded'] == 0


def main():
    with tempfile.TemporaryDirectory(prefix='synthetic-mean3-') as temp:
        root = Path(temp)
        scalar_test(root)
        backfill_test(root)
        # One-shot workflow: no polling/tmux, local plot reads, both maps then scalars.
        with redirect_stdout(io.StringIO()), patch.object(update.subprocess, 'run') as commands, \
                patch.object(scalars, 'main') as aggregate, patch.object(sync, 'sync_once') as original_sync:
            update.main(['--runtime-root', str(root), '--no-upload'])
            original_sync.assert_not_called()
            assert commands.call_count == 2
            assert all('--no-upload' in call.args[0] for call in commands.call_args_list)
            aggregate.assert_called_once()
            assert '--no-upload' in aggregate.call_args.args[0]
    print('PASS: unsmoothed mean/sample std/individual seeds; shared n=3 interval; missing seeds skipped; stable retry IDs; exact-ID cloud check and selective ended backfill')


if __name__ == '__main__':
    main()
