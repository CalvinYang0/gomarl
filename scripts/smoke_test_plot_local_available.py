#!/usr/bin/env python3
"""Synthetic QA for all local 5m6m models; cloud history is forbidden."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_local_available_3seeds as local_study
from retained_seed_curves import load_retained_directory
from smoke_test_retained_seed_curves import write_csv


def main():
    c = local_study.configure()
    plans = c.build_plans(c.ROOT)
    assert len(plans) == 63 and set(c.SCENES) == {'smac_5m6m'}
    live_models = {'qmix_state_group_qchange_isolated', 'qmix_state_group_qchange_plainmask',
                   'qmix_state_group_variance'}
    with tempfile.TemporaryDirectory(prefix='synthetic-local-5m6m-') as temp:
        root = Path(temp)
        source = root / 'figures/old-snapshot'
        source.mkdir(parents=True)
        meta, rows = [], []
        for p in plans:
            if p['label'] in live_models:
                continue
            run_id = 'synthetic-' + p['label'] + str(p['seed'])
            meta.append(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                             run_name=p['job_name'], run_id=run_id,
                             attempt_start='2026-10-10T00:00:00+00:00'))
            for step, value in [(10000, .1), (1000000, .5), (6000000, .8)]:
                rows.append(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                                 run_id=run_id, step=step, win_fraction=value))
        write_csv(source / 'seed_inventory.csv', list(meta[0]), meta)
        write_csv(source / 'seed_curves.csv', list(rows[0]), rows)
        # Duplicate snapshots must not become extra seeds or stitched histories.
        copy = root / 'figures/newer-snapshot'
        copy.mkdir()
        longer = rows + [dict(row, step=10000000, win_fraction=.9)
                         for row in rows if row['step'] == 6000000]
        write_csv(copy / 'seed_inventory.csv', list(meta[0]), meta)
        write_csv(copy / 'seed_curves.csv', list(rows[0]), longer)
        output = root / 'new-output'
        cached = load_retained_directory(root / 'figures', output, plans, c.SCENES, c.timestamp)
        assert len(cached) == 54 and all(len(value) == 1 for value in cached.values())
        assert all(value[0][3]['curve'][0][-1] == 10000000 for value in cached.values())
        for p in plans:
            if p['label'] != 'qmix_state_group_qchange_isolated' and not (
                    p['label'] == 'qmix_state_group_qchange_plainmask' and p['seed'] == 1):
                continue
            path = root / 'results/sacred' / p['map_name'] / p['label'] / str(p['seed'])
            path.mkdir(parents=True)
            (path / 'config.json').write_text(json.dumps({'wandb_run_name': p['job_name']}))
            (path / 'run.json').write_text(json.dumps({'start_time': '2026-10-16T00:00:00Z'}))
            (path / 'info.json').write_text(json.dumps({
                'test_battle_won_mean_T': [10000, 1000000, 3000000],
                'test_battle_won_mean': [.1, .4, .6]}))
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp, '--output-dir', str(output)]), \
                patch.object(c, 'cloud_run_index', side_effect=AssertionError('No cloud history')), \
                patch.object(local_study.study, 'upload') as upload:
            local_study.main()
        assert upload.call_count == 1, 'Uploading must be enabled by default'
        assert len(local_study.study._PANELS) == 20
        assert {row['section'] for row in local_study.study._PANELS} == {'5m6m'}
        assert all(row['model'] != 'qmix_state_group_variance' for row in local_study.study._PANELS)
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 63
        assert sum(row['source'] == 'retained_csv' for row in inventory) == 54
        assert sum(row['coverage'] == 'missing' for row in inventory) == 5
        with (output / 'smac_5m6m_aggregate.csv').open() as handle:
            aggregate = list(csv.DictReader(handle))
        assert len({row['model'] for row in aggregate}) == 20
        assert max(int(row['seed_count']) for row in aggregate) == 3
        assert (source / 'seed_curves.csv').is_file()
    print('PASS: twenty available models including one-fill from logs+deduplicated CSVs; 5m6m only; upload enabled (mocked); no cloud history')


if __name__ == '__main__':
    main()
