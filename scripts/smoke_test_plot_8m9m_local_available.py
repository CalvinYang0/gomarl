#!/usr/bin/env python3
"""Synthetic local-only 8m9m upload QA; no real W&B writes."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_local_available_3seeds as local_study
from smoke_test_retained_seed_curves import write_csv


def main():
    c = local_study.configure('smac_8m9m')
    plans = c.build_plans(c.ROOT)
    expected = {'qmix', 'vdn', 'qmix_state_static10', 'qmix_state_ones',
                'qmix_state_static10_l1', 'qmix_state_static10_variance'}
    assert len(plans) == 18 and {p['label'] for p in plans} == expected
    assert set(c.SCENES) == {'smac_8m9m'}
    with tempfile.TemporaryDirectory(prefix='synthetic-local-8m9m-') as temp:
        root = Path(temp)
        source = root / 'figures/retained-two-map-snapshot'
        source.mkdir(parents=True)
        metadata, rows = [], []
        for p in plans:
            if p['label'] not in {'qmix', 'vdn', 'qmix_state_static10'}:
                continue
            rid = 'synthetic-' + p['label'] + str(p['seed'])
            metadata.append(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                                 run_name=p['job_name'], run_id=rid,
                                 attempt_start='2026-10-14T00:00:00Z'))
            rows.extend(dict(scene=p['scene'], model=p['label'], seed=p['seed'],
                             run_id=rid, step=x, win_fraction=y)
                        for x, y in [(10000, .1), (1000000, .6), (10000000, .85)])
        write_csv(source / 'seed_inventory.csv', list(metadata[0]), metadata)
        write_csv(source / 'seed_curves.csv', list(rows[0]), rows)
        for p in plans:
            if p['label'] in {'qmix', 'vdn', 'qmix_state_static10'}:
                continue
            if p['label'] == 'qmix_state_static10_variance' and p['seed'] == 3:
                continue
            path = root / 'results/sacred' / p['map_name'] / p['label'] / str(p['seed'])
            path.mkdir(parents=True)
            (path / 'config.json').write_text(json.dumps({'wandb_run_name': p['job_name']}))
            (path / 'run.json').write_text(json.dumps({'start_time': '2026-10-17T00:00:00Z'}))
            end = 1200000 if p['seed'] == 3 else 3000000
            (path / 'info.json').write_text(json.dumps({
                'test_battle_won_mean_T': [10000, 1000000, end],
                'test_battle_won_mean': [.1, .5, .65]}))
        output = root / 'figures/8m9m_local_available_qmix_variants_3seeds'
        source_before = (source / 'seed_curves.csv').read_bytes()
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp]), \
                patch.object(c, 'cloud_run_index', side_effect=AssertionError('Cloud history forbidden')), \
                patch.object(local_study.study, 'upload') as upload:
            local_study.main('smac_8m9m')
        assert upload.call_count == 1
        assert set(upload.call_args.args[2]) == {'smac_8m9m'}
        panels = local_study.study._PANELS
        assert len(panels) == 6 and {p['section'] for p in panels} == {'8m9m'}
        assert {p['model'] for p in panels} == expected
        assert next(p for p in panels if p['model'] == 'qmix_state_static10_variance')['seeds'] == [1, 2]
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 18
        assert sum(row['coverage'] == 'missing' for row in inventory) == 1
        assert sum(row['source'] == 'retained_csv' for row in inventory) == 9
        assert all(row['scene'] == 'smac_8m9m' for row in inventory)
        with (output / 'smac_8m9m_aggregate.csv').open() as handle:
            aggregate = list(csv.DictReader(handle))
        assert {row['model'] for row in aggregate} == expected
        assert max(float(row['step']) for row in aggregate if row['model'] == 'qmix_state_static10_l1') == 1200000
        assert (source / 'seed_curves.csv').read_bytes() == source_before
        import os
        if os.environ.get('PLOT_8M9M_QA_OUTPUT'):
            import shutil
            shutil.copytree(output, os.environ['PLOT_8M9M_QA_OUTPUT'], dirs_exist_ok=True)
    print('PASS: six 8m9m models/18 exact names; local CSV baselines + live variants; honest missing/partial seeds; upload mocked; no cloud history or 5m6m')


if __name__ == '__main__':
    main()
