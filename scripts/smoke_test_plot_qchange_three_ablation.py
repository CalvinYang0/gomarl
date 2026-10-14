#!/usr/bin/env python3
"""Synthetic local QA for the single-factor ablation plot inventory."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_qchange_three_ablation_3seeds as study


def main():
    study.configure()
    charts = study.charts
    plans = study.build_plans(charts.ROOT)
    assert len(plans) == 12
    assert {p['label'] for p in plans} == set(study.MODELS)
    with tempfile.TemporaryDirectory(prefix='synthetic-qchange-ablation-') as temp:
        root = Path(temp)
        for plan in plans:
            if plan['label'] == 'qmix_state_group_qchange_plainmask' and plan['seed'] == 1:
                continue
            run = root / 'results/sacred' / plan['map_name'] / plan['label'] / str(plan['seed'])
            run.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps({'wandb_run_name': plan['job_name']}))
            (run / 'run.json').write_text(json.dumps({'start_time': '2026-10-15T00:00:00Z'}))
            end = 6000000 if plan['seed'] == 3 else 10000000
            (run / 'info.json').write_text(json.dumps({
                'test_battle_won_mean_T': [10000, 1000000, 3000000, end],
                'test_battle_won_mean': [.1, .4, .6, .7]}))
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp, '--local-only', '--no-upload']):
            study.main()
        output = root / 'figures' / charts.DEFAULT_OUTPUT_SUBDIR
        assert len(study._PANELS) == 4
        assert {p['section'] for p in study._PANELS} == {'5m6m'}
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 12
        assert sum(r['coverage'] == 'missing' for r in inventory) == 1
        with (output / 'smac_5m6m_aggregate.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert {r['model'] for r in rows} == set(study.MODELS)
        for row in rows:
            assert float(row['step']) <= 6000000
            assert row['seed_count'] == ('2' if row['model'] == 'qmix_state_group_qchange_plainmask' else '3')
        for suffix in ('three_seed.png', 'three_seed.pdf', 'individual_seeds.png'):
            assert (output / ('smac_5m6m_' + suffix)).stat().st_size > 1000
    print('PASS: control plus three independent ablations; twelve exact names; four 5m6m panels; honest missing seeds; no cloud/upload')


if __name__ == '__main__':
    main()
