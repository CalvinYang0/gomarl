#!/usr/bin/env python3
"""Synthetic plotting QA only; never reports these fixtures as experiment data."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_8m9m_qmix_variants_3seeds as study


def main():
    study.configure()
    c = study.charts
    plans = study.build_plans(c.ROOT)
    assert len(plans) == 6
    assert all(p['map_name'] == '8m_vs_9m' for p in plans)
    assert {p['label'] for p in plans} == {'qmix', 'qmix_state_static10'}
    with tempfile.TemporaryDirectory(prefix='synthetic-8m9m-plot-') as temp:
        runtime = Path(temp)
        for p in plans:
            path = runtime / 'results/sacred/8m_vs_9m' / p['label'] / str(p['seed'])
            path.mkdir(parents=True)
            (path / 'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
            (path / 'run.json').write_text(json.dumps(dict(start_time='2026-10-14T00:00:00Z')))
            end = 6000000 if p['seed'] == 3 else 10000000
            (path / 'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 3000000, end],
                test_battle_won_mean=[.1, .4, .6, .7])))
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp, '--local-only', '--no-upload']):
            study.main()
        output = runtime / 'figures' / c.DEFAULT_OUTPUT_SUBDIR
        for file in ('smac_8m9m_three_seed.png', 'smac_8m9m_three_seed.pdf', 'smac_8m9m_individual_seeds.png'):
            assert (output / file).stat().st_size > 1000
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 6
        with (output / 'smac_8m9m_aggregate.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert {r['model'] for r in rows} == set(study.LABELS)
        assert all(r['seed_count'] == '3' and float(r['step']) <= 6000000 for r in rows)
    print('PASS: six actual 8m9m names; no 5m6m/VDN/history mixing; PNG/PDF/individual curves; shared three-seed interval; default window=10; no cloud/upload')


if __name__ == '__main__':
    main()
