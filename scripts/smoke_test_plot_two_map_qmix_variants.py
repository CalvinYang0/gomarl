#!/usr/bin/env python3
"""Synthetic QA only: both maps, isolated-group inclusion, honest missing seeds."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_8m9m_qmix_variants_3seeds as study


def main():
    study.configure()
    c = study.charts
    plans = study.build_plans(c.ROOT)
    assert len(plans) == 66
    for scene in study.SCENE_MODELS:
        vdn = [p for p in plans if p['scene'] == scene and p['label'] == 'vdn']
        assert {p['seed'] for p in vdn} == {1, 2, 3}
        assert all(p['job_name'].endswith('fresh20261014') for p in vdn)
    with tempfile.TemporaryDirectory(prefix='synthetic-two-map-qmix-') as temp:
        root = Path(temp)
        for p in plans:
            if p['label'] == 'qmix_state_group_calibration_isolated' and p['seed'] == 1:
                continue
            run = root / 'results/sacred' / p['map_name'] / p['label'] / str(p['seed'])
            run.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
            (run / 'run.json').write_text(json.dumps(dict(start_time='2026-10-14T00:00:00Z')))
            end = 6000000 if p['seed'] == 3 else 10000000
            (run / 'info.json').write_text(json.dumps(dict(
                test_battle_won_mean_T=[10000, 1000000, 3000000, end],
                test_battle_won_mean=[.1, .4, .6, .7])))
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp, '--local-only', '--no-upload']):
            study.main()
        output = root / 'figures' / c.DEFAULT_OUTPUT_SUBDIR
        assert len(study._PANELS) == 22
        assert [r['section'] for r in study._PANELS] == ['5m6m'] * 14 + ['8m9m'] * 8
        from PIL import Image
        for record in study._PANELS:
            with Image.open(record['path']) as image:
                assert image.width >= 1400 and image.width / image.height < 1.6
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 66
        assert sum(row['coverage'] == 'missing' for row in inventory) == 1
        for scene, models in study.SCENE_MODELS.items():
            for suffix in ('three_seed.png', 'three_seed.pdf', 'individual_seeds.png'):
                assert (output / (scene + '_' + suffix)).stat().st_size > 1000
            with (output / (scene + '_aggregate.csv')).open() as handle:
                rows = list(csv.DictReader(handle))
            assert {r['model'] for r in rows} == set(models)
            for row in rows:
                assert float(row['step']) <= 6000000
                assert row['seed_count'] == ('2' if row['model'] == 'qmix_state_group_calibration_isolated' else '3')
    print('PASS: two maps; 66 exact names; 22 independent panels; two-map MC-error ablations included; missing seeds explicit; no cloud/upload')


if __name__ == '__main__':
    main()
