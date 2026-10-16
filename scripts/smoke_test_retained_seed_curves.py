#!/usr/bin/env python3
"""Synthetic QA: retained control CSV plus live ablations, never cloud or old-attempt fallback."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

import plot_5m6m_qchange_three_ablation_3seeds as study
from retained_seed_curves import load_retained_curves


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    study.configure()
    c = study.charts
    plans = study.build_plans(c.ROOT)
    controls = [p for p in plans if p['label'] == 'qmix_state_group_qchange']
    ids = {1: '7agiboug', 2: 'v81lvl07', 3: 'qk2myz18'}
    with tempfile.TemporaryDirectory(prefix='synthetic-retained-control-') as temp:
        root = Path(temp)
        source = root / 'retained'
        source.mkdir()
        meta, rows = [], []
        for p in controls:
            seed = p['seed']
            meta.append(dict(scene=p['scene'], model=p['label'], seed=seed,
                             run_name=p['job_name'], run_id=ids[seed],
                             attempt_start='2026-10-10T00:00:00+00:00'))
            for step, value in [(10000, .1), (1000000, .5), (10040000, .8)]:
                rows.append(dict(scene=p['scene'], model=p['label'], seed=seed,
                                 run_id=ids[seed], step=step, win_fraction=value))
        write_csv(source / 'seed_inventory.csv', list(meta[0]), meta)
        csv_path = source / 'seed_curves.csv'
        write_csv(csv_path, list(rows[0]), rows)
        retained = load_retained_curves(csv_path, plans, c.SCENES, c.timestamp)
        assert len(retained) == 3
        import plot_wandb_seed_aggregate as plotter
        p = controls[0]
        curve, details, note = c.select_latest(p, 'test_battle_won_mean', {}, {}, {},
                                               root, plotter, retained)
        assert curve[0][-1] == 10040000
        assert details['run_id'] == ids[p['seed']] and details['source'] == 'retained_csv'
        assert str(csv_path) in note

        # A newer attempt with no history must stay missing, not borrow the old control.
        empty = root / 'empty'
        empty.mkdir()
        (empty / 'info.json').write_text('{}')
        sacred = {p['job_name']: [(c.timestamp('2026-10-16T00:00:00Z'), empty)]}
        curve, details, _ = c.select_latest(p, 'test_battle_won_mean', {}, {}, sacred,
                                            root, plotter, retained)
        assert curve is None and details['source'] == 'sacred'

        # A newer nonempty attempt replaces, rather than joins, the saved attempt.
        (empty / 'info.json').write_text(json.dumps({
            'test_battle_won_mean_T': [10000, 20000],
            'test_battle_won_mean': [.05, .1]}))
        curve, details, _ = c.select_latest(p, 'test_battle_won_mean', {}, {}, sacred,
                                            root, plotter, retained)
        assert curve[0].tolist() == [10000, 20000] and details['source'] == 'sacred'

        # The same display label cannot import a different cohort's records.
        wrong = [dict(row, run_name=row['run_name'] + '_other_cohort') for row in meta]
        write_csv(source / 'seed_inventory.csv', list(wrong[0]), wrong)
        assert not load_retained_curves(csv_path, plans, c.SCENES, c.timestamp)
        write_csv(source / 'seed_inventory.csv', list(meta[0]), meta)
        invalid = [dict(row) for row in rows]
        invalid[0]['win_fraction'] = 80
        write_csv(csv_path, list(rows[0]), invalid)
        try:
            load_retained_curves(csv_path, plans, c.SCENES, c.timestamp)
        except ValueError:
            pass
        else:
            raise AssertionError('Percentage-valued fractions accepted')
        write_csv(csv_path, list(rows[0]), rows)

        # Three isolated seeds and one plain-mask seed exist locally; variance does not.
        for p in plans:
            if (p['label'] != 'qmix_state_group_qchange_isolated' and
                    not (p['label'] == 'qmix_state_group_qchange_plainmask' and p['seed'] == 1)):
                continue
            run = root / 'results/sacred' / p['map_name'] / p['label'] / str(p['seed'])
            run.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps({'wandb_run_name': p['job_name']}))
            (run / 'run.json').write_text(json.dumps({'start_time': '2026-10-16T00:00:00Z'}))
            (run / 'info.json').write_text(json.dumps({
                'test_battle_won_mean_T': [10000, 1000000, 3000000],
                'test_battle_won_mean': [.1, .4, .6]}))
        output = root / 'new-output'
        before = csv_path.read_bytes()
        with patch.object(sys, 'argv', ['plot', '--runtime-root', temp,
                '--output-dir', str(output), '--retained-curves', str(csv_path),
                '--available-only', '--local-only', '--no-upload']), \
                patch.object(c, 'cloud_run_index', side_effect=AssertionError('Cloud read forbidden')), \
                patch.object(study, 'upload', side_effect=AssertionError('Upload forbidden')):
            study.main()
        assert csv_path.read_bytes() == before
        assert len(study._PANELS) == 3
        assert {p['model'] for p in study._PANELS} == set(study.MODELS) - {'qmix_state_group_variance'}
        with (output / 'seed_inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        saved = [row for row in inventory if row['source'] == 'retained_csv']
        assert len(saved) == 3 and {row['run_id'] for row in saved} == set(ids.values())
        with (output / 'smac_5m6m_aggregate.csv').open() as handle:
            aggregate = list(csv.DictReader(handle))
        assert {row['model'] for row in aggregate} == set(study.MODELS) - {'qmix_state_group_variance'}
        assert {row['seed_count'] for row in aggregate if row['model'] == 'qmix_state_group_qchange'} == {'3'}
        assert {row['seed_count'] for row in aggregate if row['model'] == 'qmix_state_group_qchange_plainmask'} == {'1'}
        assert all(np.isfinite(float(row['mean_fraction_smoothed'])) for row in aggregate)
        # Copy QA outputs outside the temporary fixture only when explicitly requested.
        import os
        if os.environ.get('RETAINED_CURVE_QA_OUTPUT'):
            import shutil
            shutil.copytree(output, os.environ['RETAINED_CURVE_QA_OUTPUT'], dirs_exist_ok=True)
    print('PASS: retained three-seed control + live ablations; exact cohort/IDs; latest attempts; honest 1/3; no source overwrite/cloud/upload')


if __name__ == '__main__':
    main()
