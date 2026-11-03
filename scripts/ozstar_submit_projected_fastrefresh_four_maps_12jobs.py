#!/usr/bin/env python3
"""Exact-p0=1 projected one-fill + faststages8M; four SMAC maps x three seeds."""
import argparse
import copy
import os
from pathlib import Path
import shlex
import subprocess

import ozstar_submit_5m6m_soft_onefill_3seeds as original
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_mixer_state_refresh_9jobs import require_capacity
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = 'projected_init1_fastrefresh8m_four_maps_10m_12jobs_20261104'
LABEL = 'qmix_projected_init1_faststages8m'
MAPS = (
    ('smac_3s5z', '3s5z_vs_3s6z', 8, 9),
    ('smac_mmm2', 'MMM2', 10, 12),
    ('smac_corridor', 'corridor', 6, 24),
    ('smac_5m6m', '5m_vs_6m', 5, 6),
)
SMOKE_SCRIPTS = ('scripts/smoke_test_projected_fastrefresh.py',)


def build_plans(repo, scene=None):
    if scene is not None and scene not in {s for s, _, _, _ in MAPS}:
        raise ValueError('Unknown scene')
    result = []
    for key, name, _, _ in MAPS:
        if scene is not None and scene != key:
            continue
        for template in original.build_plans(repo):
            p = copy.deepcopy(template)
            x = dict(t.split('=', 1) for t in shlex.split(p['exports']['EXTRA_ARGS']))
            x.update(learner='projected_onefill_refresh_learner', mixer_input_initial_probability='1.0',
                qmix_condition_input='state', mixer_state_refresh_enabled='True',
                mixer_state_refresh_stage_steps='2000000', mixer_state_refresh_schedule='faster_stages',
                mixer_state_refresh_full_steps='8000000', mixer_state_refresh_eval_live='True',
                disable_video_media='True', test_battle_videos='False')
            p.update(scene=key, map_name=name, label=LABEL, memory='32G')
            p['exports'].update(GROUP_NAME=GROUP, MAP_NAME=name, EXTRA_ARGS=' '.join(k+'='+v for k, v in x.items()))
            rename(p, '{}_{}_10m_s{}_projfast20261104'.format(key, LABEL, p['seed']))
            p['sbatch_args'] = ['--mem=32G' if a.startswith('--mem=') else
                '--time=1-00:00:00' if a.startswith('--time=') else a for a in p['sbatch_args']]
            p['inventory_note'] = (
                'Exact initial1 direct field gate, post-optimizer projection [0,1], trainable throughout; '
                'raw mixer state held BEFORE interpolation; faststages K=1+floor((L-1)*(min(t,8M)/8M)^2); '
                'eval live incl terminal; replay uses collection snapshots; old GRU+two-layer ELU action head; '
                'single QMIX, unchanged soft TD, no freeze/threshold/KL/MC/auxloss; no videos; 28CPU/32G/24h')
            result.append(p)
    expected = 12 if scene is None else 3
    if len(result) != expected or len({p['job_name'] for p in result}) != expected:
        raise RuntimeError('Expected exact four-map combination scope')
    return result


def validate_maps(scene=None):
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    root = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any((root/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing installed SC2 binary')
    for key, name, allies, enemies in MAPS:
        if scene is not None and scene != key:
            continue
        p = registry.get(name, {})
        if (p.get('n_agents'), p.get('n_enemies')) != (allies, enemies):
            raise RuntimeError('Installed map identity differs: '+name)
        if not any((root/'Maps').rglob(name+'.SC2Map')):
            raise RuntimeError('Missing map asset: '+name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(s for s, _, _, _ in MAPS))
    args = parser.parse_args(argv)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.scene), SMOKE_SCRIPTS,
        map_validator=lambda: validate_maps(args.scene))


if __name__ == '__main__':
    main()
