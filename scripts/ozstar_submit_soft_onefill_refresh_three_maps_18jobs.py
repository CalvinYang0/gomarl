#!/usr/bin/env python3
"""Three maps x (historical soft one-fill / + refresh 2M) x three seeds.

10M/24h. Same old actor head and soft one-fill TD, no new auxiliary loss.
Plan-only unless SUBMIT=YES; no cancellation, cleanup or checkpoint reuse.
"""
import copy
import os
from pathlib import Path
import shlex
import subprocess

import ozstar_submit_5m6m_soft_onefill_3seeds as reference
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_mixer_state_refresh_9jobs import require_capacity
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = 'soft_onefill_refresh_three_maps_10m_18jobs_20261028'
MAPS = (
    ('smac_3s5z', '3s5z_vs_3s6z', 8, 9),
    ('smac_mmm2', 'MMM2', 10, 12),
    ('smac_corridor', 'corridor', 6, 24),
)
LABELS = ('qmix_soft_onefill_p10', 'qmix_soft_onefill_p10_refresh_2000k')
SMOKE_SCRIPTS = ('scripts/smoke_test_soft_onefill_refresh.py',)


def build_plans(repo):
    plans = []
    for scene, map_name, _, _ in MAPS:
        for original in reference.build_plans(repo):
            for refresh in (False, True):
                plan = copy.deepcopy(original)
                label = LABELS[int(refresh)]
                plan.update(scene=scene, map_name=map_name, label=label, memory='32G')
                extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
                extra.update(qmix_condition_input='state', mixer_state_refresh_enabled=str(refresh),
                    mixer_state_refresh_stage_steps='2000000',
                    learner='soft_onefill_refresh_learner' if refresh else 'soft_onefill_gate_learner')
                plan['exports'].update(GROUP_NAME=GROUP, MAP_NAME=map_name,
                    EXTRA_ARGS=' '.join(k+'='+v for k, v in extra.items()))
                rename(plan, '{}_{}_10m_s{}_softrefresh20261028'.format(scene, label, plan['seed']))
                plan['sbatch_args'] = ['--mem=32G' if a.startswith('--mem=') else
                    '--time=1-00:00:00' if a.startswith('--time=') else a for a in plan['sbatch_args']]
                plan['inventory_note'] = (
                    'Historical GRU + two-layer ELU fixed head; single complete QMIX; '
                    'static coordinate sigmoid p0=0.1, fill=1, exact gradients, joint soft-onefill TD only; '
                    + ('raw mixer state held BEFORE gate interpolation; K=1+floor(collection_t_env/2M), '
                       'max-episode-length partitions, replay uses collected snapshots; ' if refresh else
                       'mixer state refreshed every timestep; ')
                    + 'actor obs live; no hard sampling/ST/threshold/KL/MC/second mixer; 28 CPU/32G/24h')
                plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 18:
        raise RuntimeError('Expected exactly three maps x two variants x three seeds')
    return plans


def validate_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    root = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any((root/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing installed SC2_x64 under ' + str(root))
    for _, name, allies, enemies in MAPS:
        params = registry.get(name, {})
        if (params.get('n_agents'), params.get('n_enemies')) != (allies, enemies):
            raise RuntimeError('Installed map unit count mismatch: ' + name)
        if not any((root/'Maps').rglob(name+'.SC2Map')):
            raise RuntimeError('Missing SC2 map file: ' + name)


def main():
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_maps)


if __name__ == '__main__':
    main()
