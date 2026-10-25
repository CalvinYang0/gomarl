#!/usr/bin/env python3
"""Historical qmix_qtot_sum_p10 on four SMAC maps x seeds 1/2/3, 10M/24h.

Plan-only unless SUBMIT=YES. Same 5m6m recipe, no model/loss/head changes,
no ordinary baseline/ones jobs, no cancellations, cleanup or resume.
"""
import copy
import os
from pathlib import Path
import subprocess

from ozstar_submit_qtot_and_onefill_15jobs import build_plans as reference_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_mixer_state_refresh_9jobs import require_capacity
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = 'qtot_sum_p10_four_maps_10m_12jobs_20261026'
LABEL = 'qmix_qtot_sum_p10'
MAPS = (
    ('smac_8m9m', '8m_vs_9m', 8, 9),
    ('smac_corridor', 'corridor', 6, 24),
    ('smac_mmm2', 'MMM2', 10, 12),
    ('smac_3s5z', '3s5z_vs_3s6z', 8, 9),
)
SMOKE_SCRIPTS = ('scripts/smoke_test_submit_qtot_sum_four_maps.py',)


def build_plans(repo):
    originals = [p for p in reference_plans(repo) if p['label'] == LABEL]
    if len(originals) != 3 or {p['seed'] for p in originals} != {1, 2, 3}:
        raise RuntimeError('Missing the historical three-seed 5m6m Qtot-sum recipe')
    plans = []
    for scene, map_name, _, _ in MAPS:
        for original in originals:
            plan = copy.deepcopy(original)
            plan.update(scene=scene, map_name=map_name, memory='32G')
            plan['exports'].update(MAP_NAME=map_name, GROUP_NAME=GROUP)
            rename(plan, '{}_{}_10m_s{}_qtotsum20261026'.format(scene, LABEL, plan['seed']))
            plan['sbatch_args'] = ['--mem=32G' if a.startswith('--mem=') else a
                                  for a in plan['sbatch_args']]
            plan['inventory_note'] = (
                original['inventory_note'] + '; map transfer only; fresh from step zero; '
                'no 5m6m weights/checkpoints or partial histories reused; 28 CPU/32G/24h')
            plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 12:
        raise RuntimeError('Expected only twelve Qtot-sum transfer jobs')
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
