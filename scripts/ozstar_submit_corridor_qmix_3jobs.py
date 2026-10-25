#!/usr/bin/env python3
"""Corridor ordinary QMIX only, historical two-layer head, 3 seeds/10M/24h.

Plan-only unless SUBMIT=YES. Reuse old Corridor QMIX run names and suite lock,
so exact-name active/completed baseline jobs are retained. No MMM2, VDN or ones.
"""
import os
from pathlib import Path
import subprocess

from ozstar_submit_corridor_vdn_qmix_10m_3seeds import GROUP, build_plans as old_plans
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_qmix_ones_five_maps_30jobs import require_capacity
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

SMOKE_SCRIPTS = ('scripts/smoke_test_submit_corridor_qmix.py',)


def build_plans(repo):
    plans = [p for p in old_plans(repo) if p['label'] == 'qmix']
    for plan in plans:
        plan['sbatch_args'] = ['--time=1-00:00:00' if a.startswith('--time=') else a
                              for a in plan['sbatch_args']]
        plan['exports']['EXTRA_ARGS'] += (
            ' qmix_condition_input=state mixer_state_refresh_enabled=False')
        plan['target_steps'] = 10000000
        plan['inventory_note'] = (
            'Ordinary state-conditioned QMIX; historical GRU + two-layer ELU action head; '
            'no gate, one-fill, auxiliary loss or state-refresh curriculum; '
            'old Corridor baseline identity retained, 10M/24h, scalar diagnostics')
    if len(plans) != 3 or {p['seed'] for p in plans} != {1, 2, 3}:
        raise RuntimeError('Expected only three Corridor QMIX baseline seeds')
    return plans


def validate_map():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    params = get_smac_map_registry().get('corridor', {})
    if (params.get('n_agents'), params.get('n_enemies')) != (6, 24):
        raise RuntimeError('Installed Corridor map definition mismatch')
    sc2 = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any((sc2/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing installed SC2_x64 under ' + str(sc2))
    if not any((sc2/'Maps').rglob('corridor.SC2Map')):
        raise RuntimeError('Missing installed corridor.SC2Map')


def main():
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_map)


if __name__ == '__main__':
    main()
