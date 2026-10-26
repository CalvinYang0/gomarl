#!/usr/bin/env python3
"""Corridor all-ones mixer condition: matched historical QMIX, 3 seeds/10M/24h.

Plan-only unless SUBMIT=YES. Only the condition input and run identities differ
from the completed Corridor baseline. No gate, one-fill, extra loss or new head.
"""
import os
import shlex
import subprocess

import ozstar_submit_corridor_qmix_3jobs as baseline
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = 'corridor_mixer_ones_10m_3jobs_20261027'
LABEL = 'qmix_state_ones'
SMOKE_SCRIPTS = ('scripts/smoke_test_submit_corridor_ones.py',)
validate_map = baseline.validate_map


def build_plans(repo):
    plans = baseline.build_plans(repo)
    for plan in plans:
        extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
        extra['qmix_condition_input'] = 'ones'
        plan['exports'].update(GROUP_NAME=GROUP,
                               EXTRA_ARGS=' '.join(k+'='+v for k, v in extra.items()))
        plan['label'] = LABEL
        rename(plan, 'smac_corridor_{}_10m_s{}_ones20261027'.format(LABEL, plan['seed']))
        plan['inventory_note'] = (
            'Matched completed Corridor QMIX baseline: same CleanLearner, initialization, '
            'historical GRU + two-layer ELU action head and TD recipe; only mixer condition '
            'is ones in W1/W2/b1/b2 online and target; actor obs live; '
            'no gate/one-fill/extra loss/refresh/second mixer; 3 seeds, 10M/24h/24G')
    if len(plans) != 3 or {p['seed'] for p in plans} != {1, 2, 3}:
        raise RuntimeError('Expected only three Corridor all-ones seeds')
    return plans


def main():
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        baseline.require_capacity(baseline.parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_map)


if __name__ == '__main__':
    main()
