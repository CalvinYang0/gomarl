#!/usr/bin/env python3
"""Matched residual ablation: secondary input mask*state, all-ones base kept."""
import copy
import shlex

from ozstar_submit_5m6m_onefill_gate_15jobs import build_plans as references
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map

GROUP = '5m6m_residual_zerofill_10m_3jobs_20261020'
LABEL = 'qmix_residual_zerofill_p10'
SMOKE_SCRIPTS = ('scripts/smoke_test_residual_zerofill.py',)


def build_plans(repo):
    plans = []
    for original in references(repo):
        if original['label'] != 'qmix_onefill_p10_residual':
            continue
        plan = copy.deepcopy(original)
        extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
        extra['mixer_residual_fill_value'] = '0'
        plan['label'] = LABEL
        plan['exports']['GROUP_NAME'] = GROUP
        plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
        rename(plan, 'smac_5m6m_{}_10m_s{}_ablate20261020'.format(LABEL, plan['seed']))
        plan['inventory_note'] = ('Matched to onefill p10 residual: ONLY secondary closed input/reference becomes zero; '
            'W=abs(B(ones)+0.1*(R(mask*state)-R(zeros))); base b1/V still all ones; '
            'p0=0.1, same hard-ST training/soft-p target, zero residual-head initialization, joint TD; no KL/MC')
        plans.append(plan)
    if (len(plans) != 3 or len({p['job_name'] for p in plans}) != 3 or
            {p['seed'] for p in plans} != {1, 2, 3} or
            any(p['scene'] != 'smac_5m6m' or p['map_name'] != '5m_vs_6m' for p in plans)):
        raise RuntimeError('Expected exactly three unique 5m6m zero-fill residual seeds')
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_installed_map)


if __name__ == '__main__':
    main()
