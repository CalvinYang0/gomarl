#!/usr/bin/env python3
"""Static gate: detached main TD, gate-only Q-change/MC-quality supervision."""
import copy
import shlex

from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as old_controls
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh, rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = 'static_gate_score_supervision_10m_12jobs_20261018'
LABELS = ('qmix_state_static10_aux_qchange', 'qmix_state_static10_aux_calibration')
SMOKE_SCRIPTS = ('scripts/smoke_test_gate_score_supervision.py',)


def reference_plans(repo):
    return ([p for p in old_controls(repo) if p['label'] == 'qmix_state_static10'] +
            [p for p in fresh(repo) if p['scene'] == 'smac_8m9m' and p['label'] == 'qmix_state_static10'])


def build_plans(repo):
    plans = []
    for original in reference_plans(repo):
        for label in LABELS:
            plan = copy.deepcopy(original)
            criterion = label.removeprefix('qmix_state_static10_aux_')
            extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
            extra.update(mixer_gate_aux_score=criterion, mixer_gate_aux_interval='100', mixer_gate_aux_coef='1.0')
            plan['label'] = label
            plan['exports']['GROUP_NAME'] = GROUP
            plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
            rename(plan, '{}_{}_10m_s{}_ablate20261018'.format(plan['scene'], label, plan['seed']))
            plan['inventory_note'] = ('Static gate; main TD detached from gate; independent gate optimizer/BCE; '
                'detached {} probability targets; same soft-p batch-mean probes; no L1/KL/variance penalty; '
                'MC version uses complete observed returns only'.format(criterion))
            plans.append(plan)
    if len(plans) != 12 or len({p['job_name'] for p in plans}) != 12:
        raise RuntimeError('Expected twelve unique gate-only supervision experiments')
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS)


if __name__ == '__main__':
    main()
