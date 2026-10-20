#!/usr/bin/env python3
"""8m9m bounded modulation + four 5m6m weight controls; plan-only by default."""
import shlex

from ozstar_submit_5m6m_onefill_gate_15jobs import configure_onefill_plan
from ozstar_submit_static_gate_score_supervision_12jobs import reference_plans
from ozstar_submit_5m6m_residual_zerofill_3jobs import build_plans as zero_plans
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = 'weight_composition_10m_15jobs_20261020'
CONDITIONS = {
    'qmix_weight_bounded_p10': ('smac_8m9m', 'bounded', False),
    'qmix_weight_sum_p10': ('smac_5m6m', 'sum', False),
    'qmix_weight_alpha10_p10': ('smac_5m6m', 'alpha_sum', False),
    'qmix_weight_sum_bothones': ('smac_5m6m', 'sum', True),
}
LABELS = tuple(CONDITIONS) + ('qmix_residual_zerofill_p10',)
SMOKE_SCRIPTS = ('scripts/smoke_test_weight_composition.py',)


def build_plans(repo):
    plans = zero_plans(repo)
    for plan in plans:
        # Reuse the exact zero-fill name, never schedule a duplicate cohort.
        plan['exports']['GROUP_NAME'] = GROUP
    for original in reference_plans(repo):
        for label, (scene, combination, both_ones) in CONDITIONS.items():
            if original['scene'] != scene:
                continue
            plan = configure_onefill_plan(original, label,
                dict(probability='0.1', objective='td', residual='True'), GROUP,
                '{}_{}_10m_s{}_weights20261020'.format(scene, label, original['seed']))
            extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
            extra.update(mixer_residual_combination=combination, mixer_residual_fill_value='1',
                mixer_residual_secondary_ones=str(both_ones), mixer_residual_alpha_initial='0.1')
            plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
            formula = {'bounded': 'abs(B(ones))*(1+0.1*tanh(R(x)-R(ones)))',
                       'sum': 'abs(B(ones)+R(x)); NO centering or 0.1 residual coefficient',
                       'alpha_sum': 'abs(alpha*B(ones)+(1-alpha)*R(x)); global learned alpha0=0.1'}[combination]
            plan['inventory_note'] = ('Weight composition, NOT Q_tot sum: ' + formula +
                '; x=' + ('ones; no operative gate' if both_ones else 'mask*state+(1-mask); p0=0.1') +
                '; zero secondary output-head initialization; base b1/V always ones; joint TD, no extra KL/MC; '
                'hard-ST train, soft-p target/diagnostics; branch weights and alpha diagnostics')
            plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 15:
        raise RuntimeError('Expected fifteen unique weight-composition jobs')
    for label in LABELS:
        selected = [p for p in plans if p['label'] == label]
        scene = 'smac_8m9m' if label == 'qmix_weight_bounded_p10' else 'smac_5m6m'
        if {p['seed'] for p in selected} != {1, 2, 3} or any(p['scene'] != scene for p in selected):
            raise RuntimeError('Unexpected map/seed coverage for ' + label)
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS)


if __name__ == '__main__':
    main()
