#!/usr/bin/env python3
"""Five one-fill gate ablations on 5m6m, three seeds; plan-only by default."""
import copy
import shlex

from ozstar_submit_static_gate_score_supervision_12jobs import reference_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map

GROUP = '5m6m_onefill_gate_10m_15jobs_20261019'
CONDITIONS = {
    'qmix_onefill_p10': dict(probability='0.1', objective='td', residual='False'),
    'qmix_onefill_p20': dict(probability='0.2', objective='td', residual='False'),
    'qmix_onefill_p10_kl10': dict(probability='0.1', objective='kl', residual='False'),
    'qmix_onefill_p10_mc': dict(probability='0.1', objective='mc', residual='False'),
    'qmix_onefill_p10_residual': dict(probability='0.1', objective='td', residual='True'),
}
LABELS = tuple(CONDITIONS)
SMOKE_SCRIPTS = ('scripts/smoke_test_onefill_gate.py',)


def configure_onefill_plan(original, label, condition, group, name):
    """Preserve map/resource profile while removing legacy probe controls."""
    plan = copy.deepcopy(original)
    extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
    for key in tuple(extra):
        if 'threshold' in key or key in {'slot_group_interval', 'slot_importance_samples',
                   'slot_importance_ema', 'slot_demote_ratio', 'mixer_input_control'}:
            extra.pop(key)
    extra.update(learner='onefill_gate_learner', mixer_input_initial_probability=condition['probability'],
        mixer_input_gate_temperature='0.5', mixer_onefill_objective=condition['objective'],
        mixer_onefill_residual=condition['residual'], mixer_onefill_residual_scale='0.1',
        mixer_onefill_kl_prior='0.1', mixer_onefill_mc_coef='1.0',
        clean_adaptive_auxiliary_target_ratio='0.1', clean_adaptive_auxiliary_ema_decay='0.99',
        clean_adaptive_auxiliary_eps='1e-8', clean_adaptive_auxiliary_max_coef='100.0',
        mixer_gate_aux_score='none', mixer_gate_l1_coef='0.0', mixer_gate_variance_coef='0.0',
        mixer_input_detach_unimportant='False', slot_heatmap_interval='100000')
    plan['label'] = label
    plan['exports']['GROUP_NAME'] = group
    plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
    return rename(plan, name)


def build_plans(repo):
    plans = []
    for original in reference_plans(repo):
        if original['scene'] != 'smac_5m6m':
            continue
        for label, condition in CONDITIONS.items():
            plan = configure_onefill_plan(original, label, condition, GROUP,
                '{}_{}_10m_s{}_onefill20261019'.format(original['scene'], label, original['seed']))
            routing = {
                'td': 'TD jointly updates actor, mixer and gate',
                'kl': 'TD plus Bernoulli KL KEEP prior 0.1; detached EMA loss-value scaling; no extra TD path',
                'mc': 'Main TD gate detached; direct complete-return MSE updates gate only, every batch; separate optimizer/clipping',
            }[condition['objective']]
            plan['inventory_note'] = ('One-filled hard-ST stochastic gate; no test threshold, mask features or field probes; '
                'p0={probability}; objective={objective}; residual={residual}; '.format(**condition) + routing +
                ('; all-ones base plus centered W1/W2 correction, not Q_tot sum' if condition['residual'] == 'True' else '') +
                '; historical zero-fill not reused')
            plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 15:
        raise RuntimeError('Expected fifteen unique 5m6m one-fill jobs')
    if any(p['scene'] != 'smac_5m6m' or p['map_name'] != '5m_vs_6m' for p in plans):
        raise RuntimeError('This suite must never submit another map')
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_installed_map)


if __name__ == '__main__':
    main()
