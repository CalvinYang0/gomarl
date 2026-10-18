#!/usr/bin/env python3
"""Two independent MC-error ablations x two maps x three seeds, 10M each."""
import copy
from pathlib import Path
import shlex

from ozstar_submit_5m6m_calibration_mean_shared_3seeds import build_plans as shared_plans
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_fixed_head_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = 'calibration_mean_two_map_ablation_10m_12jobs_20261018'
LABELS = ('qmix_state_group_calibration_mean_isolated',
          'qmix_state_group_calibration_mean_plainmask')
SMOKE_SCRIPTS = ('scripts/smoke_test_calibration_mean_shared.py',
                 'scripts/smoke_test_calibration_mean_ablation.py')


def reference_plans(repo, scene, map_name):
    """Matched reference templates only; 8m9m references are NOT submitted."""
    references = shared_plans(repo)
    if scene == 'smac_5m6m' and map_name == '5m_vs_6m':
        return references
    if scene != 'smac_8m9m' or map_name != '8m_vs_9m':
        raise ValueError('Unsupported map pair')
    extras = {p['seed']: p['exports']['EXTRA_ARGS'] for p in references}
    templates = [p for p in build_fixed_head_plans(repo, scene, map_name, GROUP) if p['label'] == 'qmix']
    for p in templates:
        p['exports']['EXTRA_ARGS'] = extras[p['seed']]
    return templates


def build_plans(repo):
    plans = []
    for scene, map_name in (('smac_5m6m', '5m_vs_6m'), ('smac_8m9m', '8m_vs_9m')):
        for original in reference_plans(repo, scene, map_name):
            for label in LABELS:
                plan = copy.deepcopy(original)
                extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
                if label.endswith('_isolated'):
                    extra['mixer_input_detach_unimportant'] = 'True'
                    note = 'Gradient-only: detach unimportant TD from actor; legacy mask-concat retained'
                else:
                    extra['mixer_input_append_mask'] = 'False'
                    note = 'Input-only: state times group mask, no membership features; shared actor gradient retained'
                plan['label'] = label
                plan['exports']['GROUP_NAME'] = GROUP
                plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
                rename(plan, '{}_{}_10m_s{}_ablate20261018'.format(scene, label, plan['seed']))
                plan['inventory_note'] = (note + '; same MC-MSE quality score, complete-return probes and '
                    'batch-mean field replacement as calibration_mean_shared; two independent ablations')
                if scene == 'smac_8m9m':
                    plan['inventory_note'] += '; matched shared/mask-concat reference not submitted in this suite'
                plans.append(plan)
    if len(plans) != 12 or len({p['job_name'] for p in plans}) != 12:
        raise RuntimeError('Expected exactly 12 distinct ablation jobs')
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS)


if __name__ == '__main__':
    main()
