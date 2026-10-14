#!/usr/bin/env python3
"""Fill the two missing cells: Q-change/isolated and MC-error/shared TD."""
import copy
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as old_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh_plans, rename

GROUP = '5m6m_score_gradient_factorial_10m_6jobs_20261014'
SMOKE_SCRIPTS = ('scripts/smoke_test_mixer_input_controls.py',
                 'scripts/smoke_test_mixer_calibration.py',
                 'scripts/smoke_test_mixer_gradient_isolation.py',
                 'scripts/smoke_test_score_gradient_ablation.py',
                 'scripts/smoke_test_grouping_heatmap_backfill.py',
                 'scripts/smoke_test_submit_score_gradient_ablation.py')


def build_plans(repo):
    plans = []
    variants = (
        (old_plans(repo), 'qmix_state_group_qchange', 'qmix_state_group_qchange_isolated', True),
        (fresh_plans(repo), 'qmix_state_group_calibration_isolated', 'qmix_state_group_calibration_shared', False),
    )
    for source, old_label, label, detached in variants:
        for original in source:
            if original['scene'] != 'smac_5m6m' or original['label'] != old_label:
                continue
            plan = copy.deepcopy(original)
            plan['label'] = label
            plan['exports']['GROUP_NAME'] = GROUP
            extra = dict(a.split('=', 1) for a in shlex.split(plan['exports']['EXTRA_ARGS']))
            extra['mixer_input_detach_unimportant'] = str(detached)
            plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
            name = 'smac_5m6m_{}_10m_s{}_ablate20261014'.format(label, plan['seed'])
            rename(plan, name)
            plan['inventory_note'] = 'Factorial ablation: only unimportant gradient routing differs from its matched score control'
            plans.append(plan)
    if len(plans) != 6 or len({p['job_name'] for p in plans}) != 6:
        raise RuntimeError('Expected six distinct new 5m6m ablation runs')
    return plans


def main():
    if os.environ.get('SUBMIT') == 'YES':
        free = submitter.home_quota_free_gib()
        if free < 2:
            raise RuntimeError('Require at least 2 GiB free; no jobs submitted')
        from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
        validate_installed_map()
        print('Six additional video-enabled jobs; 2 GiB is only a startup guard.', flush=True)
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__),
                       SMOKE_SCRIPTS=SMOKE_SCRIPTS, build_plans=build_plans)
    original = {key: getattr(submitter, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(submitter, key, value)
        submitter.main()
    finally:
        for key, value in original.items():
            setattr(submitter, key, value)


if __name__ == '__main__':
    main()
