#!/usr/bin/env python3
"""Three clean ablations of old Q-change/shared TD; three seeds, 10M each."""
import copy
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as controls
from ozstar_submit_5m6m_score_gradient_ablation_6jobs import build_plans as existing_ablation
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

GROUP = '5m6m_qchange_three_ablation_10m_9jobs_20261015'
NEW_LABELS = ('qmix_state_group_qchange_plainmask', 'qmix_state_group_variance')
SMOKE_SCRIPTS = ('scripts/smoke_test_mixer_input_controls.py',
                 'scripts/smoke_test_mixer_calibration.py',
                 'scripts/smoke_test_mixer_gradient_isolation.py',
                 'scripts/smoke_test_qchange_three_ablation.py',
                 'scripts/smoke_test_grouping_heatmap_backfill.py',
                 'scripts/smoke_test_submit_qchange_three_ablation.py')


def build_plans(repo):
    # Reuse the exact already-declared gradient-only experiment names. Active
    # or completed jobs are retained by the common submitter; no duplicate runs.
    plans = [copy.deepcopy(p) for p in existing_ablation(repo)
             if p['label'] == 'qmix_state_group_qchange_isolated']
    for original in controls(repo):
        if original['label'] != 'qmix_state_group_qchange':
            continue
        for label in NEW_LABELS:
            plan = copy.deepcopy(original)
            plan['label'] = label
            plan['exports']['GROUP_NAME'] = GROUP
            extra = dict(a.split('=', 1) for a in shlex.split(plan['exports']['EXTRA_ARGS']))
            if label == 'qmix_state_group_qchange_plainmask':
                extra['mixer_input_append_mask'] = 'False'
                note = 'Input-only ablation: original state dimensionality; no concatenated mask features'
            else:
                extra['mixer_input_control'] = 'group_variance'
                note = 'Score-only ablation: signed population Var(Q_tot)-Var(Q_tot_slot_replaced)'
            # No detach change, perturbation change, extra loss, or KL objective.
            plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
            name = 'smac_5m6m_{}_10m_s{}_ablate20261015'.format(label, plan['seed'])
            rename(plan, name)
            plan['inventory_note'] = note + '; all other settings match old Q-change/shared control'
            plans.append(plan)
    if len(plans) != 9 or len({p['job_name'] for p in plans}) != 9:
        raise RuntimeError('Expected nine distinct ablation plans, including three reusable gradient-only runs')
    return plans


def main():
    if os.environ.get('SUBMIT') == 'YES':
        if submitter.home_quota_free_gib() < 2:
            raise RuntimeError('Require at least 2 GiB free; no jobs submitted')
        from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
        validate_installed_map()
        print('Up to nine media-enabled jobs; existing gradient-only jobs retained. 2 GiB is only a startup guard.', flush=True)
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__), SMOKE_SCRIPTS=SMOKE_SCRIPTS, build_plans=build_plans)
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
