#!/usr/bin/env python3
"""8m9m: constant mixer condition, gate L1, gate variance; three seeds each."""
import copy
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh, rename, validate_maps

GROUP = '8m9m_ones_gate_regularization_10m_9jobs_20261015'
LABELS = ('qmix_state_ones', 'qmix_state_static10_l1', 'qmix_state_static10_variance')
SMOKE_SCRIPTS = ('scripts/smoke_test_mixer_calibration.py',
                 'scripts/smoke_test_mixer_gate_regularization.py',
                 'scripts/smoke_test_submit_qchange_gate_ablation.py')


def build_plans(repo):
    plans = []
    for original in fresh(repo):
        if original['scene'] != 'smac_8m9m' or original['label'] != 'qmix_state_static10':
            continue
        for label in LABELS:
            plan = copy.deepcopy(original)
            extra = dict(v.split('=', 1) for v in shlex.split(plan['exports']['EXTRA_ARGS']))
            if label == 'qmix_state_ones':
                extra['mixer_input_control'] = 'ones'
                note = 'Only mixer condition replaced with all ones; ordinary actor observations unchanged; no gate'
            elif label == 'qmix_state_static10_l1':
                extra['mixer_gate_l1_coef'] = '0.01'
                note = 'Static gate plus 0.01*mean(keep probability); no variance penalty or KL'
            else:
                extra['mixer_gate_variance_coef'] = '0.01'
                note = 'Static gate plus detached per-slot Q_tot variance-removal penalty; no L1 or KL'
            plan['label'] = label
            plan['exports']['GROUP_NAME'] = GROUP
            plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
            rename(plan, 'smac_8m9m_{}_10m_s{}_ablate20261015'.format(label, plan['seed']))
            plan['inventory_note'] = note + '; all other settings match fresh 8m9m static-gate control'
            plans.append(plan)
    if len(plans) != 9 or len({p['job_name'] for p in plans}) != 9:
        raise RuntimeError('Expected nine distinct 8m9m runs')
    return plans


def submit_suite(group, suite_file, build, scripts, map_validator=None):
    if os.environ.get('SUBMIT') == 'YES':
        if submitter.home_quota_free_gib() < 2:
            raise RuntimeError('Require at least 2 GiB free; no jobs submitted')
        (validate_maps if map_validator is None else map_validator)()
        print('Existing exact jobs retained; 2 GiB is only a startup guard, not a batch capacity guarantee.', flush=True)
    replacement = dict(GROUP=group, SUITE_FILE=Path(suite_file), SMOKE_SCRIPTS=scripts, build_plans=build)
    original = {key: getattr(submitter, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(submitter, key, value)
        submitter.main()
    finally:
        for key, value in original.items():
            setattr(submitter, key, value)


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS)


if __name__ == '__main__':
    main()
