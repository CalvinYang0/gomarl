#!/usr/bin/env python3
"""Q-change control with MC-error ranking only; shared TD, batch-mean probes."""
import copy
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as controls
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

GROUP = '5m6m_calibration_mean_shared_10m_3seeds_20261018'
LABEL = 'qmix_state_group_calibration_mean_shared'
SMOKE_SCRIPTS = ('scripts/smoke_test_calibration_mean_shared.py',)


def build_plans(repo):
    plans = []
    for original in controls(repo):
        if original['label'] != 'qmix_state_group_qchange':
            continue
        plan = copy.deepcopy(original)
        extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
        extra.update(mixer_input_control='group_calibration', mixer_calibration_perturbation='batch_mean')
        plan['label'] = LABEL
        plan['exports']['GROUP_NAME'] = GROUP
        plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
        rename(plan, 'smac_5m6m_{}_10m_s{}_ablate20261018'.format(LABEL, plan['seed']))
        plan['inventory_note'] = ('MC-error quality ranking on complete episodes; shared auxiliary actor gradient; '
            'legacy mask-concat and Q-change batch-mean perturbation retained; no new TD loss')
        plans.append(plan)
    if len(plans) != 3 or {p['seed'] for p in plans} != {1, 2, 3}:
        raise RuntimeError('Expected exactly three calibration-mean/shared seeds')
    return plans


def main():
    if os.environ.get('SUBMIT') == 'YES':
        if submitter.home_quota_free_gib() < 2:
            raise RuntimeError('Require at least 2 GiB free; no jobs submitted')
        from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
        validate_installed_map()
        print('Three media-enabled jobs; 2 GiB is a startup guard, not a capacity guarantee.', flush=True)
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
