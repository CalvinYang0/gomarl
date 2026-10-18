#!/usr/bin/env python3
"""5m6m all-ones mixer condition; three seeds, 10M; plan-only by default."""
import copy
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as controls
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

GROUP = '5m6m_ones_10m_3seeds_20261018'
SMOKE_SCRIPTS = ('scripts/smoke_test_5m6m_ones.py',)


def build_plans(repo):
    plans = []
    for original in controls(repo):
        if original['label'] != 'qmix_state_static10':
            continue
        plan = copy.deepcopy(original)
        extra = dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))
        extra['mixer_input_control'] = 'ones'
        plan['label'] = 'qmix_state_ones'
        plan['exports']['GROUP_NAME'] = GROUP
        plan['exports']['EXTRA_ARGS'] = ' '.join(k + '=' + v for k, v in extra.items())
        rename(plan, 'smac_5m6m_qmix_state_ones_10m_s{}_ablate20261018'.format(plan['seed']))
        plan['inventory_note'] = ('All-ones mixer condition; ordinary actor observations unchanged; '
            'no gate or auxiliary losses; same settings as 8m9m ones except map and run namespace')
        plans.append(plan)
    if len(plans) != 3 or {p['seed'] for p in plans} != {1, 2, 3}:
        raise RuntimeError('Expected exactly three 5m6m all-ones seeds')
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
