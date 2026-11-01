#!/usr/bin/env python3
"""Exact-one initialization / sigmoid warmup freeze / +refresh; fresh 3x3 runs."""
import argparse
import copy
import os
import shlex
import subprocess

import ozstar_submit_soft_onefill_refresh_three_maps_18jobs as reference
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_3s5z_soft_refresh_live_9jobs import validate_map

GROUP = '3s5z_onefill_initial1_freeze5m_10m_9jobs_20261101'
MODES = {
    'initial1': ('qmix_onefill_projected_init1', 'projected_onefill_gate_learner', False, '1.0'),
    'freeze5m': ('qmix_soft_onefill_p10_freeze5m_t90', 'soft_onefill_freeze_learner', False, '0.1'),
    'freeze5m_refresh': ('qmix_soft_onefill_p10_freeze5m_t90_refresh2000k', 'soft_onefill_freeze_refresh_learner', True, '0.1'),
}
SMOKE_SCRIPTS = ('scripts/smoke_test_no_video_startup.py', 'scripts/smoke_test_onefill_warmup_freeze.py')


def build_plans(repo, variant='all'):
    if variant != 'all' and variant not in MODES:
        raise ValueError('Unknown gate variant')
    result = []
    for old in reference.build_plans(repo):
        if old['scene'] != 'smac_3s5z' or old['label'] != reference.LABELS[0]:
            continue
        for key, (label, learner, refresh, initial) in MODES.items():
            if variant != 'all' and key != variant:
                continue
            p = copy.deepcopy(old)
            x = dict(t.split('=', 1) for t in shlex.split(p['exports']['EXTRA_ARGS']))
            x.update(learner=learner, mixer_input_initial_probability=initial,
                mixer_state_refresh_enabled=str(refresh), mixer_state_refresh_stage_steps='2000000',
                mixer_state_refresh_schedule='linear', mixer_state_refresh_eval_live='True',
                mixer_soft_gate_freeze_steps='5000000', mixer_soft_gate_freeze_threshold='0.9',
                disable_video_media='True', test_battle_videos='False')
            p['exports'].update(GROUP_NAME=GROUP, EXTRA_ARGS=' '.join(k+'='+v for k, v in x.items()))
            p['label'] = label
            rename(p, 'smac_3s5z_{}_10m_s{}_warm20261101'.format(label, p['seed']))
            p['inventory_note'] = (
                'Old GRU+two-layer ELU actor, same single-QMIX soft TD; no video; '
                + ('exact initial1 direct coefficient, post-step projection [0,1], no freeze; ' if key == 'initial1' else
                   'original sigmoid initial .1; first update at/after5M freezes p>.9 to1, otherwise0; same mask online+target; ')
                + ('raw mixer-state refresh stage2M; ' if refresh else 'live mixer state; ')
                + 'eval mixer state live, closed fills1; 3seeds/10.05M/32G/28CPU/24h')
            result.append(p)
    expected = 9 if variant == 'all' else 3
    if len(result) != expected or len({p['job_name'] for p in result}) != expected:
        raise RuntimeError('Expected exact distinct new warmup jobs')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('all',)+tuple(MODES), default='all')
    args = parser.parse_args(argv)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=30)
        reference.require_capacity(reference.parse_home_quota(quota.stdout))
    reference.submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.variant),
        SMOKE_SCRIPTS, map_validator=validate_map)


if __name__ == '__main__':
    main()
