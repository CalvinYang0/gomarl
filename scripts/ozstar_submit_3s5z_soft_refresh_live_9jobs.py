#!/usr/bin/env python3
"""Three opt-in refresh variants x seeds1/2/3 on 3s5z; no submission by default."""
import argparse
import copy
import os
import shlex
import subprocess

import ozstar_submit_soft_onefill_refresh_three_maps_18jobs as reference
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

GROUP = '3s5z_soft_refresh_live_10m_9jobs_20261101'
MODES = {
    'eval_live': ('qmix_soft_onefill_p10_refresh_2000k_evallive', 'linear'),
    'faster_stages': ('qmix_soft_onefill_p10_refresh_faststages8m_evallive', 'faster_stages'),
    'larger_jumps': ('qmix_soft_onefill_p10_refresh_bigjumps8m_evallive', 'larger_jumps'),
}
SMOKE_SCRIPTS = ('scripts/smoke_test_refresh_live_schedules.py',)


def build_plans(repo, variant='all'):
    if variant != 'all' and variant not in MODES:
        raise ValueError('Unknown refresh variant')
    plans = []
    for old in reference.build_plans(repo):
        if old['scene'] != 'smac_3s5z' or old['label'] != reference.LABELS[1]:
            continue
        for key, (label, mode) in MODES.items():
            if variant != 'all' and key != variant:
                continue
            p = copy.deepcopy(old)
            extra = dict(x.split('=', 1) for x in shlex.split(p['exports']['EXTRA_ARGS']))
            extra.update(mixer_state_refresh_schedule=mode, mixer_state_refresh_eval_live='True',
                mixer_state_refresh_full_steps='8000000', disable_video_media='True', test_battle_videos='False')
            p['exports'].update(GROUP_NAME=GROUP, EXTRA_ARGS=' '.join(k+'='+v for k, v in extra.items()))
            p['label'] = label
            rename(p, 'smac_3s5z_{}_10m_s{}_refreshlive20261101'.format(label, p['seed']))
            p['inventory_note'] = (
                'Based on 20261028 soft onefill+refresh2000k: same static sigmoid p0=.1, joint TD, '
                'single QMIX, old GRU+two-layer ELU head; eval mixer state live incl terminal; '
                'schedule={}; S=2M; H=8M for new schedules; actor obs live; no videos; 32G/28CPU/24h'.format(mode))
            plans.append(p)
    expected = 9 if variant == 'all' else 3
    if len(plans) != expected or len({p['job_name'] for p in plans}) != expected:
        raise RuntimeError('Incorrect new-cohort plan identity')
    return plans


def validate_map():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    from pathlib import Path
    registry = get_smac_map_registry()
    p = registry.get('3s5z_vs_3s6z', {})
    root = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if (p.get('n_agents'), p.get('n_enemies')) != (8, 9):
        raise RuntimeError('Installed 3s5z map identity differs')
    if not any((root/'Versions').glob('Base*/SC2_x64')) or not any((root/'Maps').rglob('3s5z_vs_3s6z.SC2Map')):
        raise RuntimeError('Missing SC2 binary or 3s5z map')


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
