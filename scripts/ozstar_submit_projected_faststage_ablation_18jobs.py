#!/usr/bin/env python3
"""Three projected-gate variants on 3s5z and 5m6m, three seeds each."""
import argparse
import os
import shlex
import subprocess

import ozstar_submit_projected_fastrefresh_four_maps_12jobs as reference
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename


GROUP = 'projected_faststage_ablation_10m_18jobs_20261108'
SCENES = ('smac_3s5z', 'smac_5m6m')
VARIANTS = {
    'init01_8m': ('qmix_projected_init01_faststages8m', '0.1', '8000000', False),
    'init1_5m': ('qmix_projected_init1_faststages5m', '1.0', '5000000', False),
    'init1_5m_freeze90': ('qmix_projected_init1_faststages5m_freeze90', '1.0', '5000000', True),
}
SMOKE_SCRIPTS = ('scripts/smoke_test_projected_faststage_ablation.py',)


def build_plans(repo, scene=None, variant=None):
    if scene is not None and scene not in SCENES:
        raise ValueError('Unknown scene')
    if variant is not None and variant not in VARIANTS:
        raise ValueError('Unknown variant')
    plans = []
    for name in ((scene,) if scene else SCENES):
        for key in ((variant,) if variant else VARIANTS):
            label, initial, full, freeze = VARIANTS[key]
            for p in reference.build_plans(repo, name):
                extra = dict(t.split('=', 1) for t in shlex.split(p['exports']['EXTRA_ARGS']))
                extra.update(mixer_input_initial_probability=initial, mixer_state_refresh_full_steps=full)
                if freeze:
                    extra.update(learner='projected_onefill_freeze_refresh_learner',
                        mixer_soft_gate_freeze_threshold='0.9')
                p['label'] = label
                p['exports'].update(GROUP_NAME=GROUP,
                    EXTRA_ARGS=shlex.join(k+'='+v for k, v in extra.items()))
                rename(p, '{}_{}_10m_s{}_fastablate20261108'.format(name, label, p['seed']))
                p['inventory_note'] = (
                    'Historical projected init1 + faststages8m parent; direct coefficient p0={}, '
                    'refresh reaches live at {} env steps; {}. Same actor/head, joint TD, '
                    '10M, seeds 1/2/3, 28CPU/32G/24h, eval live, no videos'.format(initial, full,
                        'one-time freeze of online/target gates, p>0.9 -> 1, else 0' if freeze
                        else 'gate trainable throughout'))
                plans.append(p)
    expected = (1 if scene else len(SCENES)) * (1 if variant else len(VARIANTS)) * 3
    if len(plans) != expected or len({p['job_name'] for p in plans}) != expected:
        raise RuntimeError('Unexpected projected faststage ablation plan coverage')
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    args = parser.parse_args(argv)

    def validate_maps():
        for scene in ((args.scene,) if args.scene else SCENES):
            reference.validate_maps(scene)

    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=30)
        reference.require_capacity(reference.parse_home_quota(quota.stdout))
    reference.submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.scene, args.variant),
        SMOKE_SCRIPTS, map_validator=validate_maps)


if __name__ == '__main__':
    main()
