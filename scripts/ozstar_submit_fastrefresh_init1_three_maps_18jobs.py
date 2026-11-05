#!/usr/bin/env python3
"""The two Nov1 parent models, independently, on three maps x three seeds."""
import argparse
import copy
import os
from pathlib import Path
import subprocess

import ozstar_submit_3s5z_soft_refresh_live_9jobs as fast
import ozstar_submit_3s5z_onefill_warmup_9jobs as projected
import ozstar_submit_projected_fastrefresh_four_maps_12jobs as shared
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

GROUP = 'fastrefresh_and_projected_init1_three_maps_10m_18jobs_20261106'
MAPS = (
    ('smac_corridor', 'corridor', '2-00:00:00'),
    ('smac_mmm2', 'MMM2', '1-00:00:00'),
    ('smac_5m6m', '5m_vs_6m', '1-00:00:00'),
)
VARIANTS = {
    'faststages': fast.MODES['faster_stages'][0],
    'init1': projected.MODES['initial1'][0],
}
SMOKE_SCRIPTS = ('scripts/smoke_test_fastrefresh_init1_three_maps.py',)


def parent_plans(repo, variant):
    if variant == 'faststages':
        return fast.build_plans(repo, 'faster_stages')
    if variant == 'init1':
        return projected.build_plans(repo, 'initial1')
    raise ValueError('Unknown parent model')


def build_plans(repo, scene=None, variant=None):
    if scene is not None and scene not in {key for key, _, _ in MAPS}:
        raise ValueError('Unknown scene')
    if variant is not None and variant not in VARIANTS:
        raise ValueError('Unknown model')
    result = []
    for key, name, duration in MAPS:
        if scene is not None and scene != key:
            continue
        for kind, label in VARIANTS.items():
            if variant is not None and variant != kind:
                continue
            for original in parent_plans(repo, kind):
                p = copy.deepcopy(original)
                # Do not rebuild EXTRA_ARGS: preserve the exact original parent
                # learner, parameterization, schedule and actor/TD recipe.
                p.update(scene=key, map_name=name)
                p['exports'].update(GROUP_NAME=GROUP, MAP_NAME=name)
                rename(p, '{}_{}_10m_s{}_parents20261106'.format(key, label, p['seed']))
                p['sbatch_args'] = ['--time='+duration if arg.startswith('--time=') else arg
                    for arg in p['sbatch_args']]
                p['inventory_note'] = (
                    'Exact Nov1 {} parent (NOT projected+fast combination); only map/identity/walltime changed; '
                    'same old GRU+two-layer ELU head, unchanged joint TD/CPU optimizer, '
                    '28CPU/32G, nominal10M (T_MAX=10050000), no videos; walltime={}'.format(kind, duration))
                result.append(p)
    expected = (1 if scene else 3)*(1 if variant else 2)*3
    if len(result) != expected or len({p['job_name'] for p in result}) != expected:
        raise RuntimeError('Expected exact two-parent / three-map seed coverage')
    return result


def validate_maps(scene=None):
    for key, _, _ in MAPS:
        if scene is None or scene == key:
            shared.validate_maps(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(key for key, _, _ in MAPS))
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    args = parser.parse_args(argv)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=30)
        shared.require_capacity(shared.parse_home_quota(quota.stdout))
    shared.submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.scene, args.variant),
        SMOKE_SCRIPTS, map_validator=lambda: validate_maps(args.scene))


if __name__ == '__main__':
    main()
