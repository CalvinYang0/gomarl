#!/usr/bin/env python3
"""Projected .3/.7 + fast8M, init1 + fast5M/freeze90; two maps, 18 jobs.

Only these fresh SMAC jobs opt into one normal-evaluation video every 1M.
Plan-only unless SUBMIT=YES; no GRF, cancellation, deletion or old-job restart.
"""
import argparse
import copy
import os
from pathlib import Path
import shlex
import subprocess

import ozstar_submit_projected_fastrefresh_four_maps_12jobs as parent
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_mixer_state_refresh_9jobs import require_capacity
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = 'projected_microtrick_init03_init07_freeze5m_video_18jobs_20261122'
MAPS = (('smac_corridor', 'corridor', 6, 24),
        ('smac_3s_vs_5z', '3s_vs_5z', 3, 5))
VARIANTS = {
    'init03': ('qmix_projected_init03_faststages8m', '0.3', '8000000', False),
    'init07': ('qmix_projected_init07_faststages8m', '0.7', '8000000', False),
    'freeze5m': ('qmix_projected_init1_faststages5m_freeze90', '1.0', '5000000', True),
}
SMOKE_SCRIPTS = ('scripts/smoke_test_projected_microtrick_video.py',)


def build_plans(repo, scene=None, variant=None):
    if scene is not None and scene not in {row[0] for row in MAPS}:
        raise ValueError('Unsupported map; use corridor or 3s_vs_5z, NOT 3s5z')
    if variant is not None and variant not in VARIANTS:
        raise ValueError('Unsupported projected-gate variant')
    templates = parent.build_plans(repo, 'smac_5m6m')
    plans = []
    for key, map_name, _, _ in MAPS:
        if scene is not None and key != scene:
            continue
        for option, (label, initial, full, freeze) in VARIANTS.items():
            if variant is not None and option != variant:
                continue
            for template in templates:
                plan = copy.deepcopy(template)
                x = dict(v.split('=', 1) for v in shlex.split(plan['exports']['EXTRA_ARGS']))
                x.update(mixer_input_initial_probability=initial,
                         mixer_state_refresh_full_steps=full,
                         disable_video_media='False', test_battle_videos='True',
                         test_battle_video_interval='1000000',
                         test_battle_video_episodes='1', test_battle_video_fps='6')
                if freeze:
                    x.update(learner='projected_onefill_freeze_refresh_learner',
                             mixer_soft_gate_freeze_threshold='0.9')
                plan.update(scene=key, map_name=map_name, label=label, target_steps=10000000)
                plan['exports'].update(GROUP_NAME=GROUP, MAP_NAME=map_name,
                    EXTRA_ARGS=shlex.join(k+'='+v for k, v in x.items()))
                rename(plan, '{}_{}_10m_s{}_microvideo20261122'.format(key, label, plan['seed']))
                plan['sbatch_args'] = ['--time=2-00:00:00' if arg.startswith('--time=') else arg
                                      for arg in plan['sbatch_args']]
                plan['inventory_note'] = (
                    'Historical projected continuous gate, NOT Bernoulli; init={}; faststages live at {}; '
                    '{}; same actor/head, optimizer and joint TD; online/target use x=p*held_state+(1-p); '
                    'one unselected first normal-test episode at each 1M milestone, simulated top-down MP4; '
                    'no extra tests/model forwards; media allowed in W&B offline run, sync required for cloud; '
                    '10M, seeds1/2/3, 28CPU/32G/48h; no GRF or checkpoint resume').format(
                        initial, full, 'at first update >=5M freeze BOTH gates: p>0.9 ->1, else0; '
                        'other weights keep training' if freeze else 'gate trainable throughout')
                plans.append(plan)
    expected = (1 if scene else 2) * (1 if variant else 3) * 3
    if len(plans) != expected or len({p['job_name'] for p in plans}) != expected:
        raise RuntimeError('Incorrect map/model/seed matrix')
    return plans


def validate_maps(scene=None):
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    from pysc2 import maps
    registry = get_smac_map_registry()
    sc2 = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any(p.is_file() and os.access(p, os.X_OK)
               for p in (sc2/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing executable SC2 binary')
    for key, name, allies, enemies in MAPS:
        if scene is not None and key != scene:
            continue
        spec = registry.get(name, {})
        if (spec.get('n_agents'), spec.get('n_enemies')) != (allies, enemies):
            raise RuntimeError('Incorrect installed map identity: '+name)
        if not (sc2/'Maps'/maps.get(name).path).is_file():
            raise RuntimeError('Missing map asset: '+name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(row[0] for row in MAPS))
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    args = parser.parse_args(argv)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.check_output(['quota', '-v'], text=True, timeout=30)
        require_capacity(parse_home_quota(quota))
        # Mandatory, not a warning-only video opt-in: verify the encoder before jobs.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
        from utils.battle_video import check_video_dependencies
        check_video_dependencies()
    submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.scene, args.variant),
        SMOKE_SCRIPTS, map_validator=lambda: validate_maps(args.scene))


if __name__ == '__main__':
    main()
