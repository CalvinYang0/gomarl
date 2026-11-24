#!/usr/bin/env python3
"""User heatmap mask retraining + D extra noise fields; 3 maps x 3 models x 3 seeds.

Plan-only unless SUBMIT=YES. No cloud writes, deletion, cancellation or resume.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import shlex
import subprocess

import ozstar_submit_projected_fastrefresh_four_maps_12jobs as parent
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename

ROOT = Path(__file__).resolve().parents[1]
GROUP = 'smac_user_frozenmask_noise1x_10m_27jobs_20261124'
MAPS = (('smac_corridor', 'corridor', 282), ('smac_3s_vs_5z', '3s_vs_5z', 68),
        ('smac_5m6m', '5m_vs_6m', 98))
MODELS = ('qmix_fixed_heatmap_seed3', 'qmix_noise1x', 'qmix_projected_init1_faststages8m_noise1x')
MASK_PATH = 'src/config/masks/frozen_heatmaps_seed3_20261124.json'
SMOKE_SCRIPTS = ('scripts/smoke_test_fixed_mask_noise.py',)


def build_plans(repo, scene=None, experiment='all'):
    if scene is not None and scene not in {row[0] for row in MAPS}:
        raise ValueError('Unsupported map; use corridor / 3s_vs_5z / 5m_vs_6m')
    if experiment not in {'all', 'mask', 'noise'}:
        raise ValueError('Experiment must be mask/noise/all')
    mask_path = repo/MASK_PATH
    mask_digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    templates = parent.build_plans(repo, 'smac_5m6m')
    plans = []
    for key, map_name, raw_dim in MAPS:
        if scene is not None and scene != key:
            continue
        for label in MODELS:
            noise = label != MODELS[0]
            if (experiment == 'mask' and noise) or (experiment == 'noise' and not noise):
                continue
            for template in templates:
                p = copy.deepcopy(template)
                x = dict(v.split('=', 1) for v in shlex.split(p['exports']['EXTRA_ARGS']))
                gated = label == MODELS[2]
                x.update(learner='noise_projected_refresh_learner' if gated else 'fixed_onefill_mask_learner',
                    qmix_condition_input='state', mixer_input_initial_probability='1.0',
                    mixer_state_refresh_enabled=str(gated),
                    mixer_fixed_mask_kind='all_ones' if noise else 'heatmap',
                    disable_video_media='True', test_battle_videos='False')
                if not noise:
                    x.update(mixer_fixed_mask_bundle=str(mask_path), mixer_fixed_mask_sha256=mask_digest)
                x.update(mixer_state_noise=str(noise), mixer_state_noise_seed='271828')
                p.update(scene=key, map_name=map_name, label=label, target_steps=10000000,
                         raw_state_dim=raw_dim, mixer_state_dim=raw_dim*(2 if noise else 1))
                p['exports'].update(MAP_NAME=map_name, GROUP_NAME=GROUP, T_MAX='10050000',
                    EXTRA_ARGS=shlex.join(k+'='+v for k, v in x.items()))
                rename(p, '{}_{}_10m_s{}_masknoise20261124'.format(key, label, p['seed']))
                p['sbatch_args'] = ['--time=2-00:00:00' if a.startswith('--time=') else a for a in p['sbatch_args']]
                p['inventory_note'] = (
                    'Fresh 10M, 48h, 28CPU/32G, seeds1/2/3; historical actor/TD, no extra loss. '+
                    ('Fixed image-specified seed3 mask shared across all 3 NEW seeds, x=m*s+(1-m); '
                     'fixed from step0, NO curriculum/learned gate/weight transfer; bundle SHA256='+mask_digest
                     if not noise else
                     'Global mixer state [s, U(0,1)^D]; obs unchanged; private worker train/eval RNG; '
                     'noise collected once and stored in replay, same online/target condition. '+
                     ('Continuous init1 gate + original 8M refresh of FULL concatenated state; '
                      'combined-method noise robustness, not isolated gate effect.' if gated else
                      'Ungated QMIX identity-input control, live state from step0.')))
                plans.append(p)
    count = (1 if scene else 3) * ({'all': 3, 'mask': 1, 'noise': 2}[experiment]) * 3
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != count:
        raise RuntimeError('Incorrect map/model/seed matrix')
    return plans


def validate_maps(scene=None):
    from pysc2 import maps
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    sc2 = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any(p.is_file() and os.access(p, os.X_OK) for p in (sc2/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing executable SC2 binary')
    from envs import REGISTRY
    import yaml
    cfg = yaml.safe_load((ROOT/'src/config/envs/sc2.yaml').read_text())
    for key, name, dim in MAPS:
        if scene is not None and key != scene:
            continue
        if name not in registry or not (sc2/'Maps'/maps.get(name).path).is_file():
            raise RuntimeError('Missing SMAC map: '+name)
        for noise in (False, True):
            env = REGISTRY['sc2'](**dict(cfg['env_args'], map_name=name, seed=1, mixer_state_noise=noise))
            try:
                if env.get_env_info()['state_shape'] != dim*(2 if noise else 1):
                    raise RuntimeError('Installed state layout changed; do not use the frozen mask')
            finally:
                env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(row[0] for row in MAPS))
    parser.add_argument('--experiment', choices=('all', 'mask', 'noise'), default='all')
    args = parser.parse_args(argv)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.check_output(['quota', '-v'], text=True, timeout=30)
        parent.require_capacity(parent.parse_home_quota(quota))
    os.environ.setdefault('REPO_DIR', str(ROOT))
    parent.submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.scene, args.experiment),
        SMOKE_SCRIPTS, map_validator=lambda: validate_maps(args.scene))


if __name__ == '__main__':
    main()
