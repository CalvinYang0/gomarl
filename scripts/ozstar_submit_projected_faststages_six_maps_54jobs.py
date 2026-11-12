#!/usr/bin/env python3
"""Transfer projected-init1 faststages8m and matched QMIX/ones to six maps."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import ozstar_submit_projected_fastrefresh_four_maps_12jobs as projected
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota
from ozstar_submit_5m6m_mixer_state_refresh_9jobs import require_capacity
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan
from ozstar_submit_qmix_ones_five_maps_30jobs import SCENES as OLD_SCENES, probe_grf

ROOT = Path(__file__).resolve().parents[1]
GROUP = 'projected_faststages_vs_qmix_ones_smac3_grf3_10m_20261113'
SUFFIX = 'projfasttransfer20261113'
SCENES = (
    ('smac_6h8z', '6h_vs_8z', 'smac', 6, 8),
    ('smac_10m11m', '10m_vs_11m', 'smac', 10, 11),
    ('smac_3s_vs_5z', '3s_vs_5z', 'smac', 3, 5),
) + tuple(row for row in OLD_SCENES if row[2] == 'grf')
LABELS = ('qmix', 'qmix_state_ones', projected.LABEL)
SEEDS = (1, 2, 3)
SMOKE_SCRIPTS = ('scripts/smoke_test_projected_faststages_six_maps.py',)


def selected_scenes(domain='all', scene=None):
    selected = tuple(row for row in SCENES if (domain == 'all' or row[2] == domain)
                     and (scene is None or row[0] == scene))
    if not selected:
        raise ValueError('No scenes match the requested domain and scene')
    return selected


def extras(serialized):
    return dict(item.split('=', 1) for item in shlex.split(serialized))


def build_plans(repo, domain='all', scene=None):
    profiles = _load_profiles(repo)
    if profiles.ALL_PROFILES['baseline'] != {}:
        raise RuntimeError('Expected baseline without auxiliary losses')
    parent = extras(projected.build_plans(repo)[0]['exports']['EXTRA_ARGS'])
    scenes = selected_scenes(domain, scene)
    result = []
    for key, map_name, env, _, _ in scenes:
        common = extras(_extra_args(profiles, 'baseline', env))
        common.update(mixer='qmix', double_q='True', test_greedy='True',
                      disable_video_media='True', test_battle_videos='False',
                      test_value_diagnostics='True', test_value_diagnostics_interval='100000')
        if env == 'grf':
            common.update({'env_args.dump_freq': '0', 'env_args.write_full_episode_dumps': 'False',
                           'env_args.write_goal_dumps': 'False', 'env_args.render': 'False'})
        for label in LABELS:
            x = dict(common, learner='clean_learner', mixer_state_refresh_enabled='False',
                     qmix_condition_input='ones' if label == 'qmix_state_ones' else 'state')
            if label == projected.LABEL:
                # Keep the parent gate/refresh recipe; only its environment-specific
                # profile and recording policy come from this suite.
                x.update({k: v for k, v in parent.items() if k == 'learner' or k.startswith('mixer_')
                          or k in {'slot_heatmap_interval', 'qmix_condition_input'}})
                if env == 'smac':
                    x.update({k: v for k, v in parent.items() if k.startswith('test_policy_importance')
                              or k in {'test_visualizations_required', 'test_hyper_obs_importance'}})
            for seed in SEEDS:
                name = f'{key}_{label}_10m_s{seed}_{SUFFIX}'
                plan = _plan(repo, profiles, key, map_name, env, 'linear_baseline', seed, name, '32G', GROUP)
                plan['exports'].update(MODEL_TYPE='qmix_minimal', T_MAX='10050000', TEST_INTERVAL='10000',
                                       EXTRA_ARGS=' '.join(k+'='+v for k, v in x.items()))
                plan['sbatch_args'] = ['--time=1-00:00:00' if a.startswith('--time=') else a
                                       for a in plan['sbatch_args']]
                plan.update(label=label, target_steps=10000000,
                            win_metric='test_game_win_mean' if env == 'grf' else 'test_battle_won_mean',
                            inventory_note='Matched historical two-layer ELU actor, fresh 10M, 28CPU/32G/24h; '
                            'projected variant uses exact-one learned gates, quadratic refresh to live at 8M, '
                            'and live eval; controls use live or constant mixer input; no videos or checkpoints')
                result.append(plan)
    expected = len(scenes) * len(LABELS) * len(SEEDS)
    if len(result) != expected or len({p['job_name'] for p in result}) != expected:
        raise RuntimeError('Unexpected job matrix')
    return result


def validate_environments(scenes):
    smac = [row for row in scenes if row[2] == 'smac']
    if smac:
        from pysc2 import maps
        from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry

        registry = get_smac_map_registry()
        root = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII')).expanduser()
        if not any(p.is_file() and os.access(p, os.X_OK) for p in (root/'Versions').glob('Base*/SC2_x64')):
            raise RuntimeError(f'Missing executable SC2 binary under {root}')
        for _, name, _, allies, enemies in smac:
            params = registry.get(name, {})
            if (params.get('n_agents'), params.get('n_enemies')) != (allies, enemies):
                raise RuntimeError(f'Installed map identity differs: {name}')
            asset = root/'Maps'/maps.get(name).path
            if not asset.is_file():
                raise RuntimeError(f'Missing installed map asset: {asset}')
            print(f'SMAC assets preflight passed: {name}', flush=True)
    for _, name, env, _, _ in scenes:
        if env == 'grf':
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-grf', name],
                           check=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                                           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=('all', 'smac', 'grf'), default='all')
    parser.add_argument('--scene', choices=tuple(row[0] for row in SCENES), help='Submit at most nine jobs')
    parser.add_argument('--probe-grf', choices=tuple(row[1] for row in SCENES if row[2] == 'grf'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_grf:
        probe_grf(args.probe_grf)
        return
    scenes = selected_scenes(args.domain, args.scene)
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], check=True, text=True, capture_output=True, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
        print('Quota check is a startup floor, not a capacity guarantee for the entire batch. '
              'Use --scene for nine-job waves when necessary.', flush=True)
    os.environ.setdefault('REPO_DIR', str(ROOT))
    submit_suite(GROUP, __file__, lambda repo: build_plans(repo, args.domain, args.scene),
                 SMOKE_SCRIPTS, map_validator=lambda: validate_environments(scenes))


if __name__ == '__main__':
    main()
