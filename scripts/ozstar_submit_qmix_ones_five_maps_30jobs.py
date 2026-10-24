#!/usr/bin/env python3
"""Matched QMIX vs constant mixer input; default SMAC-only = 12 jobs.

--domain all explicitly includes the three GRF maps for 30 jobs.

Plan-only by default. SUBMIT=YES checks byte/inode quota, installed SC2 assets,
selected GRF reset/step probes, model smoke tests and all missing Slurm plans.
No cancellation, cleanup, resume, gate, auxiliary loss or action-head changes.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from ozstar_submit_linear_single_head_suite import _load_profiles, _plan, _extra_args
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = 'qmix_vs_ones_grf3_mmm2_3s5z_10m_30jobs_20261025'
SMAC_GROUP = 'qmix_vs_ones_mmm2_3s5z_10m_12jobs_20261025'
SCENES = (
    ('grf_counter', 'academy_counterattack_easy', 'grf', 4, 30),
    ('grf_pass', 'academy_pass_and_shoot_with_keeper', 'grf', 2, 22),
    ('grf_3v1', 'academy_3_vs_1_with_keeper', 'grf', 3, 26),
    ('smac_mmm2', 'MMM2', 'smac', 10, 12),
    ('smac_3s5z', '3s5z_vs_3s6z', 'smac', 8, 9),
)
SEEDS = (1, 2, 3)
LABELS = ('qmix', 'qmix_state_ones')
SMOKE_SCRIPTS = ('scripts/smoke_test_qmix_ones_five_maps.py',)
MIN_FREE_GIB, MIN_FREE_INODES = 5.0, 50000


def selected_scenes(domain):
    if domain not in {'smac', 'all'}:
        raise ValueError('domain must be smac or all')
    return tuple(s for s in SCENES if domain == 'all' or s[2] == domain)


def group_for(domain):
    selected_scenes(domain)
    return SMAC_GROUP if domain == 'smac' else GROUP


def build_plans(repo, domain='all'):
    scenes, group = selected_scenes(domain), group_for(domain)
    profiles = _load_profiles(repo)
    if profiles.ALL_PROFILES['baseline'] != {}:
        raise RuntimeError('Expected ungated baseline without auxiliary losses')
    plans = []
    for scene, map_name, domain, _, _ in scenes:
        for label in LABELS:
            for seed in SEEDS:
                name = '{}_{}_10m_s{}_stateones20261025'.format(scene, label, seed)
                plan = _plan(repo, profiles, scene, map_name, domain, 'linear_baseline', seed, name, '32G', group)
                extra = dict(arg.split('=', 1) for arg in _extra_args(profiles, 'baseline', domain).split())
                extra.update(learner='clean_learner', mixer='qmix',
                    qmix_condition_input='state' if label == 'qmix' else 'ones',
                    mixer_state_refresh_enabled='False', test_greedy='True',
                    test_value_diagnostics='True', test_value_diagnostics_interval='100000',
                    test_battle_videos=str(domain == 'smac'),
                    test_battle_video_interval='1000000', test_battle_video_episodes='10')
                plan['exports'].update(MODEL_TYPE='qmix_minimal', T_MAX='10050000', TEST_INTERVAL='10000',
                    EXTRA_ARGS=' '.join(k+'='+v for k, v in extra.items()))
                plan.update(label=label, target_steps=10000000,
                    inventory_note='Matched historical GRU + two-layer ELU action head; SAME CleanLearner, QMixer, '
                        'TD(lambda), initialization and optimizer; only qmix_condition_input differs within each pair; '
                        'all W1/W2/b1/b2 state paths controlled; actor obs and Qi live; no gate/mask/aux/refresh; '
                        'fresh 10M, 24h, 28 CPU/32G; original domain rewards/evaluation preserved')
                plan['sbatch_args'] = ['--time=1-00:00:00' if a.startswith('--time=') else a for a in plan['sbatch_args']]
                plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != len(scenes)*6:
        raise RuntimeError('Unexpected job count for selected maps')
    return plans


def require_capacity(quota):
    print('/home free: {:.3f} GiB / {} inodes; require >= {} GiB AND {} inodes'.format(
        quota['free_gib'], quota['free_inodes'], MIN_FREE_GIB, MIN_FREE_INODES), flush=True)
    if quota['free_gib'] < MIN_FREE_GIB or quota['free_inodes'] < MIN_FREE_INODES:
        raise RuntimeError('Insufficient byte/inode reserve; no submission, no cleanup or guard bypass')
    print('WARNING: startup reserve only, not a full-batch capacity guarantee; SMAC runs retain '
          'periodic videos. Offline results and any concurrent uploads also consume this quota.', flush=True)


def probe_grf(map_name):
    """Server-only short simulator check; original rewards/features unchanged."""
    import tempfile
    import numpy as np
    import yaml
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from envs import REGISTRY
    if map_name not in REGISTRY:
        raise RuntimeError('GRF wrapper not registered; inspect gfootball/PySC2 import dependencies: '+map_name)
    config = yaml.safe_load((root/'src/config/envs'/(map_name+'.yaml')).read_text())
    spec = next(row for row in SCENES if row[1] == map_name and row[2] == 'grf')
    with tempfile.TemporaryDirectory(prefix='gomarl-grf-preflight-') as temp:
        kwargs = dict(config['env_args'], seed=1, write_video=False, render=False,
                      write_full_episode_dumps=False, write_goal_dumps=False, dump_freq=0, logdir=temp)
        env = REGISTRY[map_name](**kwargs)
        try:
            env.reset()
            info = env.get_env_info()
            expected = (spec[3], 19, spec[4], spec[4])
            actual = tuple(info[k] for k in ('n_agents', 'n_actions', 'obs_shape', 'state_shape'))
            if actual != expected:
                raise RuntimeError('Unexpected GRF dimensions: {} != {}'.format(actual, expected))
            for _ in range(2):
                state, obs = np.asarray(env.get_state()), np.asarray(env.get_obs())
                if state.shape != (spec[4],) or obs.shape != (spec[3], spec[4]):
                    raise RuntimeError('GRF state/obs contract mismatch')
                if not np.isfinite(state).all() or not np.isfinite(obs).all():
                    raise RuntimeError('Nonfinite GRF state/obs')
                reward, _, _ = env.step(np.zeros(spec[3], dtype=np.int64))
                if not np.isfinite(reward):
                    raise RuntimeError('Nonfinite GRF reward')
            print('GRF reset/step preflight passed: {} {}'.format(map_name, actual), flush=True)
        finally:
            env.close()


def validate_environments(domain='all'):
    scenes = selected_scenes(domain)
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    sc2 = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII'))
    if not any((sc2/'Versions').glob('Base*/SC2_x64')):
        raise RuntimeError('Missing installed SC2_x64 under '+str(sc2))
    for _, name, domain, allies, enemies in scenes:
        if domain != 'smac':
            continue
        params = registry.get(name, {})
        if (params.get('n_agents'), params.get('n_enemies')) != (allies, enemies):
            raise RuntimeError('Installed SMAC map definition mismatch: '+name)
        if not any((sc2/'Maps').rglob(name+'.SC2Map')):
            raise RuntimeError('Missing installed map file: '+name)
        print('SMAC assets preflight passed: '+name, flush=True)
    for _, name, domain, _, _ in scenes:
        if domain == 'grf':
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-grf', name],
                check=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=('smac', 'all'), default='smac',
                        help='Default smac: MMM2 and 3s5z_vs_3s6z only (12 jobs); all explicitly adds GRF (30 jobs)')
    parser.add_argument('--probe-grf', choices=[s[1] for s in SCENES if s[2] == 'grf'], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_grf:
        probe_grf(args.probe_grf)
        return
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], check=True, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(group_for(args.domain), __file__, lambda repo: build_plans(repo, args.domain), SMOKE_SCRIPTS,
                 map_validator=lambda: validate_environments(args.domain))


if __name__ == '__main__':
    main()
