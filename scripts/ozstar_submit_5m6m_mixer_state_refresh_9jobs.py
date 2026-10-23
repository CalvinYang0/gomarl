#!/usr/bin/env python3
"""Historical QMIX; maximum-episode-length refresh curriculum, 3 stages x 3 seeds.

Plan-only unless SUBMIT=YES. No actor/head, gate, TD loss, cancellation or resume
changes. Byte AND inode quota checks fail closed before any job submission.
"""
import copy
import os
from pathlib import Path
import shlex
import subprocess

from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_fixed_head_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
from ozstar_resubmit_qtot_soft_18jobs_20261022 import parse_home_quota

GROUP = '5m6m_mixer_state_refresh_10m_9jobs_20261023'
STAGE_STEPS = (500000, 1000000, 2000000)
MIN_FREE_GIB = 5.0
MIN_FREE_INODES = 50000
SMOKE_SCRIPTS = ('scripts/smoke_test_submit_mixer_state_refresh.py',
                 'scripts/smoke_test_mixer_state_refresh.py')


def reference_plans(repo):
    return [p for p in build_fixed_head_plans(repo, 'smac_5m6m', '5m_vs_6m', GROUP)
            if p['label'] == 'qmix']


def build_plans(repo):
    plans = []
    for original in reference_plans(repo):
        for stage in STAGE_STEPS:
            plan = copy.deepcopy(original)
            label = 'qmix_state_refresh_{}k'.format(stage // 1000)
            name = 'smac_5m6m_{}_10m_s{}_refresh20261023'.format(label, plan['seed'])
            rename(plan, name)
            extra = dict(arg.split('=', 1) for arg in shlex.split(plan['exports']['EXTRA_ARGS']))
            extra.update(learner='mixer_state_refresh_learner',
                         mixer_state_refresh_enabled='True',
                         mixer_state_refresh_stage_steps=str(stage))
            plan['exports']['EXTRA_ARGS'] = ' '.join(k+'='+v for k, v in extra.items())
            plan['label'] = label
            plan['sbatch_args'] = ['--time=1-00:00:00' if a.startswith('--time=') else a
                                  for a in plan['sbatch_args']]
            plan['inventory_note'] = (
                'Historical qmix_minimal two-layer ELU action head, plain QMixer and unchanged TD; '
                'mixer global state only; K=1+floor(collection_t_env/{}) fixed at episode reset; '
                'boundaries=floor(j*episode_limit/K); replay/target use collected held state; '
                'actor obs live; no gate/aux loss/forced final frequency; 24h allocation'.format(stage))
            plans.append(plan)
    if len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 9:
        raise RuntimeError('Expected exactly nine unique 5m6m refresh jobs')
    return plans


def require_capacity(quota):
    print('/home free: {:.3f} GiB, {} inodes; require >= {} GiB AND {} inodes'.format(
        quota['free_gib'], quota['free_inodes'], MIN_FREE_GIB, MIN_FREE_INODES), flush=True)
    if quota['free_gib'] < MIN_FREE_GIB or quota['free_inodes'] < MIN_FREE_INODES:
        raise RuntimeError('Insufficient byte/inode headroom; no jobs submitted. Do not bypass the guard.')
    print('Startup reserve only, not a nine-run lifetime capacity guarantee; '
          'existing periodic videos and diagnostics remain enabled.', flush=True)


def main():
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_installed_map)


if __name__ == '__main__':
    main()
