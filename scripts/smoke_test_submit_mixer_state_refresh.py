#!/usr/bin/env python3
"""Offline nine-plan and fail-closed quota checks, no scheduler or W&B calls."""
from pathlib import Path
import shlex

from ozstar_submit_5m6m_mixer_state_refresh_9jobs import (
    GROUP, STAGE_STEPS, build_plans, reference_plans, require_capacity,
)
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(arg.split('=', 1) for arg in shlex.split(plan['exports']['EXTRA_ARGS']))


def main():
    root = Path(__file__).resolve().parents[1]
    reference = {p['seed']: p for p in reference_plans(root)}
    plans = build_plans(root)
    assert len(plans) == len({p['job_name'] for p in plans}) == 9
    for stage in STAGE_STEPS:
        assert {p['seed'] for p in plans if extras(p)['mixer_state_refresh_stage_steps'] == str(stage)} == {1, 2, 3}
    for plan in plans:
        source = reference[plan['seed']]
        actual = extras(plan)
        assert plan['scene'] == 'smac_5m6m' and plan['map_name'] == '5m_vs_6m'
        assert plan['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert plan['exports']['GROUP_NAME'] == GROUP
        assert '--time=1-00:00:00' in plan['sbatch_args']
        assert actual['learner'] == 'mixer_state_refresh_learner'
        assert actual['mixer_state_refresh_enabled'] == 'True'
        assert actual['mixer'] == 'qmix'
        changed = {'learner', 'mixer_state_refresh_enabled', 'mixer_state_refresh_stage_steps'}
        assert {k: v for k, v in actual.items() if k not in changed} == {
            k: v for k, v in extras(source).items() if k not in changed}
        for key in source['exports']:
            if key not in {'RUN_NAME', 'EXTRA_ARGS'}:
                assert plan['exports'][key] == source['exports'][key], key
        expected_args = [a.replace(source['job_name'], plan['job_name']) for a in source['sbatch_args']]
        expected_args = ['--time=1-00:00:00' if a.startswith('--time=') else a for a in expected_args]
        assert plan['sbatch_args'] == expected_args
    validate_config_keys(root, plans)
    for quota in (dict(free_gib=4.999, free_inodes=100000),
                  dict(free_gib=10, free_inodes=49999)):
        try:
            require_capacity(quota)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Byte/inode quota guard did not reject insufficient reserve')
    require_capacity(dict(free_gib=5, free_inodes=50000))
    print('PASS: 5m6m only, stage lengths 0.5M/1M/2M x seeds 1/2/3; '
          'historical model/training/media unchanged, 24h allocation, both quota floors')


if __name__ == '__main__':
    main()
