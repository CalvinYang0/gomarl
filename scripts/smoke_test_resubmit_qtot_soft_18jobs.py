#!/usr/bin/env python3
"""Offline rerun-plan and quota safety checks; no Slurm/cloud/simulator calls."""
import copy
from pathlib import Path
from unittest.mock import patch

import ozstar_resubmit_qtot_soft_18jobs_20261022 as rerun
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import guard_and_route, validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    sources, plans = rerun.source_plans(repo), rerun.build_plans(repo)
    assert len(plans) == len({p['job_name'] for p in plans}) == 18
    coverage = {}
    for old, new, job in zip(sources, plans, rerun.SOURCE_JOBS):
        coverage.setdefault((new['scene'], new['label']), set()).add(new['seed'])
        assert new['job_name'] == new['run_name'] == new['exports']['RUN_NAME']
        assert new['job_name'].endswith('_rerun20261022')
        assert new['rerun_of'] == dict(job_id=str(job), run_name=old['job_name'], group=old['exports']['GROUP_NAME'])
        assert new['exports']['GROUP_NAME'] == rerun.GROUP
        for key in set(old['exports']) | set(new['exports']):
            if key not in {'RUN_NAME', 'GROUP_NAME'}:
                assert old['exports'].get(key) == new['exports'].get(key), key
        assert new['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert new['sbatch_args'] == [arg.replace(old['job_name'], new['job_name']) for arg in old['sbatch_args']]
        assert '--time=1-00:00:00' in new['sbatch_args']
        assert old['target_steps'] == new['target_steps'] == 10000000
    assert len(coverage) == 6 and all(seeds == {1, 2, 3} for seeds in coverage.values())
    assert [sum(p['scene'] == scene for p in plans) for scene in
            ('smac_5m6m', 'smac_mmm2', 'smac_corridor')] == [12, 3, 3]
    routed = copy.deepcopy(plans)
    guard_and_route(routed, Path('/home/kyang/gomarl-runtime/gomarl-dual-branch'))
    validate_config_keys(repo, routed)
    assert sources == rerun.source_plans(repo)  # Never mutate original submitters.
    q = rerun.parse_home_quota(' /home 19171099 0 20969520 - 199488 0 200000 -\n')
    assert q['free_inodes'] == 512 and 1 < q['free_gib'] < 2
    for row in ('/home 19171099 0 20969520 199488 0 200000',
                '/home 19171099 0 20969520 - 199488 0 200000',
                '/home 19171099 0 20969520 199488 0 200000 -',
                '/home 19171099* 0 20969520 7days 199488 0 200000 -'):
        assert rerun.parse_home_quota(row) == q
    for bad in ('', '/home 1 0 2 -', '/home 1M 0 20G - 10 0 200000 -',
                '/home 1 0 0 - 10 0 200000 -', '/home 1 0 20000000 - 10 0 0 -',
                '/home 1 0 20000000 - 10 0 200000 -\n/home 1 0 20000000 - 10 0 200000 -'):
        try:
            rerun.parse_home_quota(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unreadable quota was accepted: ' + bad)
    for low in (q, dict(free_gib=20, free_inodes=49999), dict(free_gib=4.99, free_inodes=100000)):
        try:
            rerun.require_capacity(low)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Low byte/inode headroom accepted')
    rerun.require_capacity(dict(free_gib=5, free_inodes=50000))
    import ozstar_calibration_suite_sync as sync
    expected = {str(job): p['job_name'] for job, p in zip(rerun.SOURCE_JOBS, sources)}
    previous = sync.GROUPS
    with patch.object(sync, 'manifest_jobs', return_value=expected), patch.object(sync, 'accounting',
            return_value={job: dict(state='FAILED') for job in expected}):
        rerun.validate_sources(repo)
    assert sync.GROUPS == previous
    for bad_states in ({}, {job: dict(state='COMPLETED') for job in expected},
                       {job: dict(state='RUNNING') for job in expected}):
        with patch.object(sync, 'manifest_jobs', return_value=expected), patch.object(sync, 'accounting', return_value=bad_states):
            try:
                rerun.validate_sources(repo)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unconfirmed/active/completed source was accepted')
        assert sync.GROUPS == previous
    with patch.object(sync, 'manifest_jobs', return_value={}), patch.object(sync, 'accounting') as accounting:
        try:
            rerun.validate_sources(repo)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing receipt accepted')
        accounting.assert_not_called()
    print('PASS: 18 fresh 24h plans, exact old settings/heads, 12+3+3 coverage; fail-closed byte/inode and source-state guards')


if __name__ == '__main__':
    main()
