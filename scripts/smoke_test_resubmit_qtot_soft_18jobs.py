#!/usr/bin/env python3
"""Offline rerun-plan and quota safety checks; no Slurm/cloud/simulator calls."""
import copy
from pathlib import Path
from types import SimpleNamespace
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
    # Exact OzSTAR output reported on October 22: explanatory text must not
    # become extra quota columns, nor replace the authoritative raw integers.
    ozstar = '''Disk quotas for usr kyang (uid 12945):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
          /home 15845771       0 20969520       -  169467       0  200000       - (Disk usage is at 75%, 15.1 GiB of 20.0 GiB, Inode usage is at 84%)
Disk quotas for grp oz501 (gid 10855):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
          /fred 1112183737       0 10737418240       -  821663       0 1000000       - (Disk usage is at 10%, 1060.7 GiB of 10.0 TiB, Inode usage is at 82%)
'''
    current = rerun.parse_home_quota(ozstar)
    assert current['free_inodes'] == 30533
    assert current['free_gib'] == (20969520-15845771)/1024**2
    assert 4.88 < current['free_gib'] < 4.89
    assert current['used_kib'] == 15845771 and current['hard_inodes'] == 200000
    # Exact colored summary from the October 28 failed soft+refresh submission.
    colored = ('/home 18620972 0 20969520 - 165573 0 200000 - '
               '(\x1b[40;33;01mDisk usage is at 88%, 17.8 GiB of 20.0 GiB\x1b[0m, '
               'Inode usage is at 82%)')
    colored_quota = rerun.parse_home_quota(colored)
    assert colored_quota['free_gib'] == (20969520-18620972)/1024**2
    assert colored_quota['free_inodes'] == 34427
    assert rerun.parse_home_quota('\x1b[33m'+ozstar+'\x1b[0m') == current
    assert rerun.parse_home_quota(colored.replace('18620972', '\x1b[1m18620972\x1b[0m')) == colored_quota
    for bad in ('', '/home 1 0 2 -', '/home 1M 0 20G - 10 0 200000 -',
                '/home 1 0 0 - 10 0 200000 -', '/home 1 0 20000000 - 10 0 0 -',
                '/home 1 0 20000000 - 10 0 200000 -\n/home 1 0 20000000 - 10 0 200000 -',
                '/home 1 0 20000000 - 10 0 200000 - unexpected extra columns',
                '/home 1 0 20000000 - 10 0 200000 - (Unrecognized summary)',
                '/home 1 0 20000000 - 10 0 200000 - (\x1b[33mUnrecognized summary\x1b[0m)'):
        try:
            rerun.parse_home_quota(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unreadable quota was accepted: ' + bad)
    for low in (q, current, colored_quota, dict(free_gib=20, free_inodes=49999), dict(free_gib=4.99, free_inodes=100000)):
        try:
            rerun.require_capacity(low)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Low byte/inode headroom accepted')
    rerun.require_capacity(dict(free_gib=5, free_inodes=50000))
    # The actual failed submission path must still stop before submit_suite,
    # now because headroom is insufficient, not because parsing failed.
    with patch.dict(rerun.os.environ, SUBMIT='YES', REPO_DIR=str(repo)), \
            patch.object(rerun, 'validate_sources'), \
            patch.object(rerun.subprocess, 'run', return_value=SimpleNamespace(stdout=ozstar)), \
            patch.object(rerun, 'submit_suite') as submit:
        try:
            rerun.main()
        except RuntimeError as exc:
            assert 'Insufficient /home byte/inode headroom' in str(exc)
        else:
            raise AssertionError('Real insufficient quota reached submission')
        submit.assert_not_called()
    import ozstar_submit_soft_onefill_refresh_three_maps_18jobs as latest
    with patch.dict(rerun.os.environ, SUBMIT='YES', REPO_DIR=str(repo)), \
            patch.object(latest.subprocess, 'run', return_value=SimpleNamespace(stdout=colored)), \
            patch.object(latest, 'submit_suite') as submit:
        try:
            latest.main()
        except RuntimeError as exc:
            assert 'Insufficient byte/inode headroom' in str(exc)
        else:
            raise AssertionError('Colored insufficient quota reached latest submission')
        submit.assert_not_called()
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
