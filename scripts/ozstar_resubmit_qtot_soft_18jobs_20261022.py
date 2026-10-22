#!/usr/bin/env python3
"""Fresh rerun of the 18 quota-failed October 21 jobs; no resume/cancellation.

Model/seed/training/media/resource settings are copied unchanged. Only run and
batch identities change. Plan-only by default; SUBMIT=YES validates the failed
source IDs, user byte/inode quotas, maps, smoke tests and Slurm before submission.
The capacity floors are startup checks, NOT full-lifecycle space guarantees.
"""
import copy
from pathlib import Path
import re
import os
import subprocess

import ozstar_submit_qtot_and_onefill_15jobs as qtot
import ozstar_submit_5m6m_soft_onefill_3seeds as soft
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = 'qtot_soft_fresh_rerun_10m_18jobs_20261022'
SOURCE_JOBS = tuple(range(18944777, 18944792)) + tuple(range(18944951, 18944954))
MIN_FREE_GIB = 5.0
MIN_FREE_INODES = 50000
SMOKE_SCRIPTS = tuple(dict.fromkeys(qtot.SMOKE_SCRIPTS + soft.SMOKE_SCRIPTS +
    ('scripts/smoke_test_resubmit_qtot_soft_18jobs.py',)))


def source_plans(repo):
    plans = qtot.build_plans(repo) + soft.build_plans(repo)
    if len(plans) != 18 or len({p['job_name'] for p in plans}) != 18:
        raise RuntimeError('Expected exactly the 18 October 21 source experiments')
    return plans


def build_plans(repo):
    plans = []
    for source, job in zip(source_plans(repo), SOURCE_JOBS):
        plan = copy.deepcopy(source)
        name = '{}_{}_10m_s{}_rerun20261022'.format(source['scene'], source['label'], source['seed'])
        rename(plan, name)
        plan['exports']['GROUP_NAME'] = GROUP
        plan['rerun_of'] = dict(job_id=str(job), run_name=source['job_name'],
                                group=source['exports']['GROUP_NAME'])
        plan['inventory_note'] = (source['inventory_note'] +
            '; fresh from step zero after quota failure; source job {}; no resume/history stitching'.format(job))
        plans.append(plan)
    if len({p['job_name'] for p in plans}) != 18:
        raise RuntimeError('Rerun identities are not unique')
    return plans


def parse_home_quota(output):
    """Parse quota -v (raw KiB/counts), never df or human-rounded quota -s."""
    rows = [line.split() for line in output.splitlines() if line.split()[:1] == ['/home']]
    if len(rows) != 1 or len(rows[0]) not in (7, 8, 9):
        raise RuntimeError('Cannot identify one complete /home byte+inode quota row; no submission')
    fields = rows[0]
    def integer(token):
        if not re.fullmatch(r'[0-9]+\*?', token):
            raise RuntimeError('Non-numeric raw quota field: ' + token)
        return int(token.rstrip('*'))
    # Grace columns may be absent/blank or '-' / '7days'. Do not mistake a
    # missing byte grace column for an inode-count column shift.
    tail = fields[4:]
    if tail and not re.fullmatch(r'[0-9]+\*?', tail[0]):
        tail = tail[1:]
    if len(tail) not in (3, 4) or (len(tail) == 4 and re.fullmatch(r'[0-9]+\*?', tail[3])):
        raise RuntimeError('Ambiguous inode/grace quota columns; no submission')
    integer(fields[2])  # Verify soft limits too, despite enforcing hard limits.
    integer(tail[1])
    used_kib, hard_kib, used_files, hard_files = map(integer,
        (fields[1], fields[3], tail[0], tail[2]))
    if hard_kib <= 0 or hard_files <= 0:
        raise RuntimeError('Missing positive /home byte/inode hard limits; no submission')
    return dict(free_gib=(hard_kib-used_kib)/1024**2, free_inodes=hard_files-used_files,
                used_kib=used_kib, hard_kib=hard_kib, used_inodes=used_files, hard_inodes=hard_files)


def require_capacity(quota):
    print('/home free: {:.3f} GiB, {} inodes; require >= {:.1f} GiB AND {} inodes'.format(
        quota['free_gib'], quota['free_inodes'], MIN_FREE_GIB, MIN_FREE_INODES), flush=True)
    if quota['free_gib'] < MIN_FREE_GIB or quota['free_inodes'] < MIN_FREE_INODES:
        raise RuntimeError('Insufficient /home byte/inode headroom; no jobs submitted. '
                           'Do not bypass this guard or silently disable requested media.')
    print('Startup floor only: 18 full 10M media-enabled runs can grow beyond this reserve; '
          'other jobs/uploads also consume the same quota.', flush=True)


def validate_sources(repo):
    from ozstar_calibration_suite_sync import accounting, manifest_jobs, TERMINAL
    import ozstar_calibration_suite_sync as sync
    runtime = Path(os.environ.get('RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch'))
    previous = sync.GROUPS
    try:
        sync.GROUPS = (qtot.GROUP, soft.GROUP)
        recorded = manifest_jobs(runtime / 'ozstar_logs')
    finally:
        sync.GROUPS = previous
    expected = {str(job): plan['job_name'] for job, plan in zip(SOURCE_JOBS, source_plans(repo))}
    if any(recorded.get(job) != name for job, name in expected.items()):
        raise RuntimeError('Original receipt IDs/names differ; no guessed rerun submission')
    states = accounting(expected, repo)
    failed_states = TERMINAL - {'COMPLETED'}
    if set(states) != set(expected) or any(row['state'] not in failed_states for row in states.values()):
        raise RuntimeError('Source jobs are not all confirmed unsuccessful terminal jobs; inspect before rerunning')


def main():
    if os.environ.get('SUBMIT') == 'YES':
        repo = Path(os.environ.get('REPO_DIR', '/home/kyang/code/gomarl-dual-branch')).resolve()
        validate_sources(repo)
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True, timeout=30)
        require_capacity(parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=qtot.validate_maps)


if __name__ == '__main__':
    main()
