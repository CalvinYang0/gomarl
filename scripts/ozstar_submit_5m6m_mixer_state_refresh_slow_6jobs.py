#!/usr/bin/env python3
"""5m6m historical QMIX: 5M/10M refresh stages x three seeds, 10M total/24h.

Plan-only unless SUBMIT=YES. Fresh distinct cohort, no old jobs resubmitted,
no head/model/TD changes and no cancellation, cleanup or resume.
"""
import os
from pathlib import Path
import subprocess

import ozstar_submit_5m6m_mixer_state_refresh_9jobs as previous
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = '5m6m_mixer_state_refresh_slow_10m_6jobs_20261026'
STAGE_STEPS = (5000000, 10000000)
SMOKE_SCRIPTS = ('scripts/smoke_test_submit_slow_mixer_state_refresh.py',
                 'scripts/smoke_test_mixer_state_refresh.py')


def build_plans(repo):
    return previous.build_plans(repo, stage_steps=STAGE_STEPS, group=GROUP,
                                suffix='refresh20261026')


def main():
    if os.environ.get('SUBMIT') == 'YES':
        quota = subprocess.run(['quota', '-v'], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True, timeout=30)
        previous.require_capacity(previous.parse_home_quota(quota.stdout))
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS,
                 map_validator=previous.validate_installed_map)


if __name__ == '__main__':
    main()
