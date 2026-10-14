#!/usr/bin/env python3
"""Combined previous 5m6m ablations and new 8m9m controls, without duplicates."""
from ozstar_submit_5m6m_qchange_three_ablation_9jobs import build_plans as groups, SMOKE_SCRIPTS as GROUP_TESTS
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import build_plans as gates, SMOKE_SCRIPTS as GATE_TESTS, submit_suite

GROUP = 'qchange_gate_ablation_10m_18jobs_20261015'
SMOKE_SCRIPTS = tuple(dict.fromkeys(GROUP_TESTS + GATE_TESTS))


def build_plans(repo):
    plans = groups(repo) + gates(repo)
    if len(plans) != 18 or len({p['job_name'] for p in plans}) != 18:
        raise RuntimeError('Expected 18 distinct plans: three models per map, three seeds each')
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS)


if __name__ == '__main__':
    main()
