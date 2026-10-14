#!/usr/bin/env python3
"""Actual 8m9m QMIX/static-gate cohorts only; never import 5m6m groups."""
import sys

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as suite_plans

LABELS = {
    'qmix': ('QMIX ordinary fixed head (fresh 10M)', '#7f7f7f'),
    'qmix_state_static10': ('QMIX static state gate p0=0.1, no KL (10M)', '#bcbd22'),
}
_CLOUD_QUERY = charts.cloud_run_index


def build_plans(repo):
    plans = [p for p in suite_plans(repo) if p['scene'] == 'smac_8m9m' and p['label'] in LABELS]
    if len(plans) != 6 or len({p['job_name'] for p in plans}) != 6:
        raise RuntimeError('Expected exactly six actual 8m9m QMIX/gate runs')
    for label in LABELS:
        if {p['seed'] for p in plans if p['label'] == label} != {1, 2, 3}:
            raise RuntimeError('Missing planned seeds for ' + label)
    return plans


def configure():
    charts.GROUP = 'smac_8m9m_qmix_variants_fresh20261014_3seeds'
    charts.DEFAULT_OUTPUT_SUBDIR = '8m9m_qmix_variants_3seeds'
    charts.RESULTS_TITLE = 'QMIX and static mixer-state gate; fresh cohort'
    charts.REPORT_SEED_COVERAGE = True
    charts.TARGET_STEPS = 10000000
    charts.SCENES = {'smac_8m9m': ('8 Marines vs. 9 Marines — QMIX and mixer-input variants', 'test_battle_won_mean')}
    charts.LABELS = LABELS
    charts.SCENE_MODELS = {'smac_8m9m': tuple(LABELS)}
    charts.SCENE_FALLBACK_MODELS = {}
    charts.build_plans = build_plans
    charts.cloud_run_index = lambda project, wanted: _CLOUD_QUERY(project, wanted, allow_full_scan=False)


def main():
    configure()
    if not any(arg == '--mean-window' or arg.startswith('--mean-window=') for arg in sys.argv[1:]):
        sys.argv.extend(['--mean-window', '10'])
    charts.main()


if __name__ == '__main__':
    main()
