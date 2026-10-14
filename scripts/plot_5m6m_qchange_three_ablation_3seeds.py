#!/usr/bin/env python3
"""Only old Q-change/shared control and its three independent ablations."""
import sys

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as controls
from ozstar_submit_5m6m_qchange_three_ablation_9jobs import build_plans as ablations

LABELS = {
    'qmix_state_group_qchange': ('Control: Q-change / legacy mask-concat / shared TD', '#08519c'),
    'qmix_state_group_qchange_isolated': ('A1: auxiliary TD gradient cut only', '#17becf'),
    'qmix_state_group_qchange_plainmask': ('A2: original-dimension state x group-mask only', '#e377c2'),
    'qmix_state_group_variance': ('A3: Q_tot variance-difference score only', '#8c564b'),
}
MODELS = tuple(LABELS)
_QUERY, _RENDER = charts.cloud_run_index, charts.render
_PANELS = []


def build_plans(repo):
    plans = [p for p in controls(repo) if p['label'] == 'qmix_state_group_qchange'] + ablations(repo)
    if len(plans) != 12 or len({p['job_name'] for p in plans}) != 12:
        raise RuntimeError('Expected control plus three ablations, three distinct seeds each')
    return plans


def render(scene, entries, output, plotter, window):
    from wandb_seed_figure_sections import render_model_panels
    paths = _RENDER(scene, entries, output, plotter, window)
    _PANELS.extend(render_model_panels(scene, entries, output, plotter, window, LABELS, MODELS, charts.TARGET_STEPS))
    return paths


def upload(project, output, outputs, inventory, window):
    from wandb_seed_figure_sections import upload_sectioned
    upload_sectioned(project, output, outputs, inventory, window, charts, _PANELS)


def configure():
    _PANELS.clear()
    charts.GROUP = '5m6m_qchange_three_ablation_3seeds_20261015'
    charts.DEFAULT_OUTPUT_SUBDIR = '5m6m_qchange_three_ablation_3seeds'
    charts.RESULTS_TITLE = 'Q-change control: three single-factor ablations'
    charts.REPORT_SEED_COVERAGE = True
    charts.TARGET_STEPS = 10000000
    charts.SCENES = {'smac_5m6m': ('5 Marines vs. 6 Marines', 'test_battle_won_mean')}
    charts.LABELS = LABELS
    charts.SCENE_MODELS = {'smac_5m6m': MODELS}
    charts.SCENE_FALLBACK_MODELS = {}
    charts.build_plans, charts.render, charts.upload = build_plans, render, upload
    charts.cloud_run_index = lambda project, wanted: _QUERY(project, wanted, allow_full_scan=False)


def main():
    configure()
    if not any(arg == '--mean-window' or arg.startswith('--mean-window=') for arg in sys.argv[1:]):
        sys.argv.extend(['--mean-window', '10'])
    charts.main()


if __name__ == '__main__':
    main()
