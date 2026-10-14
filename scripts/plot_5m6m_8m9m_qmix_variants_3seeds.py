#!/usr/bin/env python3
"""Two-map QMIX mixer variants and fresh VDN; no cross-map substitution."""
import sys

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as original_controls
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh_plans

LABELS = {
    'qmix': ('QMIX ordinary fixed head (fresh 10M)', '#7f7f7f'),
    'vdn': ('VDN ordinary fixed head (fresh 10M)', '#ff7f0e'),
    'qmix_state_static10': ('QMIX static state gate p0=0.1, no KL (10M)', '#bcbd22'),
    'qmix_state_group_qchange': ('QMIX Q-change groups / shared-gradient dual TD (10M)', '#08519c'),
    'qmix_state_group_tderror': ('QMIX TD-error groups / shared-gradient dual TD (10M)', '#a50f15'),
    'qmix_state_group_calibration_isolated': ('QMIX return-error groups / isolated auxiliary TD (10M)', '#2ca02c'),
}
SCENE_MODELS = {
    'smac_5m6m': tuple(LABELS),
    'smac_8m9m': ('qmix', 'vdn', 'qmix_state_static10'),
}
_CLOUD_QUERY = charts.cloud_run_index
_RENDER = charts.render
_PANELS = []


def render(scene, entries, output_dir, plotter, window):
    from wandb_seed_figure_sections import render_model_panels
    paths = _RENDER(scene, entries, output_dir, plotter, window)
    _PANELS.extend(render_model_panels(scene, entries, output_dir, plotter, window,
                                      LABELS, SCENE_MODELS[scene], charts.TARGET_STEPS))
    return paths


def upload(project, output_dir, outputs, inventory, window):
    from wandb_seed_figure_sections import upload_sectioned
    upload_sectioned(project, output_dir, outputs, inventory, window, charts, _PANELS)


def build_plans(repo):
    plans = original_controls(repo)
    plans.extend(p for p in fresh_plans(repo) if p['label'] in {'qmix', 'vdn'} or
                 p['label'] == 'qmix_state_group_calibration_isolated' or
                 (p['scene'] == 'smac_8m9m' and p['label'] == 'qmix_state_static10'))
    if len(plans) != 27 or len({p['job_name'] for p in plans}) != 27:
        raise RuntimeError('Expected 18 5m6m + 9 8m9m distinct experiment names')
    for scene, models in SCENE_MODELS.items():
        selected = [p for p in plans if p['scene'] == scene]
        if {p['label'] for p in selected} != set(models):
            raise RuntimeError('Unexpected models for ' + scene)
        for model in models:
            if {p['seed'] for p in selected if p['label'] == model} != {1, 2, 3}:
                raise RuntimeError('Invalid planned seed coverage for ' + model)
    return plans


def configure():
    _PANELS.clear()
    charts.GROUP = 'smac_5m6m_8m9m_qmix_mixer_variants_3seeds'
    charts.DEFAULT_OUTPUT_SUBDIR = '5m6m_8m9m_qmix_variants_3seeds'
    charts.RESULTS_TITLE = 'QMIX mixer-input variants and VDN'
    charts.REPORT_SEED_COVERAGE = True
    charts.TARGET_STEPS = 10000000
    charts.SCENES = {
        'smac_5m6m': ('5 Marines vs. 6 Marines', 'test_battle_won_mean'),
        'smac_8m9m': ('8 Marines vs. 9 Marines', 'test_battle_won_mean'),
    }
    charts.LABELS = LABELS
    charts.SCENE_MODELS = SCENE_MODELS
    charts.SCENE_FALLBACK_MODELS = {}
    charts.build_plans = build_plans
    charts.render = render
    charts.upload = upload
    charts.cloud_run_index = lambda project, wanted: _CLOUD_QUERY(project, wanted, allow_full_scan=False)


def main():
    configure()
    if not any(arg == '--mean-window' or arg.startswith('--mean-window=') for arg in sys.argv[1:]):
        sys.argv.extend(['--mean-window', '10'])
    charts.main()


if __name__ == '__main__':
    main()
