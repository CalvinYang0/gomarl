#!/usr/bin/env python3
"""Two-map QMIX mixer variants and fresh VDN; no cross-map substitution."""
import sys

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as original_controls
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh_plans
from ozstar_submit_5m6m_score_gradient_ablation_6jobs import build_plans as ablation_plans
from ozstar_submit_5m6m_qchange_three_ablation_9jobs import build_plans as three_ablation_plans, NEW_LABELS
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import build_plans as gate_plans, LABELS as GATE_LABELS
from ozstar_submit_5m6m_ones_3seeds import build_plans as ones_5m6m_plans
from ozstar_submit_5m6m_calibration_mean_shared_3seeds import build_plans as calibration_mean_plans
from ozstar_submit_calibration_mean_ablation_12jobs import build_plans as calibration_ablation_plans, LABELS as CALIBRATION_ABLATION_LABELS

LABELS = {
    'qmix': ('QMIX ordinary fixed head (fresh 10M)', '#7f7f7f'),
    'vdn': ('VDN ordinary fixed head (fresh 10M)', '#ff7f0e'),
    'qmix_state_static10': ('QMIX static state gate p0=0.1, no KL (10M)', '#bcbd22'),
    'qmix_state_group_qchange': ('QMIX Q-change groups / shared-gradient dual TD (10M)', '#08519c'),
    'qmix_state_group_tderror': ('QMIX TD-error groups / shared-gradient dual TD (10M)', '#a50f15'),
    'qmix_state_group_calibration_isolated': ('QMIX return-error groups / isolated auxiliary TD (10M)', '#2ca02c'),
    'qmix_state_group_qchange_isolated': ('QMIX Q-change groups / isolated auxiliary TD (10M)', '#17becf'),
    'qmix_state_group_calibration_shared': ('QMIX return-error groups / shared-gradient dual TD (10M)', '#9467bd'),
    'qmix_state_group_calibration_mean_shared': ('QMIX MC-error ranking / batch-mean replacement / shared TD (10M)', '#e6550d'),
    'qmix_state_group_calibration_mean_isolated': ('QMIX MC-error / batch-mean / isolated auxiliary TD (10M)', '#31a354'),
    'qmix_state_group_calibration_mean_plainmask': ('QMIX MC-error / batch-mean / state-only mask / shared TD (10M)', '#756bb1'),
    'qmix_state_group_qchange_plainmask': ('QMIX Q-change groups / state-only mask / shared TD (10M)', '#e377c2'),
    'qmix_state_group_variance': ('QMIX variance-difference groups / shared TD (10M)', '#8c564b'),
    'qmix_state_ones': ('QMIX all-ones mixer condition (10M)', '#393b79'),
    'qmix_state_static10_l1': ('QMIX static gate + probability L1 (10M)', '#e377c2'),
    'qmix_state_static10_variance': ('QMIX static gate + variance-weighted probability loss (10M)', '#8c564b'),
}
SCENE_MODELS = {
    'smac_5m6m': tuple(k for k in LABELS if k not in GATE_LABELS or k == 'qmix_state_ones'),
    'smac_8m9m': ('qmix', 'vdn', 'qmix_state_static10') + GATE_LABELS + CALIBRATION_ABLATION_LABELS,
}
_CLOUD_QUERY = charts.cloud_run_index
_RENDER = charts.render
_PANELS = []


def render(scene, entries, output_dir, plotter, window):
    from wandb_seed_figure_sections import render_model_panels
    paths = _RENDER(scene, entries, output_dir, plotter, window)
    models = tuple(model for model in SCENE_MODELS[scene] if model in entries)
    _PANELS.extend(render_model_panels(scene, entries, output_dir, plotter, window,
                                      LABELS, models, charts.TARGET_STEPS))
    return paths


def upload(project, output_dir, outputs, inventory, window):
    from wandb_seed_figure_sections import upload_sectioned
    upload_sectioned(project, output_dir, outputs, inventory, window, charts, _PANELS)


def build_plans(repo):
    plans = original_controls(repo)
    plans.extend(ablation_plans(repo))
    plans.extend(p for p in three_ablation_plans(repo) if p['label'] in NEW_LABELS)
    plans.extend(gate_plans(repo))
    plans.extend(ones_5m6m_plans(repo))
    plans.extend(calibration_mean_plans(repo))
    plans.extend(calibration_ablation_plans(repo))
    plans.extend(p for p in fresh_plans(repo) if p['label'] in {'qmix', 'vdn'} or
                 p['label'] == 'qmix_state_group_calibration_isolated' or
                 (p['scene'] == 'smac_8m9m' and p['label'] == 'qmix_state_static10'))
    if len(plans) != 66 or len({p['job_name'] for p in plans}) != 66:
        raise RuntimeError('Expected 42 5m6m + 24 8m9m distinct experiment names')
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
