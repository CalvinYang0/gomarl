#!/usr/bin/env python3
"""Fresh 10M: two-map baselines, 5m6m return-calibration groups, 8m9m gate."""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_fixed_head_plans
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as control_plans

GROUP = "qmix_vdn_calibration_gate_10m_3seeds_fresh20261014"
SMOKE_SCRIPTS = ("scripts/smoke_test_mixer_input_controls.py",
                 "scripts/smoke_test_mixer_calibration.py",
                 "scripts/smoke_test_submit_calibration_gate.py")


def rename(plan, name):
    old = plan['job_name']
    plan.update(job_name=name, run_name=name, target_steps=10000000,
                inventory_note="Fresh October 14 cohort; never merged with historical runs")
    plan['exports']['RUN_NAME'] = name
    plan['sbatch_args'] = [arg.replace(old, name) for arg in plan['sbatch_args']]
    return plan


def build_plans(repo):
    controls = {p['seed']: p for p in control_plans(repo) if p['label'] == 'qmix_state_static10'}
    plans = []
    for scene, map_name in (('smac_5m6m', '5m_vs_6m'), ('smac_8m9m', '8m_vs_9m')):
        bases = build_fixed_head_plans(repo, scene, map_name, GROUP)
        for plan in bases:
            name = '{}_{}_baseline_10m_s{}_fresh20261014'.format(scene, plan['label'], plan['seed'])
            plans.append(rename(plan, name))
        # New ordinary-agent controls use exactly the same matched rollout and
        # optimizer configuration as their map's fresh QMIX baseline.
        for plan in build_fixed_head_plans(repo, scene, map_name, GROUP):
            if plan['label'] != 'qmix':
                continue
            grouped = scene == 'smac_5m6m'
            label = 'qmix_state_group_calibration' if grouped else 'qmix_state_static10'
            plan['label'] = label
            extra = controls[plan['seed']]['exports']['EXTRA_ARGS']
            if grouped:
                extra = extra.replace('mixer_input_control=static_gate', 'mixer_input_control=group_calibration')
            plan['exports']['EXTRA_ARGS'] = extra
            name = '{}_{}_10m_s{}_fresh20261014'.format(scene, label, plan['seed'])
            plans.append(rename(plan, name))
    if len(plans) != 18 or len({p['job_name'] for p in plans}) != 18:
        raise RuntimeError('Expected 18 distinct fresh experiment names')
    return plans


def validate_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    maps = get_smac_map_registry()
    root = Path(os.environ.get('SC2PATH', '/home/kyang/StarCraftII')) / 'Maps'
    for name, counts in (('5m_vs_6m', (5, 6)), ('8m_vs_9m', (8, 9))):
        params = maps.get(name, {})
        if (params.get('n_agents'), params.get('n_enemies')) != counts or params.get('map_type') != 'marines':
            raise RuntimeError('Unexpected installed map: ' + name)
        if not any(root.rglob(name + '.SC2Map')):
            raise RuntimeError('Missing map file: ' + name)


def main():
    if os.environ.get('SUBMIT') == 'YES':
        free = submitter.home_quota_free_gib()
        if free < 2:
            raise RuntimeError('Require at least 2 GiB free; no jobs submitted')
        print('Capacity warning: 18 media-enabled jobs; 2 GiB is not a capacity guarantee.', flush=True)
        validate_maps()
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__),
                       SMOKE_SCRIPTS=SMOKE_SCRIPTS, build_plans=build_plans)
    original = {key: getattr(submitter, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(submitter, key, value)
        submitter.main()
    finally:
        for key, value in original.items():
            setattr(submitter, key, value)


if __name__ == '__main__':
    main()
