#!/usr/bin/env python3
"""Exact independent factors and combined Slurm plan; no actual submissions."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shlex
import tempfile
from unittest.mock import patch

import ozstar_submit_8m9m_ones_gate_regularization_9jobs as gates
import ozstar_submit_qchange_gate_ablation_18jobs as combined
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh
from ozstar_submit_5m6m_qchange_three_ablation_9jobs import build_plans as groups
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = combined.build_plans(repo)
    validate_config_keys(repo, plans)
    assert plans[:9] == groups(repo) and plans[9:] == gates.build_plans(repo)
    controls = {p['seed']: p for p in fresh(repo) if p['scene'] == 'smac_8m9m' and p['label'] == 'qmix_state_static10'}
    for plan in plans[9:]:
        new, original = extras(plan), extras(controls[plan['seed']])
        if plan['label'] == 'qmix_state_ones':
            assert new['mixer_input_control'] == 'ones'
            new['mixer_input_control'] = 'static_gate'
        elif plan['label'] == 'qmix_state_static10_l1':
            assert new.pop('mixer_gate_l1_coef') == '0.01'
            assert 'mixer_gate_variance_coef' not in new
        else:
            assert new.pop('mixer_gate_variance_coef') == '0.01'
            assert 'mixer_gate_l1_coef' not in new
        assert new == original
        assert plan['map_name'] == '8m_vs_9m' and int(plan['exports']['T_MAX']) == 10050000
        assert plan['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert new['slot_heatmap_interval'] == '100000'
    for scene in ('smac_5m6m', 'smac_8m9m'):
        selected = [p for p in plans if p['scene'] == scene]
        assert len(selected) == 9 and len({p['label'] for p in selected}) == 3
        for label in {p['label'] for p in selected}:
            assert {p['seed'] for p in selected if p['label'] == label} == {1, 2, 3}
    calls = []
    # Existing previous batch: combined command must submit only nine 8m9m jobs.
    retained = {p['job_name']: str(80000 + i) for i, p in enumerate(plans[:9])}
    def preflight(command, **kwargs):
        if command[0] == 'sbatch':
            assert command[1] == '--test-only' and 'submit' not in calls
            calls.append('preflight')
    def command(args, **kwargs):
        if args == ['id', '-un']:
            return 'SYNTHETIC_USER'
        if args == ['git', 'rev-parse', 'HEAD']:
            return 'SYNTHETIC_COMMIT'
        assert args[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 9
        calls.append('submit')
        return str(81000 + calls.count('submit'))
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch.object(gates, 'validate_maps'), \
                 patch.object(gates.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(gates.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(gates.submitter, 'active_jobs', return_value=retained), \
                 patch.object(gates.submitter, 'completed_jobs', return_value={}), \
                 patch.object(gates.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(gates.submitter, 'run', side_effect=command):
                combined.main()
                assert calls.count('preflight') == calls.count('submit') == 9
                import json
                record = json.loads(next(Path(temp).glob(combined.GROUP + '_*.json')).read_text())
                assert len(record['retained']) == 9 and len(record['submitted']) == 9
                retained.update(record['submitted'])
                calls.clear()
                combined.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: six model cells, 18 exact names; independent penalties; prior batch reuse; all preflights before submit; idempotence; media retained')


if __name__ == '__main__':
    main()
