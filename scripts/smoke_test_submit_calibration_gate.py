#!/usr/bin/env python3
"""New-cohort shape/config and fake Slurm submission checks."""
from pathlib import Path
import shlex
import os
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import ozstar_submit_qmix_vdn_calibration_gate_18jobs as study
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans, GROUP
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    plans = build_plans(Path(__file__).resolve().parents[1])
    validate_config_keys(Path(__file__).resolve().parents[1], plans)
    assert len(plans) == 18
    assert len({p['job_name'] for p in plans}) == 18
    for map_name in ('5m_vs_6m', '8m_vs_9m'):
        subset = [p for p in plans if p['map_name'] == map_name]
        assert len(subset) == 9
        for label in ('vdn', 'qmix'):
            assert {p['seed'] for p in subset if p['label'] == label} == {1, 2, 3}
    for plan in plans:
        assert plan['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert int(plan['exports']['T_MAX']) >= 10000000
        assert 'fresh20261014' in plan['job_name']
        assert plan['exports']['RUN_NAME'] == plan['job_name']
        assert plan['exports']['GROUP_NAME'] == GROUP
        extra = dict(item.split('=', 1) for item in shlex.split(plan['exports']['EXTRA_ARGS']))
        assert extra['test_value_diagnostics'] == 'True'
        if plan['label'] == 'qmix_state_group_calibration':
            assert plan['map_name'] == '5m_vs_6m'
            assert extra['mixer_input_control'] == 'group_calibration'
            assert extra['slot_group_interval'] == '1000'
        elif plan['label'] == 'qmix_state_static10':
            assert plan['map_name'] == '8m_vs_9m'
            assert extra['mixer_input_control'] == 'static_gate'
            assert extra['mixer_input_initial_probability'] == '0.1'
        else:
            assert extra['mixer'] == plan['label']
    calls, retained = [], {}
    def preflight(command, **kwargs):
        if command[0] == 'sbatch':
            assert command[1] == '--test-only' and 'submit' not in calls
            calls.append('preflight')
    def fake_run(command, **kwargs):
        if command == ['id', '-un']:
            return 'synthetic_user'
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'synthetic_commit'
        assert command[:2] == ['sbatch', '--parsable']
        assert calls.count('preflight') == 18 - len(retained)
        calls.append('submit')
        return str(99000 + calls.count('submit'))
    original = study.submitter.build_plans
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        try:
            with redirect_stdout(io.StringIO()), \
                 patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(cwd), RUNTIME_ROOT=temp), \
                 patch.object(study, 'validate_maps'), \
                 patch.object(study.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(study.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(study.submitter, 'active_jobs', return_value=retained) as active, \
                 patch.object(study.submitter, 'completed_jobs', return_value={}), \
                 patch.object(study.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(study.submitter, 'run', side_effect=fake_run):
                study.main()
                assert calls.count('submit') == 18
                retained.update({p['job_name']: 'synthetic_active' for p in plans})
                active.return_value = retained
                calls.clear()
                study.main()
                assert not calls
                with patch.object(study.submitter, 'home_quota_free_gib', return_value=1.99):
                    try:
                        study.main()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Quota guard failed')
                assert not calls
        finally:
            os.chdir(cwd)
    assert study.submitter.build_plans is original
    print('PASS: 18 fresh names; matched map/seed/10M settings; ordinary heads; diagnostics')
    print('PASS: all preflights before submission; active-run idempotence; quota guard; no actual Slurm calls')


if __name__ == '__main__':
    main()
