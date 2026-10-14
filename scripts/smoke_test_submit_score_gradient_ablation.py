#!/usr/bin/env python3
"""Matched settings and fake Slurm checks; no actual submissions."""
from pathlib import Path
import shlex
import os
import tempfile
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import ozstar_submit_5m6m_score_gradient_ablation_6jobs as study
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as old_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(a.split('=', 1) for a in shlex.split(plan['exports']['EXTRA_ARGS']))


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == 6
    validate_config_keys(repo, plans)
    references = {('qmix_state_group_qchange_isolated', p['seed']): p
                  for p in old_plans(repo) if p['label'] == 'qmix_state_group_qchange'}
    references.update({('qmix_state_group_calibration_shared', p['seed']): p
                       for p in fresh_plans(repo) if p['label'] == 'qmix_state_group_calibration_isolated'})
    for p in plans:
        ref = references[p['label'], p['seed']]
        new, old = extras(p), extras(ref)
        isolated = p['label'].endswith('_isolated')
        assert new.pop('mixer_input_detach_unimportant') == str(isolated)
        old.pop('mixer_input_detach_unimportant', None)
        assert new == old  # No confounded input, optimizer, target or score changes.
        assert p['map_name'] == '5m_vs_6m' and int(p['exports']['T_MAX']) >= 10000000
        assert p['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert p['exports']['GROUP_NAME'] == study.GROUP
        assert p['exports']['RUN_NAME'] == p['job_name']
        assert '--job-name=' + p['job_name'] in p['sbatch_args']
    for label in {p['label'] for p in plans}:
        assert {p['seed'] for p in plans if p['label'] == label} == {1, 2, 3}
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
        assert calls.count('preflight') == 6 - len(retained)
        calls.append('submit')
        return str(99000 + calls.count('submit'))
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch('ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map'), \
                 patch.object(study.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(study.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(study.submitter, 'active_jobs', return_value=retained) as active, \
                 patch.object(study.submitter, 'completed_jobs', return_value={}), \
                 patch.object(study.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(study.submitter, 'run', side_effect=fake_run):
                study.main()
                assert calls.count('submit') == 6
                retained.update({p['job_name']: 'synthetic_active' for p in plans})
                active.return_value = retained
                calls.clear()
                study.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: six fresh matched runs; only gradient-routing difference; three seeds per cell; declared config; preflights before submit; idempotence')


if __name__ == '__main__':
    main()
