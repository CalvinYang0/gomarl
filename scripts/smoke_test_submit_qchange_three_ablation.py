#!/usr/bin/env python3
"""Single-factor plan checks and fake Slurm idempotence; no real submissions."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shlex
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_qchange_three_ablation_9jobs as study
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as originals
from ozstar_submit_5m6m_score_gradient_ablation_6jobs import build_plans as existing
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(p):
    return dict(v.split('=', 1) for v in shlex.split(p['exports']['EXTRA_ARGS']))


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    bases = {p['seed']: p for p in originals(repo) if p['label'] == 'qmix_state_group_qchange'}
    reused = {p['job_name']: p for p in existing(repo) if p['label'] == 'qmix_state_group_qchange_isolated'}
    assert len(plans) == 9 and len(reused) == 3
    for p in plans:
        new, old = extras(p), extras(bases[p['seed']])
        if p['label'] == 'qmix_state_group_qchange_isolated':
            assert p == reused[p['job_name']]
            assert new.pop('mixer_input_detach_unimportant') == 'True'
            old.pop('mixer_input_detach_unimportant', None)
        elif p['label'] == 'qmix_state_group_qchange_plainmask':
            assert new.pop('mixer_input_append_mask') == 'False'
        else:
            assert p['label'] == 'qmix_state_group_variance'
            assert new['mixer_input_control'] == 'group_variance'
            new['mixer_input_control'] = 'group_qchange'
        assert new == old, 'A supposedly single-factor ablation changed other training/probe settings'
        assert p['map_name'] == '5m_vs_6m' and int(p['exports']['T_MAX']) == 10050000
        assert p['exports']['MODEL_TYPE'] == 'qmix_minimal' and new['slot_heatmap_interval'] == '100000'
        assert new['slot_group_interval'] == '1000'
    for label in {p['label'] for p in plans}:
        assert {p['seed'] for p in plans if p['label'] == label} == {1, 2, 3}
    calls, retained = [], {}
    def preflight(command, **kwargs):
        if command[0] == 'sbatch':
            assert command[1] == '--test-only' and 'submit' not in calls
            calls.append('preflight')
    def fake_command(command, **kwargs):
        if command == ['id', '-un']:
            return 'SYNTHETIC_USER'
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'SYNTHETIC_COMMIT'
        assert command[:2] == ['sbatch', '--parsable']
        assert calls.count('preflight') == 9 - len(retained)
        calls.append('submit')
        return str(99500 + calls.count('submit'))
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch('ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map'), \
                 patch.object(study.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(study.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(study.submitter, 'active_jobs', return_value=retained), \
                 patch.object(study.submitter, 'completed_jobs', return_value={}), \
                 patch.object(study.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(study.submitter, 'run', side_effect=fake_command):
                study.main()
                assert calls.count('submit') == 9
                manifest = next(Path(temp).glob(study.GROUP + '_*.json'))
                import json
                retained.update(json.loads(manifest.read_text())['submitted'])
                calls.clear()
                study.main()
                assert not calls
                # A1 already running/completed: add only the six genuinely new
                # encoding/statistic jobs, not another gradient-cut cohort.
                retained_copy = dict(retained)
                retained.clear()
                retained.update({name: job for name, job in retained_copy.items() if name in reused})
                study.main()
                assert calls.count('preflight') == calls.count('submit') == 6
        finally:
            os.chdir(cwd)
    print('PASS: exactly three independent factors; three seeds each; old gradient-only identities reused; preflights before submit; idempotence; media retained')


if __name__ == '__main__':
    main()
