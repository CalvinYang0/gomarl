#!/usr/bin/env python3
"""Corridor baseline plan/model/mock scheduler QA, no real jobs or simulator."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import torch as th
import yaml

import ozstar_submit_corridor_qmix_3jobs as suite
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from modules.mixers.qmix import QMixer

ROOT = Path(__file__).resolve().parents[1]


def model_checks():
    plans = suite.build_plans(ROOT)
    old = [p for p in suite.old_plans(ROOT) if p['label'] == 'qmix']
    assert len(plans) == 3 and {p['seed'] for p in plans} == {1, 2, 3}
    assert [p['job_name'] for p in plans] == [p['job_name'] for p in old]
    validate_config_keys(ROOT, plans)
    th.set_num_threads(1)
    for plan, previous in zip(plans, old):
        exports = plan['exports']
        assert plan['map_name'] == 'corridor' and plan['scene'] == 'smac_corridor'
        assert exports['GROUP_NAME'] == suite.GROUP
        assert exports['MODEL_TYPE'] == 'qmix_minimal' and exports['T_MAX'] == '10050000'
        assert '--time=1-00:00:00' in plan['sbatch_args'] and plan['memory'] == '24G'
        flags = dict(x.split('=', 1) for x in shlex.split(exports['EXTRA_ARGS']))
        old_flags = dict(x.split('=', 1) for x in shlex.split(previous['exports']['EXTRA_ARGS']))
        assert all(flags[k] == v for k, v in old_flags.items())
        assert flags['qmix_condition_input'] == 'state'
        assert flags['mixer_state_refresh_enabled'] == 'False'
        overrides = {k: yaml.safe_load(v) for k, v in flags.items() if '.' not in k}
        overrides['clean_model_type'] = 'qmix_minimal'
        th.manual_seed(plan['seed'])
        mac, learner, batch, _ = make_case('baseline', 'corridor', production_shapes=True,
                                         config_overrides=overrides)
        assert isinstance(learner.mixer, QMixer) and learner.mixer.condition_input == 'state'
        assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
        assert mac.agent.hyper_out_w is None
        before = [p.detach().clone() for p in learner.params]
        learner.train(batch, 10001, 1)
        assert all(th.isfinite(p).all() for p in learner.params)
        assert any(not th.equal(a, b) for a, b in zip(before, learner.params))
    print('PASS: Corridor only, three seeds, old identities/settings/head, state mixer, 10M/24h; real model TD updates')


def asset_checks():
    import smac.env.starcraft2.maps.smac_maps as installed
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-corridor-assets-') as temp:
        root = Path(temp)
        binary = root/'Versions/Base000/SC2_x64'
        binary.parent.mkdir(parents=True)
        binary.write_text('SYNTHETIC placeholder')
        (root/'Maps').mkdir()
        (root/'Maps/corridor.SC2Map').write_text('SYNTHETIC map')
        with patch.dict(os.environ, SC2PATH=temp), patch.object(installed, 'get_smac_map_registry',
                return_value={'corridor': dict(n_agents=6, n_enemies=24)}):
            suite.validate_map()
            (root/'Maps/corridor.SC2Map').unlink()
            try: suite.validate_map()
            except RuntimeError: pass
            else: raise AssertionError('Missing Corridor map not detected')


def scheduler_checks():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-corridor-submit-') as temp:
        logs, calls, retained = Path(temp), [], {}
        def run(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable']
            assert calls.count('preflight') == 3
            assert kwargs['env']['MAP_NAME'] == 'corridor'
            calls.append('submit')
            return str(96000 + calls.count('submit'))
        def subprocess_run(cmd, **kwargs):
            if cmd == ['quota', '-v']:
                return subprocess.CompletedProcess(cmd, 0, '/home 8388608 0 20969520 - 100000 0 200000 -', '')
            if cmd[0] == 'sbatch':
                assert cmd[1] == '--test-only' and 'submit' not in calls
                calls.append('preflight')
            else:
                assert cmd[1] in suite.SMOKE_SCRIPTS
            return subprocess.CompletedProcess(cmd, 0)
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='NO', REPO_DIR=str(ROOT)), \
                    patch.object(base, 'run', side_effect=AssertionError('Preview cannot call scheduler')):
                suite.main()
            low = subprocess.CompletedProcess([], 0, '/home 17000000 0 20969520 - 190000 0 200000 -', '')
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES'), \
                    patch.object(subprocess, 'run', return_value=low), patch.object(suite, 'submit_suite') as submit:
                try: suite.main()
                except RuntimeError: pass
                else: raise AssertionError('Low byte/inode capacity must block')
                submit.assert_not_called()
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(ROOT), RUNTIME_ROOT=temp), \
                    patch.object(suite, 'validate_map') as validate, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=retained), \
                    patch.object(base, 'completed_jobs', return_value={}), \
                    patch.object(base, 'run', side_effect=run), \
                    patch.object(subprocess, 'run', side_effect=subprocess_run):
                suite.main()
                validate.assert_called_once()
                assert calls == ['preflight']*3 + ['submit']*3
                record = json.loads(next(logs.glob(suite.GROUP+'_*.json')).read_text())
                retained.update(record['submitted'])
                calls.clear()
                suite.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: preview, capacity guard, assets, all Slurm preflights first, three receipts, repeat retains existing jobs (mocked)')


def main():
    model_checks()
    asset_checks()
    scheduler_checks()


if __name__ == '__main__':
    main()
