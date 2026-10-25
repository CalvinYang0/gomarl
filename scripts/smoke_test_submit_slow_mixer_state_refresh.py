#!/usr/bin/env python3
"""Six-plan/protocol/mock-Slurm QA only; no real simulator, jobs or W&B."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_mixer_state_refresh_slow_6jobs as suite
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def plan_checks():
    plans = suite.build_plans(ROOT)
    old = suite.previous.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 6
    assert len(old) == 9 and {extras(p)['mixer_state_refresh_stage_steps'] for p in old} == {'500000', '1000000', '2000000'}
    assert {p['job_name'] for p in plans}.isdisjoint({p['job_name'] for p in old})
    for stage in suite.STAGE_STEPS:
        assert {p['seed'] for p in plans if extras(p)['mixer_state_refresh_stage_steps'] == str(stage)} == {1, 2, 3}
    for plan in plans:
        reference = next(p for p in old if p['seed'] == plan['seed'])
        actual, expected = extras(plan), extras(reference)
        assert plan['scene'] == 'smac_5m6m' and plan['map_name'] == '5m_vs_6m'
        assert plan['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert plan['exports']['GROUP_NAME'] == suite.GROUP
        assert plan['exports']['T_MAX'] == '10050000' and '--time=1-00:00:00' in plan['sbatch_args']
        assert actual['learner'] == 'mixer_state_refresh_learner'
        assert actual['mixer_state_refresh_enabled'] == 'True'
        assert {k: v for k, v in actual.items() if k != 'mixer_state_refresh_stage_steps'} == {
            k: v for k, v in expected.items() if k != 'mixer_state_refresh_stage_steps'}
        for key in reference['exports']:
            if key not in {'RUN_NAME', 'GROUP_NAME', 'EXTRA_ARGS'}:
                assert plan['exports'][key] == reference['exports'][key], key
        assert plan['sbatch_args'] == [a.replace(reference['job_name'], plan['job_name'])
                                     for a in reference['sbatch_args']]
        assert plan['job_name'].endswith('_refresh20261026')
    validate_config_keys(ROOT, plans)
    for stages in ((), (0,), (True,), (5000000, 5000000), (5000001,)):
        try: suite.previous.build_plans(ROOT, stage_steps=stages)
        except ValueError: pass
        else: raise AssertionError('Invalid stage lengths accepted')
    print('PASS: six unique 5m6m jobs, 5M/10M x seeds 1/2/3; only stage and identities changed; old nine unchanged')


def protocol_checks():
    import numpy as np
    from types import SimpleNamespace
    from utils.mixer_state_refresh import MixerStateCache, segments
    # 120 is a synthetic episode limit, not a claim about the installed map.
    for stage, cases in ((5000000, ((0, 1), (4999999, 1), (5000000, 2), (9999999, 2), (10000000, 3))),
                         (10000000, ((0, 1), (5000000, 1), (9999999, 1), (10000000, 2)))):
        args = SimpleNamespace(mixer_state_refresh_enabled=True, clean_model_type='qmix_minimal',
            mixer='qmix', learner='mixer_state_refresh_learner', mixer_state_refresh_stage_steps=stage)
        for t_env, expected in cases:
            assert segments(t_env, stage, 120) == expected
            cache = MixerStateCache(args, 120, 1, t_env)
            last = None
            for t in range(121):
                held = cache.apply({'state': [np.array([t], dtype=np.float32)]}, t)
                if t in cache.starts: last = t
                assert held['state'][0][0] == last
                assert held['mixer_state_source_t'][0] == (last,)
            assert cache.count == expected
    print('PASS: 5M/10M boundaries; per-episode hold and terminal cache; 10M stage changes after 10M (including final 50k margin)')


def scheduler_checks():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-slow-refresh-submit-') as temp:
        logs, calls, retained = Path(temp), [], {}
        def run(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 6
            assert kwargs['env']['GROUP_NAME'] == suite.GROUP
            calls.append('submit')
            return str(97000 + calls.count('submit'))
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
            for row in ('/home 18000000 0 20969520 - 100000 0 200000 -',
                        '/home 8388608 0 20969520 - 150001 0 200000 -'):
                with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES'), \
                        patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, row, '')), \
                        patch.object(suite, 'submit_suite') as submit:
                    try: suite.main()
                    except RuntimeError: pass
                    else: raise AssertionError('Insufficient bytes/inodes must block')
                    submit.assert_not_called()
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(ROOT), RUNTIME_ROOT=temp), \
                    patch.object(suite.previous, 'validate_installed_map') as validate, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=retained), \
                    patch.object(base, 'completed_jobs', return_value={}), \
                    patch.object(base, 'run', side_effect=run), \
                    patch.object(subprocess, 'run', side_effect=subprocess_run):
                suite.main()
                validate.assert_called_once()
                assert calls == ['preflight']*6 + ['submit']*6
                record = json.loads(next(logs.glob(suite.GROUP+'_*.json')).read_text())
                retained.update(record['submitted'])
                assert len(retained) == 6
                calls.clear()
                suite.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: preview, byte/inode guards, all Slurm preflights first, six receipts and repeat retention (mocked)')


def main():
    plan_checks()
    protocol_checks()
    scheduler_checks()


if __name__ == '__main__':
    main()
