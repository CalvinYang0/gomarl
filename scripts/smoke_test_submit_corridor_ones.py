#!/usr/bin/env python3
"""Matched Corridor ones plan/model/mock-Slurm QA; no real jobs or uploads."""
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

import ozstar_submit_corridor_ones_3jobs as suite
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from modules.mixers.qmix import QMixer

ROOT = Path(__file__).resolve().parents[1]


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def model_checks():
    plans, reference = suite.build_plans(ROOT), suite.baseline.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 3
    assert {p['seed'] for p in plans} == {1, 2, 3}
    assert {p['job_name'] for p in plans}.isdisjoint({p['job_name'] for p in reference})
    validate_config_keys(ROOT, plans)
    th.set_num_threads(1)
    for plan, previous in zip(plans, reference):
        actual, original = extras(plan), extras(previous)
        assert {k: v for k, v in actual.items() if k != 'qmix_condition_input'} == {
            k: v for k, v in original.items() if k != 'qmix_condition_input'}
        assert actual['qmix_condition_input'] == 'ones' and original['qmix_condition_input'] == 'state'
        for key, value in previous['exports'].items():
            if key not in {'RUN_NAME', 'GROUP_NAME', 'EXTRA_ARGS'}:
                assert plan['exports'][key] == value, key
        assert plan['exports']['GROUP_NAME'] == suite.GROUP and plan['label'] == suite.LABEL
        assert plan['map_name'] == 'corridor' and plan['memory'] == '24G'
        assert '--time=1-00:00:00' in plan['sbatch_args'] and plan['exports']['T_MAX'] == '10050000'
        assert plan['sbatch_args'] == [a.replace(previous['job_name'], plan['job_name'])
                                     for a in previous['sbatch_args']]
        pair = []
        for flags in (original, actual):
            overrides = {k: yaml.safe_load(v) for k, v in flags.items() if '.' not in k}
            overrides['clean_model_type'] = 'qmix_minimal'
            th.manual_seed(plan['seed'])
            pair.append(make_case('baseline', 'corridor', production_shapes=True, config_overrides=overrides))
        raw_mac, raw, batch, _ = pair[0]
        mac, learner, other_batch, _ = pair[1]
        assert type(raw) is type(learner) and isinstance(learner.mixer, QMixer)
        assert all(th.equal(a, b) for a, b in zip(raw.params, learner.params))
        assert th.equal(batch['obs'], other_batch['obs'])
        assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
        assert mac.agent.hyper_out_w is None
        for model in (raw_mac, mac): model.init_hidden(batch.batch_size)
        with th.no_grad():
            assert th.equal(raw_mac.forward(batch, 0), mac.forward(batch, 0))
        q = th.randn(2, 4, learner.args.n_agents)
        state = batch['state'][:, :-1]
        assert th.equal(learner.mixer(q, state), raw.mixer(q, th.ones_like(state)))
        assert th.equal(learner.mixer(q, state), learner.mixer(q, state+333))
        seen = []
        for target, mixer in ((False, learner.mixer), (True, learner.target_mixer)):
            for name in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value'):
                getattr(mixer, name).register_forward_pre_hook(
                    lambda module, inputs, t=target, n=name: seen.append((t, n, inputs[0].detach().clone())))
        before = [p.detach().clone() for p in learner.params]
        learner.train(batch, 10001, 1)
        assert all(th.isfinite(p).all() for p in learner.params)
        assert any(not th.equal(a, b) for a, b in zip(before, learner.params))
        assert {(t, n) for t, n, _ in seen} == {(t, n) for t in (True, False)
            for n in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value')}
        assert all(th.equal(x, th.ones_like(x)) for _, _, x in seen)
    print('PASS: Corridor three-seed plans match baseline except condition/identity; same head/learner/init, all online+target W/b inputs ones, TD updates')


def scheduler_checks():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-corridor-ones-submit-') as temp:
        logs, calls = Path(temp), []
        plans = suite.build_plans(ROOT)
        active = {plans[0]['job_name']: 'SYNTHETIC_ACTIVE'}
        completed = {plans[1]['job_name']: 'SYNTHETIC_COMPLETED'}
        def run(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 1
            assert kwargs['env']['MAP_NAME'] == 'corridor' and kwargs['env']['GROUP_NAME'] == suite.GROUP
            calls.append('submit')
            return 'SYNTHETIC_NEW_JOB'
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
                    else: raise AssertionError('Byte/inode guard bypassed')
                    submit.assert_not_called()
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(ROOT), RUNTIME_ROOT=temp), \
                    patch.object(suite, 'validate_map') as validate, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=active), \
                    patch.object(base, 'completed_jobs', return_value=completed), \
                    patch.object(base, 'run', side_effect=run), \
                    patch.object(subprocess, 'run', side_effect=subprocess_run):
                suite.main()
                validate.assert_called_once()
                assert calls == ['preflight', 'submit']
                record = json.loads(next(logs.glob(suite.GROUP+'_*.json')).read_text())
                assert len(record['submitted']) == 1 and len(record['retained']) == 2
                active.update(record['submitted'])
                calls.clear()
                suite.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: preview, byte/inode guards, Corridor asset validator, Slurm preflight/receipts and active/completed retention (mocked)')


def main():
    model_checks()
    scheduler_checks()


if __name__ == '__main__':
    main()
