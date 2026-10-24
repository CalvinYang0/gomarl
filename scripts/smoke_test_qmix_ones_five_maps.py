#!/usr/bin/env python3
"""Synthetic model/plan/mock-Slurm QA; no real environments, jobs or uploads."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import torch as th
import yaml

from smoke_test_counter_transformer_nine import make_case
from modules.mixers.qmix import QMixer
from modules.mixers.state_input_control import StateInputControlMixer
import ozstar_submit_qmix_ones_five_maps_30jobs as study
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys

ROOT = Path(__file__).resolve().parents[1]


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def plan_checks():
    plans = study.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 30
    smac = study.build_plans(ROOT, 'smac')
    assert len(smac) == len({p['job_name'] for p in smac}) == 12
    assert {p['map_name'] for p in smac} == {'MMM2', '3s5z_vs_3s6z'}
    assert all(p['domain'] == 'smac' and p['exports']['GROUP_NAME'] == study.SMAC_GROUP for p in smac)
    validate_config_keys(ROOT, smac)
    validate_config_keys(ROOT, plans)
    for scene, map_name, domain, _, _ in study.SCENES:
        for seed in study.SEEDS:
            pair = [p for p in plans if p['scene'] == scene and p['seed'] == seed]
            assert [p['label'] for p in pair] == ['qmix', 'qmix_state_ones']
            first, second = pair
            assert {k: v for k, v in extras(first).items() if k != 'qmix_condition_input'} == {
                k: v for k, v in extras(second).items() if k != 'qmix_condition_input'}
            assert extras(first)['qmix_condition_input'] == 'state'
            assert extras(second)['qmix_condition_input'] == 'ones'
            for key in first['exports']:
                if key not in {'RUN_NAME', 'EXTRA_ARGS'}:
                    assert first['exports'][key] == second['exports'][key], key
            for plan in pair:
                assert plan['map_name'] == map_name and plan['domain'] == domain
                exports, args = plan['exports'], extras(plan)
                assert exports['MODEL_TYPE'] == 'qmix_minimal'
                assert exports['T_MAX'] == '10050000' and plan['target_steps'] == 10000000
                assert exports['BATCH_SIZE_RUN'] == '8' and exports['BATCH_SIZE'] == '128'
                assert args['learner'] == 'clean_learner' and args['mixer'] == 'qmix'
                assert args['mixer_state_refresh_enabled'] == 'False'
                assert args['test_battle_videos'] == str(domain == 'smac')
                assert args['test_nepisode'] == '32' and exports['TEST_INTERVAL'] == '10000'
                assert plan['memory'] == '32G' and '--time=1-00:00:00' in plan['sbatch_args']
    for quota in (dict(free_gib=4.9, free_inodes=100000), dict(free_gib=10, free_inodes=49999)):
        try: study.require_capacity(quota)
        except RuntimeError: pass
        else: raise AssertionError('Quota guard bypassed')
    with redirect_stdout(io.StringIO()): study.require_capacity(dict(free_gib=5, free_inodes=50000))
    print('PASS: 30 plans, three seeds, five exact maps, same learner/resources/settings per pair; both quota floors')
    return plans


def actor_values(mac, batch):
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        return th.stack([mac.forward(batch, t) for t in range(batch.max_seq_length)], dim=1)


def model_checks(plans):
    th.set_num_threads(1)
    for scene, map_name, domain, _, _ in study.SCENES:
        for optimizer in ('adam', 'rmsprop'):
            pair = []
            for label in study.LABELS:
                plan = next(p for p in plans if p['scene'] == scene and p['seed'] == 1 and p['label'] == label)
                overrides = {k: yaml.safe_load(v) for k, v in extras(plan).items() if '.' not in k}
                overrides.update(clean_model_type='qmix_minimal', optimizer=optimizer)
                th.manual_seed(61)
                pair.append(make_case('baseline', map_name, production_shapes=True, config_overrides=overrides))
            (raw_mac, raw, batch, _), (ones_mac, ones, _, _) = pair
            assert all(th.equal(a, b) for a, b in zip(raw.params, ones.params)), 'Initialization differs'
            assert all(th.equal(a, b) for a, b in zip(raw.target_mixer.parameters(), ones.target_mixer.parameters()))
            assert len(raw_mac.agent.fixed_head) == 3 and isinstance(raw_mac.agent.fixed_head[1], th.nn.ELU)
            assert raw_mac.agent.hyper_out_w is None and ones_mac.agent.hyper_out_w is None
            assert isinstance(raw.mixer, QMixer) and isinstance(ones.mixer, QMixer)
            assert set(raw.mixer.state_dict()) == set(ones.mixer.state_dict())
            assert raw.mixer.condition_input == 'state' and ones.mixer.condition_input == 'ones'
            assert th.equal(actor_values(raw_mac, batch), actor_values(ones_mac, batch))
            changed = copy.deepcopy(batch)
            changed['state'].add_(333)
            assert th.equal(actor_values(ones_mac, batch), actor_values(ones_mac, changed))
            changed['obs'].add_(1)
            assert not th.equal(actor_values(ones_mac, batch), actor_values(ones_mac, changed))
            q = th.randn(2, 2, raw.args.n_agents, requires_grad=True)
            state = th.randn(2, 2, int(raw.args.state_shape), requires_grad=True)
            expected = raw.mixer(q, th.ones_like(state))
            value = ones.mixer(q, state)
            assert th.equal(value, expected) and th.equal(value, ones.mixer(q, state+999))
            ds, dq = th.autograd.grad(value.sum(), (state, q), allow_unused=True)
            assert ds is None and dq.abs().sum() > 0 and (dq >= 0).all()
            assert not th.equal(raw.mixer(q, state), raw.mixer(q, state+2)), 'Ordinary QMIX ignored state'
            # Old all-ones forward algebra, without using its different learner.
            legacy_args = copy.deepcopy(raw.args)
            legacy_args.mixer_input_control = 'ones'
            legacy = StateInputControlMixer(legacy_args)
            legacy.mixers[0].load_state_dict(ones.mixer.state_dict())
            assert th.equal(value, legacy.values(q, state, 'eval'))
            seen = []
            for target, mixer in ((False, ones.mixer), (True, ones.target_mixer)):
                for name in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value'):
                    getattr(mixer, name).register_forward_pre_hook(
                        lambda module, inputs, t=target, n=name: seen.append((t, n, inputs[0].detach().clone())))
            before = [p.detach().clone() for p in ones.params]
            raw.train(batch, 10001, 1)
            ones.train(batch, 10001, 1)
            assert all(th.isfinite(p).all() for p in raw.params + ones.params)
            assert any(not th.equal(a, b) for a, b in zip(before, ones.params))
            assert {(t, n) for t, n, _ in seen} == {(t, n) for t in (True, False)
                for n in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value')}
            assert all(th.equal(x, th.ones_like(x)) for _, _, x in seen)
            default_args = copy.deepcopy(raw.args)
            delattr(default_args, 'qmix_condition_input')
            th.manual_seed(101); default = QMixer(default_args); after = th.get_rng_state()
            th.manual_seed(101); explicit = QMixer(raw.args)
            assert th.equal(after, th.get_rng_state())
            assert th.equal(default(q, state), explicit(q, state))
            print('PASS:', map_name, optimizer, 'historical head/init; live actor; constant all W/b train/target; old ones algebra; TD update')


def scheduler_checks(domain):
    count = len(study.build_plans(ROOT, domain))
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-qmix-ones-submit-') as temp:
        logs = Path(temp)
        retained, calls = {}, []
        def run_command(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == count
            calls.append('submit')
            return str(95000+calls.count('submit'))
        def subprocess_command(cmd, **kwargs):
            if cmd == ['quota', '-v']:
                return subprocess.CompletedProcess(cmd, 0, '/home 8388608 0 20969520 - 100000 0 200000 -', '')
            if cmd[0] == 'sbatch':
                assert cmd[1] == '--test-only' and 'submit' not in calls
                calls.append('preflight')
            else:
                assert cmd[1] == study.SMOKE_SCRIPTS[0]
            return subprocess.CompletedProcess(cmd, 0)
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(ROOT), RUNTIME_ROOT=temp), \
                    patch.object(study, 'validate_environments') as env_probe, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=retained), \
                    patch.object(base, 'completed_jobs', return_value={}), \
                    patch.object(base, 'run', side_effect=run_command), \
                    patch.object(subprocess, 'run', side_effect=subprocess_command):
                study.main(['--domain', domain])
                env_probe.assert_called_once()
                assert calls.count('submit') == calls.count('preflight') == count
                manifest = json.loads(next(logs.glob(study.group_for(domain)+'_*.json')).read_text())
                retained.update(manifest['submitted'])
                calls.clear()
                study.main(['--domain', domain])
                assert not calls, 'Repeated submit duplicated active jobs'
        finally:
            os.chdir(cwd)
    print('PASS:', domain, count, 'Slurm preflights before submission, manifest persistence, idempotence (all mocked)')


def environment_probe_checks():
    from envs import REGISTRY
    for _, name, domain, agents, width in study.SCENES:
        if domain != 'grf':
            continue
        class FakeEnv:
            closed = False
            def reset(self): pass
            def get_env_info(self):
                return dict(n_agents=agents, n_actions=19, obs_shape=width, state_shape=width)
            def get_state(self): return np.zeros(width)
            def get_obs(self): return np.zeros((agents, width))
            def step(self, actions):
                assert actions.shape == (agents,)
                return -1, False, {}
            def close(self): self.closed = True
        fake = FakeEnv()
        def construct(**kwargs):
            assert kwargs['map_name'] == name and kwargs['n_agents'] == agents
            assert not any(kwargs[k] for k in ('write_video', 'render', 'write_full_episode_dumps', 'write_goal_dumps'))
            return fake
        with redirect_stdout(io.StringIO()), patch.dict(REGISTRY, {name: construct}):
            study.probe_grf(name)
        assert fake.closed
    # Missing capacity must stop before environment/model/Slurm preflights.
    low = subprocess.CompletedProcess(['quota', '-v'], 0, '/home 17000000 0 20969520 - 190000 0 200000 -', '')
    with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES'), \
            patch.object(subprocess, 'run', return_value=low), patch.object(study, 'submit_suite') as submit:
        try: study.main([])
        except RuntimeError: pass
        else: raise AssertionError('Submission reached after insufficient capacity')
        submit.assert_not_called()
    print('PASS: GRF reset/step probe wiring, recording disabled, close; quota failure precedes all submission work (mocked)')


def smac_only_checks():
    import smac.env.starcraft2.maps.smac_maps as installed
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-smac-assets-') as temp:
        root = Path(temp)
        binary = root/'Versions/Base000/SC2_x64'
        binary.parent.mkdir(parents=True)
        binary.write_text('SYNTHETIC placeholder, not executable')
        (root/'Maps').mkdir()
        for name in ('MMM2', '3s5z_vs_3s6z'):
            (root/'Maps'/(name+'.SC2Map')).write_text('SYNTHETIC map placeholder')
        params = {'MMM2': dict(n_agents=10, n_enemies=12), '3s5z_vs_3s6z': dict(n_agents=8, n_enemies=9)}
        with redirect_stdout(io.StringIO()), patch.dict(os.environ, SC2PATH=temp), \
                patch.object(installed, 'get_smac_map_registry', return_value=params), \
                patch.object(subprocess, 'run', side_effect=AssertionError('SMAC-only cannot launch GRF probes')):
            study.validate_environments('smac')
    for argv in ([], ['--domain', 'smac']):
        with patch.dict(os.environ, SUBMIT='NO'), patch.object(study, 'submit_suite') as submit:
            study.main(argv)
            args, kwargs = submit.call_args
            assert args[0] == study.SMAC_GROUP and len(args[2](ROOT)) == 12
            with patch.object(study, 'validate_environments') as validate:
                kwargs['map_validator']()
                validate.assert_called_once_with('smac')
    print('PASS: default/explicit SMAC-only = 12 jobs, dedicated cohort, no GRF simulator/probe/submission')


def main():
    plans = plan_checks()
    model_checks(plans)
    environment_probe_checks()
    smac_only_checks()
    scheduler_checks('all')
    scheduler_checks('smac')


if __name__ == '__main__':
    main()
