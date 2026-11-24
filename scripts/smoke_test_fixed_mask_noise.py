#!/usr/bin/env python3
"""Simulator-free checks: exact masks, noise RNG/cache, replay and nine real TD updates."""
import copy
import hashlib
import json
import shlex
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as SN
from unittest.mock import patch

import numpy as np
import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_fixed_mask_noise_27jobs as suite
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from envs import REGISTRY as ENV
from envs.starcraft import StarCraft2Env
from envs.starcraft_mixer_noise import configure_noise_env
from learners import REGISTRY
from learners.clean_learner import CleanLearner
from run.run import run_sequential, args_sanity_check, r_REGISTRY
from utils.fixed_heatmap_mask import load_heatmap_mask
from utils.mixer_state_importance import state_slot_names
from utils.mixer_state_refresh import extra_scheme, begin, scheduled_segments
from runners.parallel_runner import ParallelRunner, CloudpickleWrapper, env_worker

EXPECTED = {'corridor': [4,30,33,36,39,42,45,48,51,54,60,63,66,69,72,81,87,90,93,96,99],
            '3s_vs_5z': [0,5,15,19,23,27,30,31], '5m_vs_6m': [4,8,12,16,26,29,35]}


def extras(plan):
    return {k: yaml.safe_load(v) for k, v in
            (x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))}


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 27
    assert len(suite.build_plans(ROOT, experiment='mask')) == 9
    assert len(suite.build_plans(ROOT, experiment='noise')) == 18
    validate_config_keys(ROOT, plans)
    for scene, name, dim in suite.MAPS:
        selected = [p for p in plans if p['scene'] == scene]
        assert {(p['label'], p['seed']) for p in selected} == {
            (model, seed) for model in suite.MODELS for seed in (1, 2, 3)}
        mask, source = load_heatmap_mask(ROOT/suite.MASK_PATH, name, dim)
        assert [i for i, m in enumerate(mask) if not m] == EXPECTED[name]
        assert source['source_seed'] == 3
        for p in selected:
            x = extras(p)
            assert '--time=2-00:00:00' in p['sbatch_args']
            assert p['exports']['T_MAX'] == '10050000'
            assert not x['test_battle_videos'] and x['disable_video_media']
            gated = p['label'] == suite.MODELS[2]
            assert x['mixer_state_refresh_enabled'] == gated
            if gated:
                assert x['mixer_state_refresh_full_steps'] == 8000000
                assert x['mixer_input_initial_probability'] == 1
    print('PASS exact 27 plans, 21/8/7 closed slots, 48h/10M, unchanged init1/8M schedule', flush=True)


def noise_checks():
    cfg = yaml.safe_load((ROOT/'src/config/envs/sc2.yaml').read_text())['env_args']
    for _, name, dim in suite.MAPS:
        plain = ENV['sc2'](**dict(cfg, map_name=name, seed=7))
        def create(stream=0):
            return ENV['sc2'](**dict(cfg, map_name=name, seed=7, mixer_state_noise=True,
                                    mixer_state_noise_stream=stream))
        a, b, other = create(), create(), create(1)
        info = plain.get_env_info()
        assert a.get_env_info()['state_shape'] == 2*dim
        assert all(info[k] == a.get_env_info()[k] for k in ('obs_shape', 'n_agents', 'n_actions'))
        raw = np.arange(dim, dtype=np.float32)
        def reset(env):
            env._episode_steps = 0
        with patch.object(StarCraft2Env, 'get_state', return_value=raw), \
                patch.object(StarCraft2Env, 'reset', reset):
            np.random.seed(101)
            before = np.random.get_state()
            for e in (a, b, other):
                e.reset()
            first = a.get_state()
            assert np.array_equal(first, a.get_state())
            assert np.array_equal(first, b.get_state())
            assert np.array_equal(first[:dim], raw)
            assert not np.array_equal(first[dim:], other.get_state()[dim:])
            assert ((first[dim:] >= 0) & (first[dim:] < 1)).all()
            a.set_noise_test_mode(True)
            a.reset()
            a.get_state()
            a.set_noise_test_mode(False)
            a.reset()
            b.reset()
            assert np.array_equal(a.get_state(), b.get_state()), 'Eval consumed train noise RNG'
            a._episode_steps += 1
            assert not np.array_equal(first[dim:], a.get_state()[dim:])
            after = np.random.get_state()
            assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
        for e in (plain, a, b, other):
            e.close()
    print('PASS real SMAC dimensions; private worker/eval noise RNG; timestep cache; original state unchanged', flush=True)


def make(plan, temp):
    config = extras(plan)
    config['clean_model_type'] = plan['exports']['MODEL_TYPE']
    th.manual_seed(99)
    mac, _, raw, logger = make_case('baseline', plan['map_name'], production_shapes=True,
                                   config_overrides=config)
    args = mac.args
    args.env = 'sc2'
    configure_noise_env(args)
    args.local_results_path, args.unique_token = temp, 'SYNTHETIC_same_second'
    args.wandb_run_name = plan['job_name']
    noise = args.mixer_state_noise
    args.state_shape *= 2 if noise else 1
    scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
    scheme['state'] = dict(vshape=args.state_shape)
    scheme.update(extra_scheme(args, args.state_shape, 4))
    batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length,
        preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
    data = {k: v.clone() for k, v in raw.data.transition_data.items() if k != 'actions_onehot'}
    if noise:
        data['state'] = th.cat((data['state'], th.rand_like(data['state'])), dim=-1)
    batch.update(data, mark_filled=False)
    if args.mixer_state_refresh_enabled:
        for idx in range(batch.batch_size):
            cache = begin(args, 4, 1, idx*8000000)
            for t in range(batch.max_seq_length):
                batch.update(cache.apply(dict(state=[data['state'][idx,t].numpy()]), t),
                             bs=[idx], ts=t, mark_filled=False)
        assert th.equal(batch['mixer_state_raw'], data['state'])
        assert th.equal(batch['state'][1], data['state'][1]), '8M must be fully live'
    learner = REGISTRY[args.learner](mac, batch.scheme, logger, args)
    learner.last_image_t = 10000000
    return learner, batch


def runner_checks():
    # Inspect the exact factory passed to production worker processes.
    runner = object.__new__(ParallelRunner)
    runner.batch_size = 3
    runner.args = SN(env='sc2', env_args=dict(mixer_state_noise=True, seed=7))
    runner._env_fn = CloudpickleWrapper(lambda: None)
    runner.worker_reset_retries, runner.worker_reset_retry_delay = 1, 0
    with patch('runners.parallel_runner.Pipe', side_effect=lambda: (object(), object())), \
            patch('runners.parallel_runner.Process') as process:
        runner._start_workers()
        streams = [call.kwargs['args'][1].x.keywords['mixer_state_noise_stream'] for call in process.call_args_list]
        assert streams == [0, 1, 2]
    # Execute actual env_worker command handling including a reset retry.
    created, modes = [], []
    class FakeEnv:
        def __init__(self):
            self.idx = len(created)
            created.append(self)
        def set_noise_test_mode(self, mode):
            self.mode = mode
            modes.append((self.idx, mode))
        def reset(self):
            assert self.mode is True
            if self.idx == 0:
                raise RuntimeError('SYNTHETIC reset failure')
        def get_state(self): return [1, .25]
        def get_obs(self): return [[2]]
        def get_avail_actions(self): return [[1]]
        def close(self): pass
    class Remote:
        def __init__(self):
            self.commands = iter([('reset', {'mixer_noise_test_mode': True}), ('close', None)])
            self.sent = []
        def recv(self): return next(self.commands)
        def send(self, value): self.sent.append(value)
        def close(self): pass
    remote = Remote()
    env_worker(remote, CloudpickleWrapper(FakeEnv), reset_retries=1, reset_retry_delay=0)
    assert modes == [(0, True), (1, True)]
    assert remote.sent == [dict(state=[1, .25], obs=[[2]], avail_actions=[[1]])]
    print('PASS production parallel worker streams and eval-mode propagation through reset retry', flush=True)


def startup_checks():
    from smoke_test_no_video_startup import config_for
    import logging
    from utils.logging import Logger
    class Reached(Exception): pass
    for _, name, _ in suite.MAPS:
        for plan in [p for p in suite.build_plans(ROOT) if p['map_name'] == name and p['seed'] == 1]:
            cfg = config_for(plan)
            cfg['clean_model_type'] = plan['exports']['MODEL_TYPE']
            log = logging.getLogger('SYNTHETIC-masknoise-startup')
            args = SN(**args_sanity_check(cfg, log))
            def runner(args, logger):
                assert args.env_args.get('mixer_state_noise', False) == args.mixer_state_noise
                raise Reached()
            with patch.dict(r_REGISTRY, {args.runner: runner}):
                try:
                    run_sequential(args, Logger(log))
                except Reached:
                    pass
                else:
                    raise AssertionError('Startup did not reach runner')
    print('PASS all nine real startup preflights and opt-in noise propagation', flush=True)


def model_checks(temp):
    for scene, name, dim in suite.MAPS:
        for label in suite.MODELS:
            plan = next(p for p in suite.build_plans(ROOT, scene) if p['label'] == label)
            learner, batch = make(plan, temp)
            m = learner.mixer
            original = copy.deepcopy(learner.mac.agent.state_dict())
            mask = m.coefficients().detach().clone()
            names = state_slot_names(learner.args, learner.args.state_shape)
            assert len(names) == learner.args.state_shape
            if learner.args.mixer_state_noise:
                assert names[dim:] == ['noise_'+str(i) for i in range(dim)]
                changed = copy.deepcopy(batch)
                changed['state'][..., dim:] += 10
                learner.mac.init_hidden(batch.batch_size)
                q1 = learner.mac.forward(batch, t=0).detach()
                learner.mac.init_hidden(batch.batch_size)
                assert th.equal(q1, learner.mac.forward(changed, t=0).detach()), 'Noise leaked into actor'
            if label == suite.MODELS[0]:
                assert [i for i, x in enumerate(mask) if not x] == EXPECTED[name]
            if label != suite.MODELS[2]:
                assert not hasattr(m, 'gate_logits') and not hasattr(m, 'gate_parameter')
                x = m.controlled_input(batch['state'], m.gate_values(batch['state']))
                assert th.equal(x[..., mask == 1], batch['state'][..., mask == 1])
                assert th.all(x[..., mask == 0] == 1)
                q = th.randn(batch.batch_size, batch.max_seq_length, learner.args.n_agents)
                assert th.allclose(m(q, batch['state']), m.base(q, x))
            else:
                assert th.all(mask == 1)
                assert scheduled_segments(learner.args, 8000000, 170) == 170
            # Same augmented transition is actually retained by replay.
            replay = ReplayBuffer({k:v for k,v in batch.scheme.items() if k not in {'filled','actions_onehot'}},
                batch.groups, 2, batch.max_seq_length,
                preprocess={'actions': ('actions_onehot', [OneHot(learner.args.n_actions)])})
            replay.insert_episode_batch(batch)
            assert th.equal(replay['state'][:2], batch['state'])
            for step in (1000000, 5000000, 9000000):
                learner.train(batch, step, step//1000000)
            assert any(not th.equal(original[k], v) for k,v in learner.mac.agent.state_dict().items())
            assert all(th.isfinite(p).all() for p in learner.params)
            if label != suite.MODELS[2]:
                assert th.equal(mask, m.coefficients())
            else:
                assert m.gate_parameter.grad is not None
                assert th.any(mask != m.coefficients())
            if label == suite.MODELS[1]:
                # Identity-mask wrapper must preserve the ordinary QMIX TD recipe
                # on complete valid episodes, not just match a mixer forward.
                dense = copy.deepcopy(batch)
                dense['filled'].fill_(1)
                dense['terminated'].zero_()
                dense['terminated'][:, -2] = 1
                fresh, _ = make(plan, temp)
                plain = CleanLearner(copy.deepcopy(fresh.mac), dense.scheme, fresh.logger, fresh.args)
                plain.mixer.load_state_dict(fresh.mixer.base.state_dict())
                plain.target_mixer.load_state_dict(fresh.target_mixer.base.state_dict())
                fresh.train(dense, 1, 1)
                plain.train(dense, 1, 1)
                for a,b in zip(fresh.mac.parameters(), plain.mac.parameters()):
                    th.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
                for a,b in zip(fresh.mixer.base.parameters(), plain.mixer.parameters()):
                    th.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
            path = Path(temp)/(scene+'_'+label)
            path.mkdir()
            learner.save_models(path)
            restored, _ = make(plan, temp)
            restored.load_models(path)
            assert th.equal(m.coefficients(), restored.mixer.coefficients())
            if label == suite.MODELS[0]:
                saved = th.load(path/'onefill_training.th', weights_only=False)
                saved['mixer']['fixed_mask'][0] = 1-saved['mixer']['fixed_mask'][0]
                th.save(saved, path/'onefill_training.th')
                try:
                    restored.load_models(path)
                except ValueError as error:
                    assert 'mask differs' in str(error)
                else:
                    raise AssertionError('Checkpoint replaced the requested fixed mask')
            print('PASS real forward/backward + replay + save/load:', name, label, flush=True)


def main():
    th.set_num_threads(1)
    plan_checks()
    noise_checks()
    runner_checks()
    startup_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-masknoise-') as temp:
        model_checks(temp)
    print('PASS no simulator launched; no submission/upload/deletion')


if __name__ == '__main__':
    main()
