#!/usr/bin/env python3
"""Synthetic-only curriculum, runner/replay, historical TD and actor-head QA."""
import copy
from collections import deque
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from envs import REGISTRY as ENVS
from learners.clean_learner import CleanLearner
from learners.mixer_state_refresh_learner import MixerStateRefreshLearner
from runners.parallel_runner import ParallelRunner
from runners.episode_runner import EpisodeRunner
from utils.mixer_state_refresh import enabled, begin, extra_scheme, segments, boundaries, source_times
from utils.logging import Logger
from utils.value_diagnostics import collect_value_diagnostics, ValueDiagnosticSummary
from smoke_test_counter_transformer_nine import make_case


def options(**kwargs):
    values = dict(mixer_state_refresh_enabled=True, mixer_state_refresh_stage_steps=10,
        learner='mixer_state_refresh_learner', clean_model_type='qmix_minimal', mixer='qmix')
    values.update(kwargs)
    return SN(**values)


def schedule_checks():
    for stage in (500000, 1000000, 2000000):
        assert segments(0, stage, 120) == 1
        assert segments(stage-1, stage, 120) == 1
        assert segments(stage, stage, 120) == 2
        assert segments(2*stage, stage, 120) == 3
    assert boundaries(120, 2) == (0, 60)
    assert boundaries(120, 3) == (0, 40, 80)
    assert boundaries(7, 3) == (0, 2, 4)
    assert source_times(7, 3, 8) == (0, 0, 2, 2, 4, 4, 4, 4)
    args = options()
    cache = begin(args, 7, 2, 20)
    for t in range(8):
        raw = np.array([[t, t+10], [100+t, 200+t]], dtype=np.float32)
        before = raw.copy()
        data = cache.apply(dict(state=raw, obs=raw+1000), t)
        src = source_times(7, 3, 8)[t]
        assert np.array_equal(raw, before)
        assert np.array_equal(data['state'], [[src, src+10], [100+src, 200+src]])
        assert np.array_equal(data['mixer_state_raw'], raw)
        assert np.array_equal(data['obs'], raw+1000)
        if t == 0:
            assert data['mixer_state_segments'] == [(3,), (3,)]
        args.mixer_state_refresh_stage_steps = 1  # Active episode retains its chosen K.
    old = SN(mixer_state_refresh_enabled=False)
    assert not enabled(old) and begin(old, 7, 2, 123) is None
    assert extra_scheme(old, 2, 7) == {}
    for changes in (dict(mixer_state_refresh_stage_steps=0), dict(mixer_state_refresh_stage_steps=.5),
                    dict(clean_model_type='baseline'), dict(mixer='vdn'), dict(learner='clean_learner')):
        try:
            begin(options(**changes), 7, 1, 0)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported curriculum configuration accepted')
    print('PASS: phase boundaries, maximum-length thirds, terminal hold, fixed per-episode K, disabled/default isolation')


class FakeEnv:
    episode_limit = 7
    def __init__(self, index=0, **kwargs): self.idx, self.t = index, 0
    def reset(self): self.t = 0
    def get_state(self): return np.full(4, self.idx*100+self.t, dtype=np.float32)
    def get_obs(self): return np.full((2, 3), self.idx*100+self.t, dtype=np.float32)
    def get_avail_actions(self): return np.ones((2, 3), dtype=int)
    def get_env_info(self): return dict(n_agents=2, n_actions=3, obs_shape=3, state_shape=4, episode_limit=7)
    def step(self, actions):
        self.t += 1
        return 1., self.t == 3+2*self.idx, dict(battle_won=True, episode_limit=False)
    def get_stats(self): return {}
    def close(self): pass


class FakeConn:
    def __init__(self, idx): self.env, self.queue = FakeEnv(idx), deque()
    def send(self, message):
        cmd, payload = message
        if cmd == 'get_env_info': self.queue.append(self.env.get_env_info()); return
        if cmd == 'get_stats': self.queue.append({}); return
        if cmd == 'reset': self.env.reset()
        else: reward, done, info = self.env.step(payload)
        data = dict(state=self.env.get_state(), obs=self.env.get_obs(), avail_actions=self.env.get_avail_actions())
        if cmd != 'reset': data.update(reward=reward, terminated=done, info=info)
        self.queue.append(data)
    def poll(self, timeout): return bool(self.queue)
    def recv(self): return self.queue.popleft()


class FakeMAC:
    action_selector = SN(epsilon=0)
    def init_hidden(self, batch_size): pass
    def select_actions(self, batch, t_ep, t_env, bs=None, test_mode=False):
        count = batch.batch_size if bs is None else len(bs)
        return th.zeros(count, 2, dtype=th.long)


def fake_workers(runner):
    runner.parent_conns = tuple(FakeConn(i) for i in range(runner.batch_size))
    runner.worker_conns = ()
    runner.ps = [SN(is_alive=lambda: True, exitcode=None) for _ in range(runner.batch_size)]


def runner_checks():
    logger = Logger(logging.getLogger('SYNTHETIC-state-refresh'))
    for parallel in (False, True):
        for active in (False, True):
            args = options(mixer_state_refresh_enabled=active, env='SYNTHETIC_refresh', env_args={},
                batch_size_run=2 if parallel else 1, device='cpu', test_nepisode=2,
                runner_log_interval=1000, seed=1)
            if parallel:
                with patch.object(ParallelRunner, '_start_workers', fake_workers), patch.dict(ENVS, SYNTHETIC_refresh=FakeEnv):
                    runner = ParallelRunner(args, logger)
            else:
                with patch.dict(ENVS, SYNTHETIC_refresh=FakeEnv): runner = EpisodeRunner(args, logger)
            scheme = dict(state=dict(vshape=4), obs=dict(vshape=3, group='agents'),
                actions=dict(vshape=(1,), group='agents', dtype=th.long),
                avail_actions=dict(vshape=3, group='agents', dtype=th.int),
                reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8))
            scheme.update(extra_scheme(args, 4, 7))
            runner.setup(scheme, {'agents': 2}, {'actions': ('actions_onehot', [OneHot(3)])}, FakeMAC())
            for t_env, test in ((0, False), (10, True), (20, False), (20, True)):
                runner.t_env = t_env
                batch = runner.run(test_mode=test)
                if test: assert runner.t_env == t_env
                for idx in range(args.batch_size_run):
                    length = 3+2*idx if parallel else 3
                    times = th.arange(length+1).float()+100*idx
                    assert th.equal(batch['obs'][idx, :length+1, 0, 0], times)
                    if active:
                        count = segments(t_env, 10, 7)
                        sources = th.tensor(source_times(7, count, length+1))
                        assert th.equal(batch['mixer_state_source_t'][idx, :length+1, 0], sources)
                        assert th.equal(batch['state'][idx, :length+1, 0], sources.float()+100*idx)
                        assert th.equal(batch['mixer_state_raw'][idx, :length+1, 0], times)
                        assert batch['mixer_state_segments'][idx].item() == count
                    else:
                        assert 'mixer_state_raw' not in batch.scheme
                        assert th.equal(batch['state'][idx, :length+1, 0], times)
    print('PASS: both actual runner paths, live actor obs, short/unequal episodes, resets, train/eval schedule, unchanged disabled path')


def cached_batch(original, args):
    scheme = {key: value for key, value in original.scheme.items() if key not in {'filled', 'actions_onehot'}}
    scheme.update(extra_scheme(args, args.state_shape, 7))
    batch = EpisodeBatch(scheme, original.groups, original.batch_size, original.max_seq_length,
        preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
    batch.update({key: value.clone() for key, value in original.data.transition_data.items()
                  if key != 'actions_onehot'}, mark_filled=False)
    for idx in range(original.batch_size):
        cache = begin(args, 7, 1, 10*(idx+1))
        for t in range(original.max_seq_length):
            values = cache.apply(dict(state=[original['state'][idx, t].cpu().numpy()]), t)
            batch.update(values, bs=[idx], ts=t, mark_filled=False)
    return batch


def actor_q(mac, batch):
    mac.init_hidden(batch.batch_size)
    with th.no_grad(): return th.stack([mac.forward(batch, t=t) for t in range(batch.max_seq_length)], 1)


def learner_checks():
    for scene in ('5m_vs_6m', '8m_vs_9m'):
        for optimizer in ('adam', 'rmsprop'):
            th.manual_seed(37)
            mac, _, raw, logger = make_case('baseline', scene, production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', learner='clean_learner', optimizer=optimizer))
            args = copy.deepcopy(mac.args)
            args.mixer_state_refresh_enabled, args.mixer_state_refresh_stage_steps = True, 10
            args.learner = 'mixer_state_refresh_learner'
            batch = cached_batch(raw, args)
            assert th.equal(actor_q(mac, raw), actor_q(mac, batch)), 'Mixer state altered the actor'
            rng = th.get_rng_state()
            plain = CleanLearner(copy.deepcopy(mac), batch.scheme, logger, args)
            th.set_rng_state(rng)
            curr = MixerStateRefreshLearner(copy.deepcopy(mac), batch.scheme, logger, args)
            assert all(th.equal(a, b) for a, b in zip(plain.params, curr.params))
            calls = []
            for model, target in ((curr.mixer, False), (curr.target_mixer, True)):
                model.register_forward_pre_hook(lambda module, inputs, target=target:
                    calls.append((target, inputs[1].detach().clone())))
            plain.train(batch, 1000000, 1)
            curr.train(batch, 1000000, 1)
            assert all(th.equal(a, b) for a, b in zip(plain.params, curr.params)), 'Historical TD/optimizer changed'
            assert any(target and th.equal(states, batch['state']) for target, states in calls)
            assert any(not target and th.equal(states, batch['state'][:, :-1]) for target, states in calls)
            replay_scheme = {k: v for k, v in batch.scheme.items() if k not in {'filled', 'actions_onehot'}}
            replay = ReplayBuffer(replay_scheme, batch.groups, 4, batch.max_seq_length,
                preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
            replay.insert_episode_batch(batch)
            curr._check_batch(replay.sample(2))
            summary = ValueDiagnosticSummary()
            collect_value_diagnostics(curr.mac, curr.mixer, batch, args.gamma, summary)
            assert summary.moments['q_tot'][0] > 0
            damaged = copy.deepcopy(batch)
            damaged['state'][0, 1, 0] += 1
            try: curr._check_batch(damaged)
            except ValueError: pass
            else: raise AssertionError('Raw/held bypass was not detected')
            with tempfile.TemporaryDirectory(prefix='SYNTHETIC-refresh-checkpoint-') as temp:
                curr.save_models(temp)
                curr.load_models(temp)
                bad = copy.deepcopy(curr.refresh_contract)
                curr.refresh_contract['stage_steps'] = 99
                try: curr.load_models(temp)
                except ValueError: pass
                else: raise AssertionError('Changed curriculum silently resumed')
                curr.refresh_contract = bad
            print('PASS:', scene, optimizer, 'same historical actor/TD updates; cached online+target+diagnostics; mixed-stage replay; checkpoint contract')


def main():
    schedule_checks()
    runner_checks()
    learner_checks()


if __name__ == '__main__': main()
