#!/usr/bin/env python3
"""Synthetic schedule/real runner/TD/replay/checkpoint checks; no cloud or jobs."""
import copy
import os
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch
import sys

import numpy as np
import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_3s5z_soft_refresh_live_9jobs as suite
import smoke_test_mixer_state_refresh as legacy
from smoke_test_counter_transformer_nine import make_case
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY
from utils.mixer_state_refresh import begin, validate, extra_scheme, scheduled_segments, source_times
from utils.logging import Logger


def options(mode='linear', **kw):
    args = legacy.options(learner='soft_onefill_refresh_learner',
        mixer_state_refresh_stage_steps=2, mixer_state_refresh_full_steps=8,
        mixer_state_refresh_schedule=mode, mixer_state_refresh_eval_live=True)
    vars(args).update(kw)
    return args


def schedule_checks():
    for mode in ('linear', 'faster_stages', 'larger_jumps'):
        args = options(mode)
        validate(args)
        for limit in (1, 2, 7, 150):
            values = [scheduled_segments(args, t, limit) for t in range(11)]
            assert values[0] == 1 and values == sorted(values)
            assert all(1 <= k <= limit for k in values)
            assert all(scheduled_segments(args, t, limit, True) == limit for t in range(11))
            if mode != 'linear':
                assert values[8:] == [limit]*3
        cache = begin(args, 7, 1, 0, test_mode=True)
        for t in range(8):
            data = cache.apply(dict(state=[np.array([t], dtype=np.float32)], obs=[[t+100]]), t)
            assert data['state'][0][0] == t and data['mixer_state_source_t'] == [(t,)]
            assert data['obs'] == [[t+100]]
        # A schedule is frozen at reset, even if args change mid-episode.
        cache = begin(args, 7, 1, 0)
        args.mixer_state_refresh_eval_live = False
        args.mixer_state_refresh_schedule = 'larger_jumps'
        args.mixer_state_refresh_full_steps = 2
        for t in range(8):
            assert cache.apply(dict(state=[np.array([t])]), t)['state'][0][0] == 0
    for mode, expected in (('faster_stages', [1, 10, 38, 84, 150]),
                           ('larger_jumps', [1, 4, 13, 43, 150])):
        args = options(mode, mixer_state_refresh_stage_steps=2000000, mixer_state_refresh_full_steps=8000000)
        assert [scheduled_segments(args, t*2000000, 150) for t in range(5)] == expected
    for change in (dict(mixer_state_refresh_schedule='typo'), dict(mixer_state_refresh_eval_live='False'),
                   dict(mixer_state_refresh_full_steps=0), dict(mixer_state_refresh_full_steps=7)):
        args = options('larger_jumps', **change)
        try: validate(args)
        except ValueError: pass
        else: raise AssertionError('Invalid configuration accepted')
    legacy.schedule_checks()
    print('PASS: legacy schedule unchanged; new schedules monotonic, bounded, live at H; eval live incl t=L')


def runner_checks():
    import logging
    logger = Logger(logging.getLogger('SYNTHETIC-refresh-live'))
    for parallel in (False, True):
        for mode in ('linear', 'faster_stages', 'larger_jumps'):
            args = options(mode, env='SYNTHETIC_refresh', env_args={}, batch_size_run=2 if parallel else 1,
                device='cpu', test_nepisode=2, runner_log_interval=1000, seed=1)
            with patch.dict(legacy.ENVS, SYNTHETIC_refresh=legacy.FakeEnv):
                if parallel:
                    with patch.object(legacy.ParallelRunner, '_start_workers', legacy.fake_workers):
                        runner = legacy.ParallelRunner(args, logger)
                else:
                    runner = legacy.EpisodeRunner(args, logger)
            scheme = dict(state=dict(vshape=4), obs=dict(vshape=3, group='agents'),
                actions=dict(vshape=(1,), group='agents', dtype=th.long),
                avail_actions=dict(vshape=3, group='agents', dtype=th.int),
                reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8))
            scheme.update(extra_scheme(args, 4, 7))
            runner.setup(scheme, {'agents': 2}, {'actions': ('actions_onehot', [OneHot(3)])}, legacy.FakeMAC())
            for t_env, test in ((0, False), (0, True), (4, False), (8, False), (8, True)):
                runner.t_env = t_env
                batch = runner.run(test_mode=test)
                if test: assert runner.t_env == t_env
                count = scheduled_segments(args, t_env, 7, test)
                live = test or (mode != 'linear' and count == 7)
                for idx in range(args.batch_size_run):
                    length = 3+2*idx if parallel else 3
                    sources = th.tensor(source_times(7, count, length+1, live_terminal=live))
                    assert th.equal(batch['state'][idx, :length+1, 0], sources.float()+100*idx)
                    assert th.equal(batch['obs'][idx, :length+1, 0, 0], th.arange(length+1).float()+100*idx)
    print('PASS: both production runners; test live at t_env=0; train schedule preserved; actor obs live')


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == 9 and {p['map_name'] for p in plans} == {'3s5z_vs_3s6z'}
    for variant, (label, mode) in suite.MODES.items():
        selected = suite.build_plans(ROOT, variant)
        assert {p['seed'] for p in selected} == {1, 2, 3}
        for p in selected:
            x = dict(k.split('=', 1) for k in shlex.split(p['exports']['EXTRA_ARGS']))
            assert p['label'] == label and x['mixer_state_refresh_schedule'] == mode
            assert x['mixer_state_refresh_eval_live'] == 'True' and x['test_battle_videos'] == 'False'
            assert x['disable_video_media'] == 'True' and p['exports']['MODEL_TYPE'] == 'qmix_minimal'
            assert '--time=1-00:00:00' in p['sbatch_args'] and p['exports']['T_MAX'] == '10050000'
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite.reference, 'submit_suite') as submit:
        suite.main(['--variant', 'larger_jumps'])
        assert len(submit.call_args.args[2](ROOT)) == 3
    with patch.dict(os.environ, SUBMIT='YES'), patch.object(suite.subprocess, 'run') as quota, \
            patch.object(suite.reference, 'parse_home_quota', return_value={'free_gib': 0, 'free_inodes': 0}), \
            patch.object(suite.reference, 'submit_suite', side_effect=AssertionError('Guard bypass')):
        quota.return_value.stdout = 'synthetic quota'
        try: suite.main([])
        except RuntimeError: pass
        else: raise AssertionError('Insufficient quota accepted')
    from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
    validate_config_keys(ROOT, plans)
    print('PASS: 9 distinct fresh plans / optional 3, same head/settings, no videos, quota guard, no actual jobs')
    return plans


def model_checks(plans):
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-refresh-live-') as temp:
        for label, _ in suite.MODES.values():
            p = next(p for p in plans if p['label'] == label)
            overrides = {k: yaml.safe_load(v) for k, v in
                (x.split('=', 1) for x in shlex.split(p['exports']['EXTRA_ARGS'])) if '.' not in k}
            for optimizer in ('adam', 'rmsprop'):
                th.manual_seed(91)
                overrides.update(clean_model_type='qmix_minimal', optimizer=optimizer)
                mac, _, raw, logger = make_case('baseline', '3s5z_vs_3s6z', production_shapes=True, config_overrides=overrides)
                args = mac.args
                args.local_results_path, args.unique_token = temp, 'SYNTHETIC_fixture'
                args.wandb_run_name = 'SYNTHETIC_QA_NOT_EXPERIMENT'
                # Small episode with a genuine terminal t=L; keep production feature dimensions.
                args.mixer_state_refresh_episode_limit = 4
                scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
                scheme.update(extra_scheme(args, args.state_shape, 4))
                batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length,
                    preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
                batch.update({k: v.clone() for k, v in raw.data.transition_data.items() if k != 'actions_onehot'}, mark_filled=False)
                # Old+current-stage episodes coexist; no current-step reinterpretation.
                for idx, collected in enumerate((0, 8000000)):
                    cache = begin(args, 4, 1, collected)
                    for t in range(raw.max_seq_length):
                        batch.update(cache.apply(dict(state=[raw['state'][idx, t].numpy()]), t), bs=[idx], ts=t, mark_filled=False)
                assert th.equal(legacy.actor_q(mac, raw), legacy.actor_q(mac, batch))
                if overrides['mixer_state_refresh_schedule'] != 'linear':
                    assert th.equal(batch['state'][1, 4], raw['state'][1, 4])
                    # Audit a valid terminal t=L too, not just padded values.
                    batch['filled'][1] = 1
                    batch['terminated'][1] = 0
                    batch['terminated'][1, 3] = 1
                plain_args = copy.deepcopy(args)
                plain_args.learner, plain_args.mixer_state_refresh_enabled = 'soft_onefill_gate_learner', False
                rng = th.get_rng_state()
                plain = REGISTRY['soft_onefill_gate_learner'](copy.deepcopy(mac), batch.scheme, logger, plain_args)
                th.set_rng_state(rng)
                learner = REGISTRY['soft_onefill_refresh_learner'](copy.deepcopy(mac), batch.scheme, logger, args)
                assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
                assert learner.refresh_contract['version'] == 2
                assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params))
                plain.last_image_t = learner.last_image_t = 10000000
                plain.train(batch, 9000000, 1)
                learner.train(batch, 9000000, 1)
                assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params)), 'TD/optimizer changed'
                replay = ReplayBuffer(scheme, batch.groups, 4, raw.max_seq_length,
                    preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
                replay.insert_episode_batch(batch)
                learner._check_batch(replay.sample(2))
                corrupt = copy.deepcopy(batch)
                corrupt['state'][0, 1, 0] += 1
                try: learner._check_batch(corrupt)
                except ValueError: pass
                else: raise AssertionError('Corrupt held state accepted')
                path = Path(temp)/(label+optimizer)
                path.mkdir()
                learner.save_models(path)
                learner.load_models(path)
                learner.refresh_contract['eval_live'] = False
                try: learner.load_models(path)
                except ValueError: pass
                else: raise AssertionError('Changed evaluation contract accepted')
                print('PASS:', label, optimizer, 'same soft TD, old actor, mixed replay, terminal live, checkpoint contract')


def main():
    schedule_checks()
    runner_checks()
    model_checks(plan_checks())


if __name__ == '__main__':
    main()
