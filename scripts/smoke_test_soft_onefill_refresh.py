#!/usr/bin/env python3
"""Synthetic model/runner/replay and mock-scheduler QA; no SC2/cloud/jobs."""
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

import torch as th
import yaml

import ozstar_submit_soft_onefill_refresh_three_maps_18jobs as suite
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from smoke_test_gate_score_supervision import equal_state
import smoke_test_mixer_state_refresh as refresh_qa
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY
from modules.mixers.soft_onefill_gate import SoftOneFillGateMixer
from utils.mixer_state_refresh import begin, extra_scheme, segments

ROOT = Path(__file__).resolve().parents[1]


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 18
    references = {p['seed']: p for p in suite.reference.build_plans(ROOT)}
    for scene, name, _, _ in suite.MAPS:
        for label in suite.LABELS:
            selected = [p for p in plans if p['scene'] == scene and p['label'] == label]
            assert {p['seed'] for p in selected} == {1, 2, 3}
            assert all(p['map_name'] == name for p in selected)
    for plan in plans:
        original = references[plan['seed']]
        x, old = extras(plan), extras(original)
        refresh = plan['label'] == suite.LABELS[1]
        assert x['learner'] == ('soft_onefill_refresh_learner' if refresh else 'soft_onefill_gate_learner')
        assert x['mixer_state_refresh_enabled'] == str(refresh)
        assert x['mixer_state_refresh_stage_steps'] == '2000000' and x['qmix_condition_input'] == 'state'
        changed = {'learner', 'qmix_condition_input', 'mixer_state_refresh_enabled', 'mixer_state_refresh_stage_steps'}
        assert {k: v for k, v in x.items() if k not in changed} == {k: v for k, v in old.items() if k not in changed}
        for key in original['exports']:
            if key not in {'MAP_NAME', 'GROUP_NAME', 'RUN_NAME', 'EXTRA_ARGS'}:
                assert plan['exports'][key] == original['exports'][key], key
        assert plan['exports']['MODEL_TYPE'] == 'qmix_minimal' and plan['exports']['T_MAX'] == '10050000'
        assert '--time=1-00:00:00' in plan['sbatch_args'] and '--mem=32G' in plan['sbatch_args']
    validate_config_keys(ROOT, plans)
    print('PASS: exact three maps/two variants/three seeds, unchanged soft-onefill settings and old head; 10M/24h')
    return plans


def held_batch(raw, args):
    # Small synthetic episode length, production map observation/state dimensions.
    scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
    scheme.update(extra_scheme(args, args.state_shape, 7))
    batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length,
        preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
    batch.update({k: v.clone() for k, v in raw.data.transition_data.items() if k != 'actions_onehot'}, mark_filled=False)
    for idx in range(raw.batch_size):
        cache = begin(args, 7, 1, (idx+1)*2000000)
        for t in range(raw.max_seq_length):
            data = cache.apply(dict(state=[raw['state'][idx, t].numpy()]), t)
            batch.update(data, bs=[idx], ts=t, mark_filled=False)
    return batch


def model_checks(plans):
    th.set_num_threads(1)
    for stage, expected in ((0, 1), (1999999, 1), (2000000, 2), (4000000, 3),
                            (8000000, 5), (10000000, 6)):
        assert segments(stage, 2000000, 120) == expected
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-soft-refresh-') as temp:
        for _, name, agents, _ in suite.MAPS:
            for optimizer in ('adam', 'rmsprop'):
                plan = next(p for p in plans if p['map_name'] == name and p['seed'] == 1 and p['label'] == suite.LABELS[1])
                overrides = {k: yaml.safe_load(v) for k, v in extras(plan).items() if '.' not in k}
                overrides.update(clean_model_type='qmix_minimal', optimizer=optimizer)
                th.manual_seed(149)
                mac, _, raw, logger = make_case('baseline', name, production_shapes=True, config_overrides=overrides)
                args = mac.args
                args.local_results_path, args.unique_token = temp, 'SYNTHETIC_'+name+optimizer
                args.wandb_run_name = 'SYNTHETIC_QA_NOT_EXPERIMENT'
                args.seed = 1
                batch = held_batch(raw, args)
                assert th.equal(raw['obs'], batch['obs'])
                assert not th.equal(raw['state'], batch['state'])
                assert th.equal(refresh_qa.actor_q(mac, raw), refresh_qa.actor_q(mac, batch))
                initial_actor = copy.deepcopy(mac.agent.state_dict())
                rng = th.get_rng_state().clone()
                plain_args = copy.deepcopy(args)
                plain_args.learner, plain_args.mixer_state_refresh_enabled = 'soft_onefill_gate_learner', False
                plain = REGISTRY[plain_args.learner](copy.deepcopy(mac), batch.scheme, logger, plain_args)
                th.set_rng_state(rng)
                learner = REGISTRY['soft_onefill_refresh_learner'](copy.deepcopy(mac), batch.scheme, logger, args)
                assert equal_state(initial_actor, learner.mac.agent.state_dict())
                assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params))
                assert isinstance(learner.mixer, SoftOneFillGateMixer) and learner.mixer.temperature is None
                assert learner.gate_optimiser is None and learner.objective == 'td'
                assert args.n_agents == agents and len(learner.mac.agent.fixed_head) == 3
                assert isinstance(learner.mac.agent.fixed_head[1], th.nn.ELU)
                assert learner.mac.agent.hyper_out_w is None
                th.testing.assert_close(learner.mixer.gate_logits.sigmoid(), th.full_like(learner.mixer.gate_logits, .1))
                # All four hypernetwork paths, online and target, must see gated HELD state.
                seen = []
                for target, mixer in ((False, learner.mixer), (True, learner.target_mixer)):
                    for head in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value'):
                        getattr(mixer.base, head).register_forward_pre_hook(
                            lambda module, inputs, t=target, n=head: seen.append((t, n, inputs[0].detach().clone())))
                plain.last_image_t = learner.last_image_t = 0
                gate_before = learner.mixer.gate_logits.detach().clone()
                with patch('torch.rand_like', side_effect=AssertionError('Hard mask sampled')):
                    plain.train(batch, 8000000, 1)
                    learner.train(batch, 8000000, 1)
                assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params)), 'Combo changed original soft TD/optimizer'
                assert th.equal(plain.last_losses, learner.last_losses)
                assert th.isfinite(learner.last_losses).all() and not th.equal(gate_before, learner.mixer.gate_logits)
                assert not equal_state(initial_actor, learner.mac.agent.state_dict())
                assert all(p.grad is None for p in learner.target_mixer.parameters())
                assert {(t, n) for t, n, _ in seen} == {(t, n) for t in (True, False)
                    for n in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value')}
                for target, _, x in seen:
                    # First forward probabilities are .1 on both copies; recomputed gates, not held logits.
                    s = batch['state'] if target else batch['state'][:, :-1]
                    expected = (.1*s + .9).reshape(-1, s.shape[-1])
                    th.testing.assert_close(x, expected)
                # Mixed-stage replay keeps its collected refresh rate, not current 8M stage.
                scheme = {k: v for k, v in batch.scheme.items() if k not in {'filled', 'actions_onehot'}}
                replay = ReplayBuffer(scheme, batch.groups, 4, batch.max_seq_length,
                    preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
                replay.insert_episode_batch(batch)
                learner._check_batch(replay.sample(2))
                damaged = copy.deepcopy(batch)
                damaged['state'][0, 1, 0] += 1
                try: learner._check_batch(damaged)
                except ValueError: pass
                else: raise AssertionError('Held/raw corruption accepted')
                # Target updates and resume must retain both soft-gate and refresh contracts.
                plain.train(batch, 8000001, 200)
                learner.train(batch, 8000001, 200)
                assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params))
                assert equal_state(learner.mixer.state_dict(), learner.target_mixer.state_dict())
                path = Path(temp)/(name+optimizer)
                path.mkdir()
                learner.save_models(path)
                learner.mac.init_hidden(batch.batch_size)
                resumed = REGISTRY['soft_onefill_refresh_learner'](copy.deepcopy(learner.mac), batch.scheme, logger, args)
                resumed.load_models(path)
                assert equal_state(learner.optimiser.state_dict(), resumed.optimiser.state_dict())
                assert equal_state(learner.mixer.state_dict(), resumed.mixer.state_dict())
                resumed.refresh_contract['stage_steps'] = 500000
                try: resumed.load_models(path)
                except ValueError: pass
                else: raise AssertionError('Changed refresh checkpoint accepted')
                print('PASS:', name, optimizer, 'same exact soft TD update, live actor, held/gated W+b online+target, gate gradients, replay, resume')
    options = refresh_qa.options
    with patch.object(refresh_qa, 'options', side_effect=lambda **kw: options(learner='soft_onefill_refresh_learner', **kw)):
        refresh_qa.runner_checks()


def asset_checks():
    import smac.env.starcraft2.maps.smac_maps as installed
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-soft-refresh-assets-') as temp:
        root = Path(temp)
        binary = root/'Versions/Base000/SC2_x64'
        binary.parent.mkdir(parents=True)
        binary.write_text('SYNTHETIC placeholder')
        (root/'Maps').mkdir()
        registry = {}
        for _, name, agents, enemies in suite.MAPS:
            registry[name] = dict(n_agents=agents, n_enemies=enemies)
            (root/'Maps'/(name+'.SC2Map')).write_text('SYNTHETIC map')
        with patch.dict(os.environ, SC2PATH=temp), patch.object(installed, 'get_smac_map_registry', return_value=registry):
            suite.validate_maps()
            (root/'Maps/corridor.SC2Map').unlink()
            try: suite.validate_maps()
            except RuntimeError: pass
            else: raise AssertionError('Missing map accepted')


def scheduler_checks():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-soft-refresh-submit-') as temp:
        logs, calls = Path(temp), []
        plans = suite.build_plans(ROOT)
        active = {plans[0]['job_name']: 'SYNTHETIC_ACTIVE'}
        completed = {plans[1]['job_name']: 'SYNTHETIC_COMPLETED'}
        def run(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 16
            assert kwargs['env']['GROUP_NAME'] == suite.GROUP
            calls.append('submit')
            return str(99000 + calls.count('submit'))
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
                    patch.object(suite, 'validate_maps') as validate, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=active), \
                    patch.object(base, 'completed_jobs', return_value=completed), \
                    patch.object(base, 'run', side_effect=run), \
                    patch.object(subprocess, 'run', side_effect=subprocess_run):
                suite.main()
                validate.assert_called_once()
                assert calls == ['preflight']*16 + ['submit']*16
                record = json.loads(next(logs.glob(suite.GROUP+'_*.json')).read_text())
                assert len(record['submitted']) == 16 and len(record['retained']) == 2
                active.update(record['submitted'])
                calls.clear()
                suite.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: preview, byte+inode guard, map assets, all preflights before submission, receipts and active/completed idempotence (mocked)')


def main():
    plans = plan_checks()
    model_checks(plans)
    asset_checks()
    scheduler_checks()


if __name__ == '__main__':
    main()
