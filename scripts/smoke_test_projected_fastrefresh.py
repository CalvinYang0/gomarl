#!/usr/bin/env python3
"""Four-map projected gate + refresh integration; synthetic data, no jobs/cloud."""
import copy
import logging
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_projected_fastrefresh_four_maps_12jobs as suite
import smoke_test_refresh_live_schedules as schedules
import smoke_test_mixer_state_refresh as legacy
import smoke_test_no_video_startup as startup
from smoke_test_counter_transformer_nine import make_case
from smoke_test_gate_score_supervision import equal_state
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY
from utils.mixer_state_refresh import begin, extra_scheme, scheduled_segments


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 12
    for scene, name, _, _ in suite.MAPS:
        selected = suite.build_plans(ROOT, scene)
        assert len(selected) == 3 and {p['seed'] for p in selected} == {1, 2, 3}
        assert {p['map_name'] for p in selected} == {name}
    for p in plans:
        x = startup.config_for(p)
        assert x['learner'] == 'projected_onefill_refresh_learner'
        assert x['mixer_input_initial_probability'] == 1
        assert x['mixer_onefill_objective'] == 'td' and not x['mixer_onefill_residual']
        assert x['mixer_state_refresh_enabled'] and x['mixer_state_refresh_eval_live']
        assert x['mixer_state_refresh_schedule'] == 'faster_stages'
        assert x['mixer_state_refresh_full_steps'] == 8000000
        assert p['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert p['exports']['T_MAX'] == '10050000'
        assert '--time=1-00:00:00' in p['sbatch_args'] and '--mem=32G' in p['sbatch_args']
        assert x['disable_video_media'] and not x['test_battle_videos']
        assert x['test_visualizations_required'] and x['test_policy_importance']
        args = SN(**startup.args_sanity_check(x, logging.getLogger('SYNTHETIC-combined-startup')))
        # Execute the actual startup preflight which failed in the previous batch.
        with patch('utils.visualization_requirements.check_video_dependencies',
                   side_effect=AssertionError('No video encoder check permitted')), \
                patch.dict(startup.r_REGISTRY, {args.runner: lambda **kw:
                    (_ for _ in ()).throw(startup.RunnerReached())}):
            try: startup.run_sequential(args, startup.Logger(logging.getLogger('SYNTHETIC-startup')))
            except startup.RunnerReached: pass
            else: raise AssertionError('Production startup never reached runner construction')
    from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
    validate_config_keys(ROOT, plans)
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite, 'submit_suite') as submit:
        suite.main(['--scene', 'smac_corridor'])
        assert len(submit.call_args.args[2](ROOT)) == 3
    with patch.dict(os.environ, SUBMIT='YES'), patch.object(suite.subprocess, 'run') as quota, \
            patch.object(suite, 'parse_home_quota', return_value={'free_gib': 0, 'free_inodes': 0}), \
            patch.object(suite, 'submit_suite', side_effect=AssertionError('Guard bypass')):
        quota.return_value.stdout = 'SYNTHETIC quota'
        try: suite.main([])
        except RuntimeError: pass
        else: raise AssertionError('Insufficient quota accepted')
    # No SC2 binary or map download: check installed-map identity and asset guard using fixtures.
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-map-assets-') as temp:
        root = Path(temp)
        (root/'Versions/Base1').mkdir(parents=True)
        (root/'Versions/Base1/SC2_x64').touch()
        (root/'Maps').mkdir()
        for _, name, _, _ in suite.MAPS:
            (root/'Maps'/(name+'.SC2Map')).touch()
        with patch.dict(os.environ, SC2PATH=temp):
            suite.validate_maps()
            (root/'Maps/corridor.SC2Map').unlink()
            try: suite.validate_maps()
            except RuntimeError: pass
            else: raise AssertionError('Missing map accepted')
    print('PASS: 12 fresh plans, 4 maps x 3 seeds, 24h/32G, actual no-video startup, quota/map guards; no submission')
    return plans


def model_check(p, optimizer, temp):
    overrides = {k: yaml.safe_load(v) for k, v in
        (x.split('=', 1) for x in shlex.split(p['exports']['EXTRA_ARGS'])) if '.' not in k}
    overrides.update(clean_model_type='qmix_minimal', optimizer=optimizer)
    th.manual_seed(91)
    mac, _, raw, logger = make_case('baseline', p['map_name'], production_shapes=True, config_overrides=overrides)
    args = mac.args
    args.local_results_path, args.unique_token = temp, 'SYNTHETIC_'+p['scene']+optimizer
    args.wandb_run_name, args.seed = 'SYNTHETIC_QA_NOT_EXPERIMENT', p['seed']
    # A full four-step episode, keeping the real map's actor/state feature dimensions.
    args.mixer_state_refresh_episode_limit = 4
    scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
    scheme.update(extra_scheme(args, args.state_shape, 4))
    preprocessing = {'actions': ('actions_onehot', [OneHot(args.n_actions)])}
    batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length, preprocess=preprocessing)
    batch.update({k: v.clone() for k, v in raw.data.transition_data.items() if k != 'actions_onehot'}, mark_filled=False)
    for idx, collected in enumerate((0, 8000000)):
        cache = begin(args, 4, 1, collected)
        for t in range(raw.max_seq_length):
            batch.update(cache.apply(dict(state=[raw['state'][idx, t].numpy()]), t), bs=[idx], ts=t, mark_filled=False)
    batch['filled'][1] = 1
    batch['terminated'][1] = 0
    batch['terminated'][1, 3] = 1
    assert th.equal(batch['state'][0], raw['state'][0, 0].expand_as(batch['state'][0]))
    assert th.equal(batch['state'][1], raw['state'][1]), 'Full-live terminal state was held'
    assert th.equal(legacy.actor_q(mac, raw), legacy.actor_q(mac, batch)), 'Actor observation was frozen'
    plain_args = copy.deepcopy(args)
    plain_args.learner, plain_args.mixer_state_refresh_enabled = 'projected_onefill_gate_learner', False
    rng = th.get_rng_state()
    plain = REGISTRY[plain_args.learner](copy.deepcopy(mac), batch.scheme, logger, plain_args)
    th.set_rng_state(rng)
    learner = REGISTRY[args.learner](copy.deepcopy(mac), batch.scheme, logger, args)
    assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
    assert learner.refresh_contract['version'] == 2
    m = learner.mixer
    assert not hasattr(m, 'gate_logits') and not hasattr(m, 'gate_frozen')
    assert th.equal(m.coefficients(), th.ones_like(m.coefficients()))
    assert th.equal(m.controlled_input(batch['state'], m.gate_values(batch['state'])), batch['state'])
    # All FOUR generated parameter paths see held state AFTER one-fill interpolation.
    with th.no_grad(): m.gate_parameter.copy_(th.linspace(0, 1, m.state_dim))
    expected = m.controlled_input(batch['state'], m.gate_values(batch['state'])).reshape(-1, m.state_dim)
    seen, hooks = {}, []
    for name in ('hyper_w1', 'hyper_w_final', 'hyper_b1', 'value'):
        hooks.append(getattr(m.base, name).register_forward_pre_hook(
            lambda module, inputs, key=name: seen.update({key: inputs[0].detach().clone()})))
    m(th.rand(batch.batch_size, batch.max_seq_length, args.n_agents), batch['state'])
    for hook in hooks: hook.remove()
    assert len(seen) == 4 and all(th.equal(x, expected) for x in seen.values())
    with th.no_grad(): m.gate_parameter.fill_(1)
    assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params))
    plain.last_image_t = learner.last_image_t = 10000000
    # Same projected TD on identical held batches: compare before/at/after the old freeze boundary.
    for t_env in (1000000, 5000000, 9000000):
        plain.train(batch, t_env, 1)
        learner.train(batch, t_env, 1)
        assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params)), 'TD/optimizer changed'
        assert m.gate_parameter.requires_grad and m.gate_parameter.grad is not None
        assert th.isfinite(learner.last_losses).all()
        assert th.all((m.coefficients() >= 0) & (m.coefficients() <= 1))
    assert th.any(m.coefficients() < 1), 'Gate failed to learn inward from exactly one'
    with th.no_grad(): m.gate_parameter[:2] = th.tensor([-5., 5.])
    learner._after_main_optimizer_step()
    assert m.gate_parameter[0] == 0 and m.gate_parameter[1] == 1
    learner._update_targets()
    assert th.equal(m.coefficients(), learner.target_mixer.coefficients())
    replay = ReplayBuffer(scheme, batch.groups, 4, raw.max_seq_length, preprocess=preprocessing)
    replay.insert_episode_batch(batch)
    learner._check_batch(replay.sample(2))
    corrupt = copy.deepcopy(batch)
    corrupt['state'][0, 1, 0] += 1
    try: learner._check_batch(corrupt)
    except ValueError: pass
    else: raise AssertionError('Corrupt held state accepted')
    path = Path(temp)/(p['scene']+optimizer)
    path.mkdir()
    learner.save_models(path)
    before = copy.deepcopy(learner.mixer.state_dict())
    learner.load_models(path)
    assert equal_state(before, learner.mixer.state_dict())
    learner.refresh_contract['eval_live'] = False
    try: learner.load_models(path)
    except ValueError: pass
    else: raise AssertionError('Changed refresh contract accepted')
    print('PASS:', p['scene'], optimizer, 'exact1, all4 held-state paths, unchanged projected TD/actor, trainable past5M, replay/target/resume')


def main():
    th.set_num_threads(1)
    plans = plan_checks()
    # Reuse real EpisodeRunner + ParallelRunner fixtures with the new learner opt-in.
    old_options = schedules.options
    def projected_options(*args, **kwargs):
        result = old_options(*args, **kwargs)
        result.learner = 'projected_onefill_refresh_learner'
        return result
    with patch.object(schedules, 'options', projected_options):
        schedules.schedule_checks()
        schedules.runner_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-projected-fastrefresh-') as temp:
        for p in plans:
            if p['seed'] == 1:
                for optimizer in ('adam', 'rmsprop'):
                    model_check(p, optimizer, temp)
    print('PASS: projected init1 + faststages8M integration; no SC2 launch, jobs, videos or uploads')


if __name__ == '__main__':
    main()
