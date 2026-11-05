#!/usr/bin/env python3
"""Exact parent-config transfer, real startup/learners on synthetic map data."""
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
import ozstar_submit_fastrefresh_init1_three_maps_18jobs as suite
import smoke_test_no_video_startup as startup
import smoke_test_mixer_state_refresh as legacy
from smoke_test_counter_transformer_nine import make_case
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from learners import REGISTRY
from utils.mixer_state_refresh import begin, extra_scheme


def extras(plan):
    return {key: yaml.safe_load(value) for key, value in
        (x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS'])) if '.' not in key}


def held_batch(raw, args):
    scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
    scheme.update(extra_scheme(args, args.state_shape, 7))
    batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length,
        preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
    batch.update({k: v.clone() for k, v in raw.data.transition_data.items() if k != 'actions_onehot'}, mark_filled=False)
    for idx in range(raw.batch_size):
        cache = begin(args, 7, 1, (idx+1)*2000000)
        for t in range(raw.max_seq_length):
            batch.update(cache.apply(dict(state=[raw['state'][idx, t].numpy()]), t),
                bs=[idx], ts=t, mark_filled=False)
    return batch


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 18
    for scene, name, duration in suite.MAPS:
        selected = suite.build_plans(ROOT, scene)
        assert len(selected) == 6 and {p['map_name'] for p in selected} == {name}
        for variant, label in suite.VARIANTS.items():
            scoped = suite.build_plans(ROOT, scene, variant)
            assert {p['seed'] for p in scoped} == {1, 2, 3} and {p['label'] for p in scoped} == {label}
            for p in scoped:
                parent = next(q for q in suite.parent_plans(ROOT, variant) if q['seed'] == p['seed'])
                assert p['exports']['EXTRA_ARGS'] == parent['exports']['EXTRA_ARGS'], 'Parent controls drifted'
                assert {k: v for k, v in p['exports'].items() if k not in {'GROUP_NAME', 'MAP_NAME', 'RUN_NAME'}} == {
                    k: v for k, v in parent['exports'].items() if k not in {'GROUP_NAME', 'MAP_NAME', 'RUN_NAME'}}
                assert p['exports']['MODEL_TYPE'] == parent['exports']['MODEL_TYPE'] == 'qmix_minimal'
                assert p['exports']['USE_CUDA'] == 'False'
                assert p['exports']['T_MAX'] == '10050000'
                assert '--cpus-per-task=28' in p['sbatch_args'] and '--mem=32G' in p['sbatch_args']
                assert '--time='+duration in p['sbatch_args']
                assert not any('gpu' in arg.lower() for arg in p['sbatch_args'])
                assert p['job_name'] != parent['job_name']
                x = extras(p)
                assert x['disable_video_media'] and not x['test_battle_videos']
                if variant == 'init1':
                    assert x['learner'] == 'projected_onefill_gate_learner'
                    assert x['mixer_input_initial_probability'] == 1 and not x['mixer_state_refresh_enabled']
                else:
                    assert x['learner'] == 'soft_onefill_refresh_learner'
                    assert x['mixer_input_initial_probability'] == .1 and x['mixer_state_refresh_enabled']
                    assert x['mixer_state_refresh_schedule'] == 'faster_stages'
                    assert x['mixer_state_refresh_full_steps'] == 8000000 and x['mixer_state_refresh_eval_live']
    log = logging.getLogger('SYNTHETIC-parent-transfer-startup')
    for plan in plans:
        config = startup.config_for(plan)
        assert config['test_visualizations_required'] and config['test_policy_importance']
        args = SN(**startup.args_sanity_check(config, log))
        with patch('utils.visualization_requirements.check_video_dependencies',
                side_effect=AssertionError('No encoder check allowed')), \
                patch.dict(startup.r_REGISTRY, {args.runner: lambda **kw:
                    (_ for _ in ()).throw(startup.RunnerReached())}):
            try: startup.run_sequential(args, startup.Logger(log))
            except startup.RunnerReached: pass
            else: raise AssertionError('Actual startup did not reach runner')
    from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
    validate_config_keys(ROOT, plans)
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite.shared, 'submit_suite') as submit:
        suite.main(['--scene', 'smac_corridor', '--variant', 'init1'])
        assert len(submit.call_args.args[2](ROOT)) == 3
    with patch.dict(os.environ, SUBMIT='YES'), patch.object(suite.subprocess, 'run') as quota, \
            patch.object(suite.shared, 'parse_home_quota', return_value={'free_gib': 0, 'free_inodes': 0}), \
            patch.object(suite.shared, 'submit_suite', side_effect=AssertionError('Guard bypass')):
        quota.return_value.stdout = 'SYNTHETIC quota'
        try: suite.main([])
        except RuntimeError: pass
        else: raise AssertionError('Insufficient quota accepted')
    with patch.object(suite.shared, 'validate_maps') as validate:
        suite.validate_maps()
        assert [call.args[0] for call in validate.call_args_list] == [scene for scene, _, _ in suite.MAPS]
    print('PASS: exact18 / 3maps x 2parents x 3seeds; unchanged parent controls; CPU/no-video startup; corridor48h/others24h; guards')
    return plans


def model_check(plan, optimizer, temp):
    x = extras(plan)
    x.update(clean_model_type='qmix_minimal', optimizer=optimizer)
    th.manual_seed(71)
    mac, _, raw, logger = make_case('baseline', plan['map_name'], production_shapes=True, config_overrides=x)
    args = mac.args
    args.local_results_path, args.unique_token = temp, 'SYNTHETIC_'+plan['scene']+plan['label']+optimizer
    args.wandb_run_name, args.seed = 'SYNTHETIC_QA_NOT_EXPERIMENT', plan['seed']
    batch = held_batch(raw, args) if args.mixer_state_refresh_enabled else raw
    assert th.equal(legacy.actor_q(mac, raw), legacy.actor_q(mac, batch))
    assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
    learner = REGISTRY[args.learner](mac, batch.scheme, logger, args)
    learner.last_image_t = 10000000
    m = learner.mixer
    initial = m.coefficients().detach().clone()
    assert th.allclose(initial, th.full_like(initial, x['mixer_input_initial_probability']))
    assert not hasattr(m, 'gate_frozen')
    if args.mixer_state_refresh_enabled:
        assert hasattr(m, 'gate_logits') and learner.refresh_contract['schedule_mode'] == 'faster_stages'
        learner._check_batch(batch)
        # Exact parent soft TD on the SAME held inputs; refresh must not alter the loss.
        plain_args = copy.deepcopy(args)
        plain_args.learner, plain_args.mixer_state_refresh_enabled = 'soft_onefill_gate_learner', False
        rng = th.get_rng_state()
        plain = REGISTRY[plain_args.learner](copy.deepcopy(mac), batch.scheme, logger, plain_args)
        th.set_rng_state(rng)
        plain.mixer.load_state_dict(learner.mixer.state_dict())
        plain.target_mixer.load_state_dict(learner.target_mixer.state_dict())
        plain.last_image_t = 10000000
    else:
        assert not hasattr(m, 'gate_logits') and th.equal(initial, th.ones_like(initial))
        assert th.equal(m.controlled_input(raw['state'], m.gate_values(raw['state'])), raw['state'])
        plain = None
    actor_before = [p.detach().clone() for p in mac.parameters()]
    for step in (1000000, 5000000, 9000000):
        learner.train(batch, step, 1)
        if plain is not None:
            plain.train(batch, step, 1)
            assert all(th.equal(a, b) for a, b in zip(learner.params, plain.params)), 'Parent TD changed'
        assert th.isfinite(learner.last_losses).all()
        assert th.all((m.coefficients() >= 0) & (m.coefficients() <= 1))
    assert any(not th.equal(a, b) for a, b in zip(actor_before, mac.parameters()))
    assert not th.equal(initial, m.coefficients()), 'Gate never learned'
    learner._update_targets()
    assert th.equal(m.coefficients(), learner.target_mixer.coefficients())
    print('PASS:', plan['scene'], plan['label'], optimizer, 'real map dimensions/TD, old actor, gate trainable past5M, target copy')


def main():
    th.set_num_threads(1)
    plans = plan_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-two-parent-transfer-') as temp:
        for plan in plans:
            if plan['seed'] == 1:
                for optimizer in ('adam', 'rmsprop'):
                    model_check(plan, optimizer, temp)
    print('PASS: no real jobs, SC2 launch, videos, uploads or deletion')


if __name__ == '__main__':
    main()
