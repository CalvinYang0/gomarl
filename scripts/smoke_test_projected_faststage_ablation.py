#!/usr/bin/env python3
"""Synthetic checks for projected p0, 5M live cutoff and one-time binary freeze."""
import copy
import os
from pathlib import Path
import shlex
import sys
import tempfile
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_projected_faststage_ablation_18jobs as suite
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from smoke_test_gate_score_supervision import equal_state
from smoke_test_no_video_startup import config_for
from smoke_test_soft_onefill_refresh import held_batch
from learners import REGISTRY
from utils.mixer_state_refresh import scheduled_segments


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def plan_checks():
    plans = suite.build_plans(ROOT)
    expected = {(s, v[0], n) for s in suite.SCENES for v in suite.VARIANTS.values() for n in (1, 2, 3)}
    assert len(plans) == 18 and {(p['scene'], p['label'], p['seed']) for p in plans} == expected
    for scene in suite.SCENES:
        reference = {p['seed']: p for p in suite.reference.build_plans(ROOT, scene)}
        for key, (label, initial, full, freeze) in suite.VARIANTS.items():
            for p in suite.build_plans(ROOT, scene, key):
                base = reference[p['seed']]
                x = extras(base)
                x.update(mixer_input_initial_probability=initial, mixer_state_refresh_full_steps=full)
                if freeze:
                    x.update(learner='projected_onefill_freeze_refresh_learner',
                        mixer_soft_gate_freeze_threshold='0.9')
                assert extras(p) == x
                assert p['label'] == label and p['map_name'] == base['map_name']
                assert set(p['sbatch_args']) - set(base['sbatch_args']) == {'--job-name='+p['job_name']}
                ignored = {'EXTRA_ARGS', 'RUN_NAME', 'GROUP_NAME'}
                assert {k: v for k, v in p['exports'].items() if k not in ignored} == {
                    k: v for k, v in base['exports'].items() if k not in ignored}
    validate_config_keys(ROOT, plans)
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite.reference, 'submit_suite') as submit:
        suite.main(['--scene', 'smac_3s5z', '--variant', 'init1_5m_freeze90'])
        assert len(submit.call_args.args[2](ROOT)) == 3
    return plans


def make(plan, temp):
    config = config_for(plan)
    config['clean_model_type'] = plan['exports']['MODEL_TYPE']
    th.manual_seed(99)
    mac, _, raw, logger = make_case('baseline', plan['map_name'], production_shapes=True,
        config_overrides=config)
    args = mac.args
    args.local_results_path, args.unique_token = temp, plan['job_name']
    args.wandb_run_name = 'SYNTHETIC_QA_NOT_EXPERIMENT'
    batch = held_batch(raw, args)
    learner = REGISTRY[args.learner](mac, batch.scheme, logger, args)
    learner.last_image_t = 10000000
    return learner, batch


def check_models(scene, temp):
    plans = {key: suite.build_plans(ROOT, scene, key)[0] for key in suite.VARIANTS}
    p10, b10 = make(plans['init01_8m'], temp)
    m = p10.mixer
    assert not hasattr(m, 'gate_logits') and th.all(m.coefficients() == 0.1)
    s = b10['state']
    assert th.allclose(m.controlled_input(s, m.gate_values(s)), .1*s+.9)
    assert scheduled_segments(p10.args, 5000000, 170) == 67
    assert scheduled_segments(p10.args, 8000000, 170) == 170
    p10.train(b10, 1000000, 1)
    assert m.gate_parameter.grad is not None and th.any(m.coefficients() != .1)
    assert th.all((m.coefficients() >= 0) & (m.coefficients() <= 1))

    plain, plain_batch = make(plans['init1_5m'], temp)
    frozen, batch = make(plans['init1_5m_freeze90'], temp)
    assert th.all(frozen.mixer.coefficients() == 1)
    for learner in (plain, frozen):
        assert [scheduled_segments(learner.args, i*1000000, 170) for i in range(6)] == [1, 7, 28, 61, 109, 170]
        assert scheduled_segments(learner.args, 4999999, 170) == 169
        assert scheduled_segments(learner.args, 0, 170, test_mode=True) == 170
    plain.train(plain_batch, 4999999, 1)
    frozen.train(batch, 4999999, 1)
    assert all(th.equal(x, y) for x, y in zip(plain.params, frozen.params)), 'Pre-freeze TD changed'
    m = frozen.mixer
    assert m.gate_parameter.requires_grad and m.gate_parameter in frozen.optimiser.state
    assert m.freeze_t_env.item() == -1
    with th.no_grad():
        m.gate_parameter.fill_(0.5)
        m.gate_parameter[:3] = th.tensor([.89, .90, .91])
        frozen.target_mixer.gate_parameter.fill_(.99)
    target_before = copy.deepcopy(frozen.target_mixer.base.state_dict())
    agent_before = copy.deepcopy(frozen.mac.agent.state_dict())
    mixer_before = copy.deepcopy(m.base.state_dict())
    frozen.train(batch, 5000003, 2)  # First update can cross 5M by several env steps.
    assert m.freeze_t_env.item() == frozen.target_mixer.freeze_t_env.item() == 5000003
    assert m.coefficients()[:3].tolist() == [0, 0, 1] and m.coefficients().sum() == 1
    assert th.equal(m.coefficients(), frozen.target_mixer.coefficients())
    assert equal_state(target_before, frozen.target_mixer.base.state_dict()), 'Freeze copied unrelated target weights'
    assert not equal_state(agent_before, frozen.mac.agent.state_dict())
    assert not equal_state(mixer_before, m.base.state_dict())
    assert not m.gate_parameter.requires_grad and m.gate_parameter.grad is None
    assert not frozen.target_mixer.gate_parameter.requires_grad and m.gate_parameter not in frozen.optimiser.state
    mask = m.coefficients().clone()
    x = m.controlled_input(batch['state'], m.gate_values(batch['state']))
    assert th.all(x[..., mask == 0] == 1) and th.equal(x[..., mask == 1], batch['state'][..., mask == 1])
    frozen._update_targets()
    frozen.train(batch, 6000000, 3)
    assert th.equal(m.coefficients(), mask) and m.freeze_t_env.item() == 5000003
    plain.train(plain_batch, 5000003, 2)
    assert plain.mixer.gate_parameter.requires_grad and plain.mixer.gate_parameter.grad is not None

    path = Path(temp)/scene
    path.mkdir()
    frozen.save_models(path)
    resumed, replay = make(plans['init1_5m_freeze90'], temp)
    resumed.load_models(path)
    assert not resumed.mixer.gate_parameter.requires_grad
    assert not resumed.target_mixer.gate_parameter.requires_grad
    resumed.train(replay, 7000000, 4)
    assert th.equal(resumed.mixer.coefficients(), mask)
    print('PASS:', scene, 'projected p0=.1, 5M live cutoff, pre-freeze parity, exact threshold, target, TD, resume')


def main():
    th.set_num_threads(1)
    plan_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-projfast-ablation-') as temp:
        for scene in suite.SCENES:
            check_models(scene, temp)
    print('PASS: 18 scoped plans; no simulator, submission or uploads')


if __name__ == '__main__':
    main()
