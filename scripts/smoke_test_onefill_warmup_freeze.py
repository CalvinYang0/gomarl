#!/usr/bin/env python3
"""Synthetic gate/TD/replay/resume and mock submission; no real jobs or uploads."""
import copy
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from unittest.mock import patch

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_3s5z_onefill_warmup_9jobs as suite
import ozstar_submit_3s5z_refresh_and_warmup_18jobs as combined
from smoke_test_counter_transformer_nine import make_case
from smoke_test_soft_onefill_refresh import held_batch
from smoke_test_gate_score_supervision import equal_state
from learners import REGISTRY


def extras(p):
    return {k: yaml.safe_load(v) for k, v in (x.split('=', 1) for x in shlex.split(p['exports']['EXTRA_ARGS'])) if '.' not in k}


def make(p, optimizer, temp):
    x = extras(p)
    x.update(optimizer=optimizer, clean_model_type='qmix_minimal')
    th.manual_seed(29)
    mac, _, raw, logger = make_case('baseline', '3s5z_vs_3s6z', production_shapes=True, config_overrides=x)
    args = mac.args
    args.local_results_path, args.unique_token, args.wandb_run_name = temp, p['label'], 'SYNTHETIC_QA_NOT_EXPERIMENT'
    args.seed = p['seed']
    batch = held_batch(raw, args) if args.mixer_state_refresh_enabled else raw
    learner = REGISTRY[args.learner](mac, batch.scheme, logger, args)
    learner.last_image_t = 10000000
    return learner, batch, logger


def projected_checks(p, optimizer, temp):
    learner, batch, logger = make(p, optimizer, temp)
    m = learner.mixer
    assert not hasattr(m, 'gate_logits') and th.equal(m.coefficients(), th.ones_like(m.coefficients()))
    s = batch['state']
    assert th.equal(m.controlled_input(s, m.gate_values(s)), s)
    # Exact boundary p=1 still admits an inward gradient, unlike infinite logits.
    learner.optimiser.zero_grad(set_to_none=True)
    m.coefficients().sum().backward()
    assert th.equal(m.gate_parameter.grad, th.ones_like(m.gate_parameter))
    learner.optimiser.step()
    learner._after_main_optimizer_step()
    assert th.all(m.coefficients() < 1) and th.all(m.coefficients() >= 0)
    with th.no_grad():
        m.gate_parameter[0], m.gate_parameter[1] = -5, 5
    learner._after_main_optimizer_step()
    assert m.gate_parameter[0] == 0 and m.gate_parameter[1] == 1
    learner._update_targets()
    assert th.equal(m.coefficients(), learner.target_mixer.coefficients())
    before = copy.deepcopy(learner.mac.agent.state_dict())
    learner.train(batch, 1000000, 1)
    assert not equal_state(before, learner.mac.agent.state_dict())
    assert th.all((m.coefficients() >= 0) & (m.coefficients() <= 1))
    assert th.isfinite(learner.last_losses).all()
    path = Path(temp)/('projected-'+optimizer)
    path.mkdir()
    learner.save_models(path)
    other, _, _ = make(p, optimizer, temp)
    other.load_models(path)
    assert equal_state(learner.mixer.state_dict(), other.mixer.state_dict())
    assert equal_state(learner.optimiser.state_dict(), other.optimiser.state_dict())
    print('PASS: projected exact1', optimizer, 'real inward gradients, projection before target copy, TD/actor/resume')


def freeze_checks(p, optimizer, temp):
    learner, batch, logger = make(p, optimizer, temp)
    args = learner.args
    # Before cutoff, exact old soft-onefill updates on the same (held) batch.
    plain_args = copy.deepcopy(args)
    plain_args.learner, plain_args.mixer_state_refresh_enabled = 'soft_onefill_gate_learner', False
    state = th.get_rng_state()
    plain = REGISTRY['soft_onefill_gate_learner'](copy.deepcopy(learner.mac), batch.scheme, logger, plain_args)
    plain.mixer.load_state_dict({k: v for k, v in learner.mixer.state_dict().items() if k not in {'gate_frozen', 'frozen_mask', 'freeze_t_env'}})
    plain.target_mixer.load_state_dict(plain.mixer.state_dict())
    th.set_rng_state(state)
    plain.last_image_t = 10000000
    plain.train(batch, 4999999, 1)
    learner.train(batch, 4999999, 1)
    assert all(th.equal(a, b) for a, b in zip(plain.params, learner.params)), 'Warmup TD changed'
    assert learner.mixer.gate_logits in learner.optimiser.state
    pre = Path(temp)/(p['label']+optimizer+'-pre')
    pre.mkdir()
    learner.save_models(pre)
    resumed, _, _ = make(p, optimizer, temp)
    resumed.load_models(pre)
    assert resumed.mixer.gate_logits.requires_grad and not resumed.mixer.gate_frozen.item()
    # Use values straddling threshold, including an EXACT floating threshold equality.
    values = learner.mixer.coefficients().detach().clone()
    values[:] = .2
    values[:3] = th.tensor([.89, .9, .91])
    with th.no_grad():
        learner.mixer.gate_logits.copy_(th.logit(values))
        learner.target_mixer.gate_logits.fill_(-3)
    weights_before = copy.deepcopy(learner.target_mixer.base.state_dict())
    gate_before = learner.mixer.gate_logits.detach().clone()
    learner._before_training_update(4999999)
    assert not learner.mixer.gate_frozen.item()
    # Exercise equality exactly without depending on logit/sigmoid roundoff.
    with patch.object(learner.mixer, 'coefficients', return_value=values):
        learner._before_training_update(5000000)
    expected = (values > .9).float()
    assert expected[:3].tolist() == [0, 0, 1]
    assert th.equal(learner.mixer.coefficients(), expected)
    assert th.equal(learner.target_mixer.coefficients(), expected)
    assert equal_state(weights_before, learner.target_mixer.base.state_dict()), 'Freeze prematurely copied target weights'
    assert learner.mixer.freeze_t_env.item() == 5000000
    assert not learner.mixer.gate_logits.requires_grad and learner.mixer.gate_logits.grad is None
    assert learner.mixer.gate_logits not in learner.optimiser.state
    before = copy.deepcopy(learner.mac.agent.state_dict())
    # No repeated image rendering needed in the optimizer test.
    learner.last_image_t = 10000000
    for step in (5000000, 6000000, 10000000):
        learner.train(batch, step, 2)
    assert th.equal(gate_before, learner.mixer.gate_logits)
    assert th.equal(learner.mixer.coefficients(), expected) and th.equal(learner.target_mixer.coefficients(), expected)
    assert not equal_state(before, learner.mac.agent.state_dict()), 'Other networks accidentally frozen'
    assert learner.mixer.gate_logits not in learner.optimiser.state
    learner._update_targets()
    assert th.equal(learner.target_mixer.coefficients(), expected)
    s = batch['state']
    x = learner.mixer.controlled_input(s, learner.mixer.gate_values(s))
    assert th.equal(x[..., expected == 0], th.ones_like(x[..., expected == 0]))
    assert th.equal(x[..., expected == 1], s[..., expected == 1])
    post = Path(temp)/(p['label']+optimizer+'-post')
    post.mkdir()
    learner.save_models(post)
    resumed.load_models(post)
    assert not resumed.mixer.gate_logits.requires_grad
    assert resumed.mixer.freeze_t_env.item() == 5000000
    resumed.last_image_t = 10000000
    resumed.train(batch, 7000000, 3)
    assert th.equal(resumed.mixer.coefficients(), expected)
    assert resumed.mixer.gate_logits not in resumed.optimiser.state
    # Snapshot must show actual 0/1 mask, not stale warmup sigmoid values.
    resumed._snapshot(7000000)
    files = list((Path(temp)/'mixer_soft_onefill_gate'/args.unique_token).glob('*.json'))
    record = json.loads(max(files, key=lambda q: q.stat().st_mtime_ns).read_text())
    assert record['gate_value'] == expected.tolist() and record['gate_freeze']['frozen']
    resumed.contract['freeze_threshold'] = .8
    try: resumed.load_models(post)
    except ValueError: pass
    else: raise AssertionError('Changed freeze contract accepted')
    if args.mixer_state_refresh_enabled:
        learner._check_batch(batch)
    # The requested rule allows all gates to close. Never add an implicit top-k.
    for probability in (.1, .95):
        edge, edge_batch, _ = make(p, optimizer, temp)
        with th.no_grad():
            edge.mixer.gate_logits.fill_(th.logit(th.tensor(probability)).item())
        edge._before_training_update(5000000)
        expected_value = float(probability > .9)
        assert th.all(edge.mixer.coefficients() == expected_value)
        edge.last_image_t = 10000000
        edge.train(edge_batch, 5000000, 1)
        assert th.isfinite(edge.last_losses).all()
    print('PASS:', p['label'], optimizer, 'unchanged warmup TD, exact cutoff/threshold, frozen optimizer, target/actor/replay/resume/heatmap')


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == 9 and len(combined.build_plans(ROOT)) == 18
    for variant in suite.MODES:
        assert {p['seed'] for p in suite.build_plans(ROOT, variant)} == {1, 2, 3}
    for p in plans:
        x = extras(p)
        assert p['scene'] == 'smac_3s5z' and p['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert x['mixer_soft_gate_freeze_steps'] == 5000000 and x['mixer_soft_gate_freeze_threshold'] == .9
        assert x['disable_video_media'] and not x['test_battle_videos']
    from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
    validate_config_keys(ROOT, plans)
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(combined.refresh, 'main') as a, patch.object(combined.warmup, 'main') as b:
        combined.main([])
        a.assert_called_once_with([])
        b.assert_called_once_with([])
    with patch.dict(os.environ, SUBMIT='YES'), patch.object(suite.subprocess, 'run') as quota, \
            patch.object(suite.reference, 'parse_home_quota', return_value={'free_gib': 0, 'free_inodes': 0}), \
            patch.object(suite.reference, 'submit_suite', side_effect=AssertionError('Guard bypass')):
        quota.return_value.stdout = 'SYNTHETIC quota'
        try: suite.main([])
        except RuntimeError: pass
        else: raise AssertionError('Quota guard bypassed')
    print('PASS: new9 and combined18 exact identities, old cohort retained, guard enforced; no jobs submitted')
    return plans


def main():
    th.set_num_threads(1)
    plans = plan_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-gate-freeze-') as temp:
        for optimizer in ('adam', 'rmsprop'):
            for p in plans:
                if p['seed'] != 1:
                    continue
                if extras(p)['learner'] == 'projected_onefill_gate_learner':
                    projected_checks(p, optimizer, temp)
                else:
                    freeze_checks(p, optimizer, temp)


if __name__ == '__main__':
    main()
