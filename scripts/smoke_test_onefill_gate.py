#!/usr/bin/env python3
"""Synthetic QA only: gate semantics, MC isolation, residual, resume, plans."""
import copy
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from smoke_test_counter_transformer_nine import make_case
from learners.onefill_gate_learner import OneFillGateLearner
from modules.mixers.onefill_gate import OneFillGateMixer, bernoulli_gate_kl
from modules.mixers.state_input_control import StateInputControlMixer
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics
from smoke_test_gate_score_supervision import equal_state
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
import ozstar_submit_5m6m_onefill_gate_15jobs as study
import ozstar_submit_8m9m_ones_gate_regularization_9jobs as common


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def small_mixer(residual=False, layers=1):
    return OneFillGateMixer(SimpleNamespace(state_shape=4, n_agents=2, mixing_embed_dim=8,
        hypernet_layers=layers, hypernet_embed=8, mixer_input_initial_probability=.1,
        mixer_input_gate_temperature=.5, mixer_onefill_residual=residual, mixer_onefill_residual_scale=.1))


def semantics():
    th.manual_seed(31)
    mixer = small_mixer()
    states = th.randn(1, 50000, 4)
    mask = mixer.sample_mask(states, 'train')
    assert set(mask.detach().unique().tolist()) == {0., 1.}
    assert abs(mask.mean().item() - .1) < .004
    actual = mixer.controlled_input(states, mask)
    assert th.equal(actual[mask == 0], th.ones_like(actual[mask == 0]))
    assert th.equal(actual[mask == 1], states[mask == 1])
    mask.sum().backward()
    assert mixer.gate_logits.grad.abs().sum() > 0
    assert not hasattr(mixer, 'threshold')
    assert th.allclose(mixer.sample_mask(states, 'target'), th.full_like(states, .1))
    assert th.equal(mixer.sample_mask(states, 'eval'), mixer.sample_mask(states, 'target'))
    original = copy.deepcopy(mixer)
    mixer.threshold = .999  # Even a stale unrelated attribute cannot change behavior.
    q = th.randn(1, 3, 2)
    assert th.equal(mixer(q, states[:, :3]), original(q, states[:, :3]))
    # Closed fields have exactly zero state-input derivative.
    s = th.randn(1, 3, 4, requires_grad=True)
    m = th.tensor([0., 1., 0., 1.])
    mixer.controlled_input(s, m).sum().backward()
    assert th.equal(s.grad, m.expand_as(s))
    # Old zero-fill behavior is unchanged.
    old = StateInputControlMixer(SimpleNamespace(mixer_input_control='static_gate',
        state_shape=4, n_agents=2, mixing_embed_dim=8))
    assert th.equal(old.controlled_input(s, m), s * m)
    # Exact formula and restoring direction toward KEEP prior 0.1.
    logits = th.tensor([math.log(.3/.7), math.log(.05/.95)], requires_grad=True)
    p = logits.sigmoid()
    expected = (p * (p/.1).log() + (1-p) * ((1-p)/.9).log()).mean()
    kl = bernoulli_gate_kl(logits, .1)
    assert th.allclose(kl, expected)
    kl.backward()
    assert logits.grad[0] > 0 and logits.grad[1] < 0
    at_prior = th.full((4,), math.log(.1/.9), requires_grad=True)
    assert abs(bernoulli_gate_kl(at_prior, .1).item()) < 1e-7


def residual():
    for layers in (1, 2):
        th.manual_seed(71)
        mixer = small_mixer(True, layers)
        q = th.randn(2, 3, 2, requires_grad=True)
        s = th.randn(2, 3, 4)
        baseline = mixer.base(q, th.ones_like(s))
        assert th.equal(mixer.conditioned_value(q, s, th.ones_like(s)), baseline)
        # Move residual heads: full close STILL gives the exact base, but an
        # opened field can now change weights. Bias/value never see state.
        with th.no_grad():
            for network in (mixer.residual_w1, mixer.residual_w2):
                head = network if isinstance(network, th.nn.Linear) else network[-1]
                head.weight.normal_()
                head.bias.normal_()
        closed = mixer.conditioned_value(q, s, th.zeros_like(s))
        assert th.equal(closed, baseline)
        opened = mixer.conditioned_value(q, s, th.ones_like(s))
        assert not th.allclose(opened, baseline)
        first, second = mixer.weight_parameters(s)
        assert (first >= 0).all() and (second >= 0).all()
        gradient = th.autograd.grad(opened.sum(), q)[0]
        assert (gradient >= 0).all(), 'QMIX monotonicity lost'
        # Adding a uniform bias to R cancels under centering.
        before = opened.detach().clone()
        with th.no_grad():
            for network in (mixer.residual_w1, mixer.residual_w2):
                (network if isinstance(network, th.nn.Linear) else network[-1]).bias.add_(10)
        assert th.allclose(mixer.conditioned_value(q, s, th.ones_like(s)), before, atol=1e-5)


def case(map_name, objective, residual_flag, optimizer, root, mc_coef=1., probability=.1):
    th.manual_seed(97)
    mac, _, batch, logger = make_case('baseline', map_name, production_shapes=True,
        config_overrides=dict(clean_model_type='qmix_minimal', learner='onefill_gate_learner',
            mixer_onefill_objective=objective, mixer_onefill_residual=residual_flag,
            mixer_input_initial_probability=probability, mixer_onefill_mc_coef=mc_coef,
            optimizer=optimizer, grad_norm_clip=.01))
    args = mac.args
    args.local_results_path, args.unique_token = str(root), 'SYNTHETIC_ONEFILL'
    args.wandb_run_name, args.seed = 'SYNTHETIC_QA_NOT_EXPERIMENT', 1
    # Complete second episode includes a terminal transition and its next state.
    batch['filled'][1, 3] = 1
    learner = OneFillGateLearner(mac, batch.scheme, logger, args)
    learner.last_image_t = 10
    return learner, batch, logger


def training(root):
    for scene in ('5m_vs_6m', '8m_vs_9m'):
        for optimizer in ('adam', 'rmsprop'):
            for condition in study.CONDITIONS.values():
                objective, res = condition['objective'], condition['residual'] == 'True'
                a, batch, logger = case(scene, objective, res, optimizer, root,
                    probability=float(condition['probability']))
                old_main = [p.detach().clone() for p in a.main_params]
                old_gate = a.mixer.gate_logits.detach().clone()
                if objective == 'kl':
                    assert a._kl_coefficient(th.tensor(10.), th.tensor(2.)) == .5
                    assert math.isclose(a._kl_coefficient(th.tensor(20.), th.tensor(4.)), .5)
                    a.ema_td = a.ema_kl = None
                th.manual_seed(103)
                a.train(batch, 10, 1)
                assert th.isfinite(a.last_losses).all()
                assert any(not th.equal(x, y) for x, y in zip(old_main, a.main_params))
                if not res:
                    assert not th.equal(old_gate, a.mixer.gate_logits)
                assert 'mixer_input/gate_eval_open_count' not in logger.stats
                assert a.target_mixer.gate_logits.grad is None
                # Exercise the production evaluation diagnostics, not only the
                # direct training forward; effective residual weights included.
                summary = ValueDiagnosticSummary()
                collect_value_diagnostics(a.mac, a.mixer, batch, a.args.gamma, summary)
                for name in ('q_tot', 'mixer_w1', 'mixer_w_final', 'mixer_dqtot_dqi', 'q_mc_bias'):
                    assert summary.moments[name][0] > 0
                if objective == 'mc':
                    ids = {id(p) for p in a.main_params}
                    assert id(a.mixer.gate_logits) not in ids
                    assert a.last_gate_aux_loss is not None
                    # MC magnitude cannot affect current main optimizer update,
                    # including shared clipping and optimizer state.
                    b, bbatch, _ = case(scene, objective, False, optimizer, root, mc_coef=1e6)
                    th.manual_seed(103)
                    b.train(bbatch, 10, 1)
                    assert th.equal(a.last_losses, b.last_losses)
                    assert all(th.allclose(x, y, rtol=1e-6, atol=1e-8) for x, y in zip(a.main_params, b.main_params))
                    # Direct MSE graph contains only gate gradients, not actor,
                    # mixer parameters, state, or recorded rewards/returns.
                    for p in a.params:
                        p.grad = None
                    chosen = a._rollout(a.mac, batch)[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
                    states = batch['state'][:, :-1].detach().requires_grad_(True)
                    rewards = batch['reward'][:, :-1].detach().requires_grad_(True)
                    terminated = batch['terminated'][:, :-1].float()
                    valid = batch['filled'][:, :-1].float() * batch['filled'][:, 1:].float()
                    valid[:, 1:] *= (1-terminated[:, :-1]).cumprod(1)
                    loss, count = a._mc_gate_loss(chosen, states, rewards, terminated, valid)
                    assert count == 3
                    loss.backward()
                    assert a.mixer.gate_logits.grad is not None
                    assert all(p.grad is None for p in a.main_params)
                    assert states.grad is None and rewards.grad is None
                    # Padding and incomplete-episode fields cannot affect MC.
                    changed = states.detach().clone()
                    changed[0] = 1e8
                    changed[1, 3:] = -1e8
                    th.manual_seed(901)
                    x, _ = a._mc_gate_loss(chosen, states, rewards, terminated, valid)
                    th.manual_seed(901)
                    y, _ = a._mc_gate_loss(chosen, changed, rewards, terminated, valid)
                    assert th.equal(x, y)
                    # Skip incomplete returns WITHOUT moving existing momentum.
                    no_complete = copy.deepcopy(batch)
                    no_complete['terminated'].zero_()
                    gate = a.mixer.gate_logits.detach().clone()
                    opt = copy.deepcopy(a.gate_optimiser.state_dict())
                    a.train(no_complete, 11, 2)
                    assert a.last_gate_aux_loss is None
                    assert th.equal(gate, a.mixer.gate_logits)
                    assert equal_state(opt, a.gate_optimiser.state_dict())
                # Full checkpoint incl. lagged targets, EMA and separate gate.
                path = root / '{}_{}_{}_{}'.format(scene, optimizer, objective, res)
                path.mkdir(exist_ok=True)
                a.save_models(path)
                fresh, fresh_batch, _ = case(scene, objective, res, optimizer, root,
                    probability=float(condition['probability']))
                fresh.load_models(path)
                assert equal_state(a.mixer.state_dict(), fresh.mixer.state_dict())
                assert equal_state(a.target_mixer.state_dict(), fresh.target_mixer.state_dict())
                assert all(th.equal(x, y) for x, y in zip(a.mac.parameters(), fresh.mac.parameters()))
                assert all(th.equal(x, y) for x, y in zip(a.target_mac.parameters(), fresh.target_mac.parameters()))
                assert equal_state(a.optimiser.state_dict(), fresh.optimiser.state_dict())
                assert a.ema_td == fresh.ema_td and a.ema_kl == fresh.ema_kl
                if objective == 'mc':
                    assert equal_state(a.gate_optimiser.state_dict(), fresh.gate_optimiser.state_dict())
                bad = copy.deepcopy(fresh)
                bad.contract['fill_value'] = 0
                before = [p.detach().clone() for p in bad.params]
                try:
                    bad.load_models(path)
                except ValueError as error:
                    assert 'contract differs' in str(error)
                else:
                    raise AssertionError('Incompatible checkpoint accepted')
                assert all(th.equal(x, y) for x, y in zip(before, bad.params))
                # Seeded next update agrees after restore; target lag and
                # adaptive coefficient must not silently reset on load.
                th.manual_seed(631)
                a.train(batch, 12, 3)
                th.manual_seed(631)
                fresh.train(batch, 12, 3)
                assert all(th.allclose(x, y, rtol=1e-6, atol=1e-8) for x, y in zip(a.params, fresh.params))
                empty = copy.deepcopy(batch)
                empty['filled'].zero_()
                before = copy.deepcopy(a.optimiser.state_dict())
                a.train(empty, 12, 3)
                assert equal_state(before, a.optimiser.state_dict())
                print('PASS:', scene, optimizer, objective, 'residual=' + str(res), flush=True)
    # Native image and JSON path without any cloud call.
    a, _, logger = case('5m_vs_6m', 'td', False, 'adam', root)
    logger.use_wandb = True
    logger.wandb_current_t, logger.wandb_current_data = 10, {}
    logger.wandb_module = SimpleNamespace(Image=lambda path: ('IMAGE', path), save=lambda *args, **kwargs: None)
    a._snapshot(10)
    assert 'mixer_input/onefill_probability_heatmap' in logger.wandb_current_data
    snapshot = json.loads(next((root / 'mixer_onefill_gate').rglob('*.json')).read_text())
    assert len(snapshot['keep_probability']) == 98 and snapshot['contract']['fill_value'] == 1
    assert 'threshold' not in snapshot['contract']


def submission(root):
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    assert len(plans) == len({p['job_name'] for p in plans}) == 15
    for plan in plans:
        extra = extras(plan)
        assert extra['learner'] == 'onefill_gate_learner'
        assert not any('threshold' in key for key in extra)
        assert extra['mixer_onefill_kl_prior'] == '0.1'
        assert plan['exports']['T_MAX'] == '10050000'
    assert {p['scene'] for p in plans} == {'smac_5m6m'}
    assert {p['map_name'] for p in plans} == {'5m_vs_6m'}
    for label in study.LABELS:
        assert {p['seed'] for p in plans if p['label'] == label} == {1, 2, 3}
    # All sbatch preflights before actual submission; exact names reused.
    retained = {p['job_name']: str(10000+i) for i, p in enumerate(plans[:6])}
    calls = []
    def preflight(command, **kwargs):
        if command[0] == 'sbatch':
            assert command[1] == '--test-only' and 'submit' not in calls
            calls.append('preflight')
    def command(args, **kwargs):
        if args == ['id', '-un']:
            return 'SYNTHETIC_USER'
        if args == ['git', 'rev-parse', 'HEAD']:
            return 'SYNTHETIC_COMMIT'
        assert args[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 9
        calls.append('submit')
        return str(20000+calls.count('submit'))
    cwd = Path.cwd()
    try:
        with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=str(root)), \
             patch.object(study, 'validate_installed_map'), \
             patch.object(common.submitter, 'home_quota_free_gib', return_value=2.1), \
             patch.object(common.submitter, 'guard_and_route', return_value={'logs': root}), \
             patch.object(common.submitter, 'active_jobs', return_value=retained), \
             patch.object(common.submitter, 'completed_jobs', return_value={}), \
             patch.object(common.submitter.subprocess, 'run', side_effect=preflight), \
             patch.object(common.submitter, 'run', side_effect=command):
            study.main()
            assert calls.count('submit') == calls.count('preflight') == 9
            manifest = json.loads(next(root.glob(study.GROUP + '_*.json')).read_text())
            retained.update(manifest['submitted'])
            calls.clear()
            study.main()
            assert not calls
    finally:
        os.chdir(cwd)
    import plot_5m6m_8m9m_qmix_variants_3seeds as charts
    import ozstar_calibration_suite_sync as sync
    import ozstar_clean_except_qmix_variants as cleanup
    assert study.GROUP in sync.GROUPS and study.GROUP in cleanup.KEEP_GROUPS
    assert len(charts.build_plans(repo)) == 93
    assert set(study.LABELS).issubset(charts.SCENE_MODELS['smac_5m6m'])
    assert set(study.LABELS).isdisjoint(charts.SCENE_MODELS['smac_8m9m'])
    print('PASS: 15 exact 5m6m jobs; no 8m9m jobs/thresholds; plotting/sync inventories; mock preflight/idempotence; no real submission')


def main():
    th.set_num_threads(1)
    semantics()
    residual()
    with tempfile.TemporaryDirectory(prefix='gomarl-onefill-SYNTHETIC-QA-') as temp:
        root = Path(temp)
        training(root)
        submission(root)
    print('PASS: one-fill sampling, KL direction, MC-only gradients, missing-return skip, residual fallback/monotonicity, both optimizers/maps, checkpoint/media')


if __name__ == '__main__':
    main()
