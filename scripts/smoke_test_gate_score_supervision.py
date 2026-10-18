#!/usr/bin/env python3
"""Exact targets, two-way gradient isolation, optimizer/checkpoint/media safety."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th
import wandb

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from modules.mixers.state_input_control import StateInputControlMixer
from utils.mixer_gate_supervision import gate_score_targets, gate_supervision_loss
from utils.mixer_state_importance import log_state_snapshot
import ozstar_submit_static_gate_score_supervision_12jobs as study
import ozstar_submit_8m9m_ones_gate_regularization_9jobs as common
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def equal_state(a, b):
    if isinstance(a, th.Tensor):
        return th.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def formulas():
    class Mixer(th.nn.Module):
        variant, state_dim = 'static_gate', 2
        def __init__(self):
            super().__init__()
            self.gate_logits = th.nn.Parameter(th.zeros(2))
            self.weight = th.nn.Parameter(th.tensor(1.))
        def conditioned_value(self, q, s, group, mask):
            assert not mask.requires_grad
            return ((s * mask)[..., :1] + 2 * (s * mask)[..., 1:2]) * q[..., :1] * self.weight
    mixer = Mixer()
    q = th.ones(1, 3, 1, requires_grad=True)
    s = th.tensor([[[1., 1.], [3., 1.], [9999., 9999.]]], requires_grad=True)
    valid = th.tensor([[[1.], [1.], [0.]]])
    g = th.tensor([[[1.5], [2.5], [9999.]]], requires_grad=True)
    change = gate_score_targets(mixer, q, s, valid)
    assert change['count'] == 2 and th.equal(change['raw_scores'], th.tensor([.5, 0.]))
    assert th.equal(change['targets'], th.tensor([1., 0.]))
    mc = gate_score_targets(mixer, q, s, valid, criterion='calibration', returns=g)
    assert th.allclose(mc['raw_scores'], th.tensor([-1/17, 0.]))
    assert th.allclose(mc['targets'], th.sigmoid(th.tensor([1/17, 0.])))
    assert th.equal(mc['original_mc_mse'], th.zeros(2))
    assert th.equal(mc['perturbed_mc_mse'], th.tensor([.25, 0.]))
    # Positive replacement benefit is bad and produces a lower keep target.
    bad = gate_score_targets(mixer, q, s, valid, criterion='calibration', returns=g * 0 + 2)
    assert th.allclose(bad['raw_scores'], th.tensor([1/16, 0.]))
    assert bad['targets'][0] < bad['targets'][1]
    scaled = gate_score_targets(mixer, q * 100, s, valid, criterion='calibration', returns=g * 100)
    assert th.allclose(mc['targets'], scaled['targets'])
    assert all(not value.requires_grad for key, value in mc.items() if isinstance(value, th.Tensor))
    targets = change['targets'].clone().requires_grad_(True)
    gate_supervision_loss(mixer.gate_logits, targets).backward()
    assert th.equal(mixer.gate_logits.grad, th.tensor([-.25, .25]))
    assert targets.grad is None and mixer.weight.grad is None and q.grad is None and s.grad is None and g.grad is None
    # Probe soft p even when the hard evaluation threshold would close all fields.
    with th.no_grad():
        mixer.gate_logits.fill_(math.log(.05 / .95))
    assert gate_score_targets(mixer, q, s, valid)['raw_scores'][0] > 0
    assert gate_score_targets(mixer, q, s * 0, valid) is None
    assert gate_score_targets(mixer, q, s, valid * 0) is None
    s2 = th.tensor([[[1., 1.], [3., 1.], [8., 1.], [9999., 9999.]]])
    complete = th.tensor([[[1.], [1.], [0.], [0.]]])
    reference = th.tensor([[[1.], [1.], [1.], [0.]]])
    with th.no_grad():
        mixer.gate_logits.zero_()
    result = gate_score_targets(mixer, th.ones(1, 4, 1), s2, complete,
        criterion='calibration', returns=th.tensor([[[1.5], [2.5], [0.], [0.]]]), reference_valid=reference)
    assert th.allclose(result['raw_scores'], th.tensor([-5/17, 0.]))


def low_level_routing():
    for optimizer in (th.optim.Adam, th.optim.RMSprop):
        args = SimpleNamespace(mixer_input_control='static_gate', state_shape=4, n_agents=2,
            mixing_embed_dim=4, mixer_gate_aux_score='qchange')
        mixer = StateInputControlMixer(args)
        q = th.randn(2, 3, 2, requires_grad=True)
        s = th.randn(2, 3, 4)
        legacy = copy.deepcopy(mixer)
        legacy.gate_aux_score = 'none'
        th.manual_seed(101)
        actual = mixer.values(q, s, 'train')
        th.manual_seed(101)
        expected = legacy.values(q, s, 'train')
        assert th.equal(actual, expected), 'Only gradient routing, not main forward sampling, changes'
        loss = actual.square().mean()
        loss.backward()
        assert mixer.gate_logits.grad is None and q.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mixer.mixers.parameters())
        backbone = list(mixer.mixers.parameters())
        main = optimizer(backbone, lr=.001)
        gate = optimizer([mixer.gate_logits], lr=.001)
        gate_before = mixer.gate_logits.detach().clone()
        main.step()
        assert th.equal(gate_before, mixer.gate_logits)
        old = [p.detach().clone() for p in backbone]
        for p in backbone:
            p.grad = None
        gate.zero_grad()
        gate_supervision_loss(mixer.gate_logits, th.tensor([1., 0., 1., 0.])).backward()
        gate.step()
        assert not th.equal(gate_before, mixer.gate_logits)
        assert all(p.grad is None and th.equal(a, p) for a, p in zip(old, backbone))


def case(map_name, score, coefficient, root):
    th.manual_seed(97)
    mac, _, batch, logger = make_case('baseline', map_name, production_shapes=True,
        config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control='static_gate',
            mixer_gate_aux_score=score, mixer_gate_aux_interval=1, mixer_gate_aux_coef=coefficient,
            slot_importance_samples=8))
    args = mac.args
    tag = '{}_{}_{}'.format(map_name, score, coefficient)
    args.local_results_path, args.unique_token = str(root), 'SYNTHETIC_' + tag
    args.wandb_run_name, args.seed = 'SYNTHETIC_GATE_SCORE_QA_NOT_EXPERIMENT_' + tag, 1
    batch['filled'][1, 3] = 1
    learner = MixerInputLearner(mac, batch.scheme, logger, args)
    return learner, batch, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-dir', type=Path)
    options = parser.parse_args()
    th.set_num_threads(1)
    formulas()
    low_level_routing()
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    references = {(p['scene'], p['seed']): p for p in study.reference_plans(repo)}
    assert len(plans) == len({p['job_name'] for p in plans}) == 12
    for plan in plans:
        extra = extras(plan)
        assert extra.pop('mixer_gate_aux_score') in {'qchange', 'calibration'}
        assert extra.pop('mixer_gate_aux_interval') == '100'
        assert extra.pop('mixer_gate_aux_coef') == '1.0'
        assert extra == extras(references[plan['scene'], plan['seed']])
        assert plan['exports']['T_MAX'] == '10050000'
        assert plan['exports']['GROUP_NAME'] == study.GROUP
        assert plan['exports']['RUN_NAME'] == plan['job_name']
    for scene in ('smac_5m6m', 'smac_8m9m'):
        for label in study.LABELS:
            assert {p['seed'] for p in plans if p['scene'] == scene and p['label'] == label} == {1, 2, 3}
    with tempfile.TemporaryDirectory(prefix='gomarl-gate-score-QA-') as temp:
        root = Path(temp)
        for map_name in ('5m_vs_6m', '8m_vs_9m'):
            for score in ('qchange', 'calibration'):
                a, batch_a, logger = case(map_name, score, 1., root)
                b, batch_b, _ = case(map_name, score, 1000000., root)
                dim = 98 if map_name == '5m_vs_6m' else 179
                assert a.mixer.mixers[0].state_dim == dim and not a.mixer.grouped and not a.mixer.append_mask
                main_ids = {id(p) for group in a.optimiser.param_groups for p in group['params']}
                gate_ids = {id(p) for group in a.gate_optimiser.param_groups for p in group['params']}
                assert main_ids.isdisjoint(gate_ids) and gate_ids == {id(a.mixer.gate_logits)}
                assert len(a.main_params) + 1 == len(a.params)
                th.manual_seed(103)
                a.train(batch_a, 10, 1)
                th.manual_seed(103)
                b.train(batch_b, 10, 1)
                assert th.equal(a.last_losses, b.last_losses)
                assert all(th.equal(x, y) for x, y in zip(a.main_params, b.main_params)), 'Huge auxiliary loss leaked into main update'
                assert a.last_gate_aux_loss is not None and a.mixer.gate_aux_updates.item() == 1
                assert a.mixer.gate_aux_probe_count.item() > 0
                assert a.gate_l1_coef == a.gate_variance_coef == 0
                assert (a.mixer.gate_aux_targets >= 0).all() and (a.mixer.gate_aux_targets <= 1).all()
                # Aux-only step changes gate but not model/actor or their gradients.
                for p in a.params:
                    p.grad = None
                main_before = [p.detach().clone() for p in a.main_params]
                gate_before = a.mixer.gate_logits.detach().clone()
                states, rewards = batch_a['state'][:, :-1], batch_a['reward'][:, :-1]
                terminated = batch_a['terminated'][:, :-1].float()
                valid = batch_a['filled'][:, :-1].float() * batch_a['filled'][:, 1:].float()
                valid[:, 1:] *= (1 - terminated[:, :-1]).cumprod(1)
                a._train_gate(batch_a, states, valid, rewards, terminated, 20)
                assert all(p.grad is None and th.equal(old, p) for old, p in zip(main_before, a.main_params))
                assert not th.equal(gate_before, a.mixer.gate_logits)
                # Save gate media from real dimensions, using SDK Image offline.
                saved = []
                logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, -1, {}
                logger.wandb = SimpleNamespace(id='SYNTHETIC_QA', summary={}, log=lambda *a, **kw: None)
                logger.wandb_module = SimpleNamespace(Image=wandb.Image, save=lambda path, **kw: saved.append(Path(path)))
                snapshot_path = log_state_snapshot(a.mixer, a.args, logger, 20, a.train_steps)
                snapshot = json.loads(snapshot_path.read_text())
                assert snapshot['gate_aux_score'] == score and snapshot['main_td_gate_detached']
                assert snapshot['gate_aux_probe_count'] > 0 and len(snapshot['gate_aux_targets']) == dim
                assert isinstance(logger.wandb_current_data['mixer_input/gate_aux_heatmap'], wandb.Image)
                if options.preview_dir and map_name == '5m_vs_6m':
                    import shutil
                    options.preview_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot_path.with_suffix('.png'), options.preview_dir / (score + '.png'))
                checkpoint = root / (map_name + '_' + score)
                checkpoint.mkdir()
                a.save_models(checkpoint)
                expected_gate = a.mixer.gate_logits.detach().clone()
                expected_optim = copy.deepcopy(a.gate_optimiser.state_dict())
                with th.no_grad():
                    a.mixer.gate_logits.add_(1)
                a.load_models(checkpoint)
                assert th.equal(expected_gate, a.mixer.gate_logits) and equal_state(expected_optim, a.gate_optimiser.state_dict())
                a.gate_aux_score = 'none'
                try:
                    a.load_models(checkpoint)
                    raise AssertionError('Wrong gate criterion must not resume')
                except ValueError:
                    pass
                a.gate_aux_score = score
                # Missing MC or no Q-change signal must skip gate optimizer,
                # including its existing momentum state, while main TD may train.
                gate_before = a.mixer.gate_logits.detach().clone()
                optim_before = copy.deepcopy(a.gate_optimiser.state_dict())
                updates = a.mixer.gate_aux_updates.item()
                a._train_gate(batch_a, states * 0 if score == 'qchange' else states,
                    valid, rewards, terminated * 0, 30)
                assert th.equal(gate_before, a.mixer.gate_logits)
                assert equal_state(optim_before, a.gate_optimiser.state_dict())
                assert a.mixer.gate_aux_updates.item() == updates and a.mixer.gate_aux_probe_count.item() == 0
                assert a.last_gate_aux_loss is None
                a.gate_aux_interval = 100
                a.train_steps = 1
                with patch('learners.mixer_input_learner.gate_score_targets', side_effect=AssertionError('Not due')):
                    a._train_gate(batch_a, states, valid, rewards, terminated, 40)
                print('PASS: {} {}; real dimensions/TD; main update invariant to 1e6 aux scale; aux-only gate; skip/no momentum; media/checkpoint'.format(map_name, score))
        # Legacy static gate retains its TD-trained gate and old checkpoint mode.
        legacy, batch, _ = case('5m_vs_6m', 'none', 1., root)
        assert legacy.gate_optimiser is None and any(p is legacy.mixer.gate_logits for p in legacy.optimiser.param_groups[0]['params'])
        checkpoint = root / 'legacy'
        checkpoint.mkdir()
        legacy.save_models(checkpoint)
        schedule_path = checkpoint / 'mixer_input_schedule.th'
        schedule = th.load(schedule_path, map_location='cpu')
        for key in list(schedule):
            if key.startswith('gate_aux_'):
                schedule.pop(key)
        th.save(schedule, schedule_path)
        legacy.load_models(checkpoint)
        import ozstar_calibration_suite_sync as sync
        import ozstar_clean_except_qmix_variants as storage
        import plot_5m6m_8m9m_qmix_variants_3seeds as plots
        assert study.GROUP in sync.GROUPS and study.GROUP in storage.KEEP_GROUPS
        assert {p['job_name'] for p in plans} <= {p['job_name'] for p in plots.build_plans(repo)}
        calls, retained = [], {}
        def preflight(command, **kwargs):
            if command[0] == 'sbatch':
                assert command[1] == '--test-only' and 'submit' not in calls
                calls.append('preflight')
        def command(command, **kwargs):
            if command == ['id', '-un']:
                return 'SYNTHETIC_USER'
            if command == ['git', 'rev-parse', 'HEAD']:
                return 'SYNTHETIC_COMMIT'
            assert command[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 12
            calls.append('submit')
            return str(94000 + calls.count('submit'))
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch.object(common, 'validate_maps'), \
                 patch.object(common.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(common.submitter, 'guard_and_route', return_value={'logs': root}), \
                 patch.object(common.submitter, 'active_jobs', return_value=retained), \
                 patch.object(common.submitter, 'completed_jobs', return_value={}), \
                 patch.object(common.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(common.submitter, 'run', side_effect=command):
                study.main()
                assert calls.count('preflight') == calls.count('submit') == 12
                record = json.loads(next(root.glob(study.GROUP + '_*.json')).read_text())
                retained.update(record['submitted'])
                calls.clear()
                study.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: exact targets/directions and detached supervision; Adam/RMSprop two-way isolation; '
          'legacy compatibility; 12-job plan, sync/plot coverage and mock Slurm idempotence; no real uploads/jobs')


if __name__ == '__main__':
    main()
