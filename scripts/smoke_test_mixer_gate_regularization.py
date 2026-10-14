#!/usr/bin/env python3
"""Exact gate-loss gradients and real 8m9m shapes/media; no SC2 or cloud."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch as th
import wandb

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from utils.mixer_gate_regularization import gate_variance_coefficients, gate_penalties


def formulas():
    class Toy(th.nn.Module):
        variant, state_dim = 'static_gate', 2
        def __init__(self):
            super().__init__()
            self.gate_logits = th.nn.Parameter(th.zeros(2))
            self.register_buffer('gate_variance_difference', th.tensor([1., 4.]))
        def conditioned_value(self, q, s, group, mask):
            return (s * mask).sum(-1, keepdim=True) * q[..., :1]
    mixer = Toy()
    q = th.ones(1, 3, 1, requires_grad=True)
    states = th.tensor([[[1., 2.], [3., 2.], [999., 999.]]], requires_grad=True)
    valid = th.tensor([[[1.], [1.], [0.]]])
    delta, before, after, count = gate_variance_coefficients(mixer, q, states, valid)
    assert count == 2 and th.equal(delta, th.tensor([.25, 0.]))
    assert th.equal(before - after, delta) and not delta.requires_grad
    assert gate_variance_coefficients(mixer, q, states, valid * 0) is None
    one = valid.clone()
    one[:, 1] = 0
    assert gate_variance_coefficients(mixer, q, states, one) is None
    l1, variance = gate_penalties(mixer, .01, .01)
    assert l1.item() == .5 and variance.item() == 1.25
    l1_grad = th.autograd.grad(l1, mixer.gate_logits, retain_graph=True)[0]
    variance_grad = th.autograd.grad(variance, mixer.gate_logits)[0]
    assert th.equal(l1_grad, th.tensor([.125, .125]))
    assert th.equal(variance_grad, th.tensor([.125, .5]))  # Larger variance => stronger closing gradient.
    assert q.grad is None and states.grad is None
    mixer.gate_variance_difference.fill_(-5)
    assert gate_penalties(mixer, 0, .01)[1].item() == 0  # Never reward opening a gate with negative gain.


def main():
    th.set_num_threads(1)
    formulas()
    with tempfile.TemporaryDirectory(prefix='gomarl-gate-QA-') as temp:
        for label, variant, l1_coef, var_coef in (
                ('ones', 'ones', 0., 0.), ('l1', 'static_gate', .01, 0.),
                ('variance', 'static_gate', 0., .01)):
            th.manual_seed(81)
            mac, _, batch, logger = make_case('baseline', '8m_vs_9m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=variant,
                    mixer_gate_l1_coef=l1_coef, mixer_gate_variance_coef=var_coef,
                    mixer_gate_variance_interval=1, slot_importance_samples=8))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_' + label
            args.wandb_run_name, args.seed = 'SYNTHETIC_GATE_QA_' + label, 1
            saved = []
            logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, -1, {}
            logger.wandb = SimpleNamespace(id='SYNTHETIC_QA', summary={}, log=lambda *a, **kw: None)
            logger.wandb_module = SimpleNamespace(Image=wandb.Image, save=lambda path, **kw: saved.append(Path(path)))
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            mixer = learner.mixer
            assert (args.n_agents, args.n_actions, args.state_shape) == (8, 15, 179)
            assert mac.agent.hyper_out_w is None and mac.agent.fixed_head is not None
            assert len(mixer.mixers) == 1 and mixer.mixers[0].state_dim == 179 and not mixer.append_mask
            if label == 'ones':
                assert mixer.gate_logits is None
                q = th.randn(batch.batch_size, 1, 8, requires_grad=True)
                s = batch['state'][:, :1].clone()
                for mode in ('train', 'target', 'eval'):
                    value = mixer.values(q, s, mode)
                    assert th.equal(value, mixer.values(q, s + 1234, mode))
                    assert th.equal(mixer.controlled_input(s, mixer.masks(s, mode)[0]), th.ones_like(s))
                assert th.count_nonzero(th.autograd.grad(value.sum(), q)[0]) > 0
                actor_before = learner._rollout(mac, batch).detach()
                original_state = batch['state'].clone()
                batch['state'].add_(100)
                assert th.equal(actor_before, learner._rollout(mac, batch).detach())
                batch['state'].copy_(original_state)
            else:
                assert th.allclose(mixer.gate_logits.sigmoid(), th.full((179,), .1))
                if var_coef:
                    mixer.gate_variance_difference.fill_(1)
                a, b = gate_penalties(mixer, l1_coef, var_coef)
                grad = th.autograd.grad(a + b, list(mac.parameters()) + list(mixer.parameters()), allow_unused=True)
                params = list(mac.parameters()) + list(mixer.parameters())
                for parameter, gradient in zip(params, grad):
                    if parameter is mixer.gate_logits:
                        assert gradient is not None and (gradient > 0).all()
                    else:
                        assert gradient is None or not th.count_nonzero(gradient)
            learner.train(batch, 10, 1)
            assert learner.last_losses.numel() == 1 and th.isfinite(learner.last_losses).all()
            assert th.isfinite(learner.last_gate_penalties).all()
            record = json.loads(saved[-1].read_text())
            assert record['mixer_condition_dim'] == 179
            assert record['gate_l1_coef'] == l1_coef and record['gate_variance_coef'] == var_coef
            assert isinstance(logger.wandb_current_data['mixer_input/partition_probability_heatmap'], wandb.Image)
            if label != 'ones':
                p, mask = record['masks_or_probability']
                assert mask == [float(x > .1) for x in p]
            if var_coef:
                assert record['gate_variance_probe_count'] >= 2
                assert th.allclose(mixer.gate_variance_difference, mixer.gate_original_variance - mixer.gate_masked_variance)
                assert learner.last_gate_penalties[0] == 0
            if l1_coef:
                assert learner.last_gate_penalties[1] == 0
            checkpoint = Path(temp) / ('checkpoint_' + label)
            checkpoint.mkdir()
            learner.save_models(checkpoint)
            learner.load_models(checkpoint)
            learner.gate_l1_coef += 1
            try:
                learner.load_models(checkpoint)
                raise AssertionError('Different penalty settings must not resume the same run')
            except ValueError:
                pass
            print('PASS: 8m9m {} — ordinary actor; 179-D condition; TD; gate-only penalties; media; checkpoint'.format(label))
    print('PASS: population variance, fixed Q, valid probes, larger variance => larger closing gradient, no L1/variance mixing')


if __name__ == '__main__':
    main()
