#!/usr/bin/env python3
"""Real 5m6m learner/gradient/media tests; exact variance formula; no SC2/cloud."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch as th
import wandb

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from utils.mixer_state_importance import score_state_slot_variance


def formulas():
    class Mixer:
        state_dim = 2
        def masks(self, s, mode):
            return s.new_ones(1, 2)
        def conditioned_value(self, q, s, group, mask):
            return 2 * (s * mask)[..., :1] + (s * mask)[..., 1:2] + q[..., :1]
    q = th.zeros(1, 3, 1)
    states = th.tensor([[[0., 0.], [2., 0.], [10000., 10000.]]])
    valid = th.tensor([[[1.], [1.], [0.]]])
    difference, count, original, masked = score_state_slot_variance(Mixer(), q, states, valid)
    assert count == 2 and th.equal(difference, th.tensor([4., 0.]))
    assert th.equal(original, th.tensor([4., 4.])) and th.equal(masked, th.tensor([0., 4.]))
    # Population variance, not sample variance, absolute delta, or MSE to G.
    assert score_state_slot_variance(Mixer(), q, states, valid * 0) is None
    one = valid.clone()
    one[:, 1] = 0
    assert score_state_slot_variance(Mixer(), q, states, one) is None
    correlated = th.tensor([[[0., 0.], [1., -2.], [0., 0.]]])
    negative, *_ = score_state_slot_variance(Mixer(), q, correlated, valid)
    assert th.equal(negative, th.tensor([-1., -1.]))  # Never clip or sigmoid.
    class Grouped(Mixer):
        def masks(self, s, mode):
            return s.new_tensor([[1., 0.], [0., 1.]])
        def conditioned_value(self, q, s, group, mask):
            # Out-of-group changes cannot contribute to that branch's score.
            return super().conditioned_value(q, s, group, mask)
    grouped, *_ = score_state_slot_variance(Grouped(), q, correlated, valid)
    assert th.equal(grouped, th.tensor([1., 1.]))


def main():
    th.set_num_threads(1)
    formulas()
    with tempfile.TemporaryDirectory(prefix='gomarl-clean-ablation-QA-') as temp:
        for variant, isolated, append in (('group_qchange', True, True),
                                          ('group_qchange', False, False),
                                          ('group_variance', False, True)):
            th.manual_seed(71)
            mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=variant,
                    mixer_input_detach_unimportant=isolated, mixer_input_append_mask=append,
                    slot_group_interval=1, slot_importance_samples=8))
            args = mac.args
            tag = '{}_detach{}_append{}'.format(variant, isolated, append)
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_' + tag
            args.wandb_run_name, args.seed = 'SYNTHETIC_QA_NOT_EXPERIMENT_' + tag, 1
            saved = []
            logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, -1, {}
            logger.wandb = SimpleNamespace(id='SYNTHETIC_QA', summary={}, log=lambda *a, **kw: None)
            logger.wandb_module = SimpleNamespace(Image=wandb.Image, save=lambda path, **kw: saved.append(Path(path)))
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            mixer = learner.mixer
            assert args.state_shape == 98 and not mixer.important_slots.any()
            assert mac.agent.hyper_out_w is None and mac.agent.fixed_head is not None
            assert all(core.state_dim == (196 if append else 98) for core in mixer.mixers)
            mixer.important_slots[:2] = True
            state = batch['state'][:, :1].detach().clone().requires_grad_(True)
            fixed_q = th.randn(batch.batch_size, 1, 5)
            baseline = mixer.conditioned_value(fixed_q, state, 0, mixer.masks(state)[0])
            altered = state.detach().clone()
            altered[..., 2:] += 1000
            assert th.equal(baseline.detach(), mixer.conditioned_value(fixed_q, altered, 0, mixer.masks(altered)[0]))
            gradient = th.autograd.grad(baseline.sum(), state, retain_graph=True)[0]
            assert th.count_nonzero(gradient[..., 2:]) == 0
            first_layers = [mixer.mixers[0].hyper_w1[0], mixer.mixers[0].hyper_w_final[0],
                            mixer.mixers[0].hyper_b1, mixer.mixers[0].value[0]]
            columns = th.autograd.grad(baseline.sum(), [layer.weight for layer in first_layers])
            assert all(th.count_nonzero(g[:, 2:98]) == 0 for g in columns)
            if not append:
                assert th.equal(mixer.controlled_input(state, mixer.masks(state)[0]), state * mixer.masks(state)[0])
            values = learner._rollout(mac, batch)
            chosen = values[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            joint = mixer.values(chosen, batch['state'][:, :-1], 'train', detach_unimportant=isolated)
            gradients = th.autograd.grad(joint[..., 1].square().mean(), list(mac.parameters()), allow_unused=True)
            no_actor_gradient = all(g is None or th.count_nonzero(g) == 0 for g in gradients)
            assert no_actor_gradient == isolated
            learner.train(batch, 10, 1)
            assert learner.last_losses.numel() == 2 and th.isfinite(learner.last_losses).all()
            assert mixer.grouping_updates.item() == 1
            assert th.equal(mixer.important_slots, learner.target_mixer.important_slots)
            record = json.loads(saved[-1].read_text())
            assert record['append_membership_features'] == append
            assert record['mixer_condition_dim'] == (196 if append else 98)
            assert record['score_important_membership'][:2] == [True, True]
            assert logger.wandb_current_data['mixer_input/grouping_membership']['nrows'] == 98
            assert isinstance(logger.wandb_current_data['mixer_input/grouping_heatmap'], wandb.Image)
            if variant == 'group_variance':
                assert record['variance_probe_count'] >= 2
                assert th.allclose(mixer.score_ema, mixer.variance_difference_scores)
                assert th.allclose(mixer.variance_difference_scores,
                                   mixer.original_qtot_variance - mixer.masked_qtot_variance)
                zeros = th.zeros(98)
                mixer.important_slots.zero_()
                mixer.score_initialized.zero_()
                migration = mixer.apply_scores(zeros, zeros, variance_difference=-th.ones(98))
                assert migration['promoted'] is None and not mixer.important_slots.any()
            checkpoint = Path(temp) / tag
            checkpoint.mkdir()
            learner.save_models(checkpoint)
            expected = mixer.important_slots.clone()
            mixer.important_slots.logical_not_()
            learner.load_models(checkpoint)
            assert th.equal(expected, mixer.important_slots)
            mixer.append_mask = not append
            try:
                learner.load_models(checkpoint)
                raise AssertionError('Cannot resume with a different input encoding')
            except ValueError:
                pass
            mixer.append_mask = append
            mixer.variant = 'group_tderror'
            try:
                learner.load_models(checkpoint)
                raise AssertionError('Cannot resume with a different importance criterion')
            except ValueError:
                pass
            print('PASS: {} — real 5m6m shapes, gradient routes, state-only masking, signed metric, native heatmaps, checkpoints'.format(tag))
    print('PASS: exact population variance difference, fixed Q, group-only probes, padding exclusion, negative scores')


if __name__ == '__main__':
    main()
