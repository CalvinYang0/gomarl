#!/usr/bin/env python3
"""Prove isolated branch cannot update actor, including clipping/optimizer."""
import copy
from pathlib import Path
import tempfile
import torch as th

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner


def zero(grads):
    return all(g is None or th.count_nonzero(g) == 0 for g in grads)


def main():
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        for optimizer in ('adam', 'rmsprop'):
            th.manual_seed(37)
            mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control='group_calibration',
                                      mixer_input_detach_unimportant=True, optimizer=optimizer,
                                      slot_group_interval=1000, slot_importance_samples=8,
                                      grad_norm_clip=.01))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_ISOLATION_' + optimizer
            args.seed, args.wandb_run_name = 1, 'SYNTHETIC_ISOLATION'
            a = MixerInputLearner(mac, batch.scheme, logger, args)
            a.last_image_t = 10  # No grouping/media during the update comparison.
            a.mixer.important_slots[0] = True
            a.target_mixer.sync_partition(a.mixer)
            b = copy.deepcopy(a)
            online = a._rollout(a.mac, batch)
            chosen = online[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            values = a.mixer.values(chosen, batch['state'][:, :-1], 'train', detach_unimportant=True)
            actor = list(a.mac.parameters())
            first, second = list(a.mixer.mixers[0].parameters()), list(a.mixer.mixers[1].parameters())
            loss0, loss1 = values[..., 0].square().mean(), values[..., 1].square().mean()
            grads = th.autograd.grad(loss1, actor + first + second, allow_unused=True, retain_graph=True)
            assert zero(grads[:len(actor) + len(first)])
            assert not zero(grads[len(actor) + len(first):])
            grads = th.autograd.grad(loss0, actor + first + second, allow_unused=True)
            assert not zero(grads[:len(actor)])
            assert not zero(grads[len(actor):len(actor) + len(first)])
            assert zero(grads[len(actor) + len(first):])
            assert not (set(map(id, a.main_params)) & set(map(id, a.other_params)))
            # Inflate ONLY the other branch's output/loss. Its large gradient
            # must not alter actor/main updates via combined clipping/state.
            original = b.mixer.conditioned_value
            b.mixer.conditioned_value = lambda q, s, group, mask: original(q, s, group, mask) * (10000 if group == 1 else 1)
            old_other = [p.detach().clone() for p in a.other_params]
            a.train(batch, 10, 1)
            b.train(batch, 10, 1)
            assert all(th.equal(x, y) for x, y in zip(a.main_params, b.main_params))
            assert any(not th.equal(x, y) for x, y in zip(old_other, a.other_params))
            assert th.equal(a.last_double_actions, b.last_double_actions)
            path = Path(temp) / optimizer
            path.mkdir()
            a.save_models(path)
            assert (path / 'opt_unimportant.th').is_file()
            a.load_models(path)
            assert a.other_optimiser.state_dict()['state']
            b.detach_unimportant = False
            saved = [p.detach().clone() for p in b.params]
            try:
                b.load_models(path)
            except ValueError as error:
                assert 'routing differs' in str(error)
            else:
                raise AssertionError('Cannot resume across gradient-routing modes')
            assert all(th.equal(x, y) for x, y in zip(saved, b.params))
            print('PASS: {} detached actor; both mixers train; separate clipping; identical actor update under huge auxiliary loss; checkpoint'.format(optimizer))


if __name__ == '__main__':
    main()
