#!/usr/bin/env python3
"""Actual learner updates for both missing factorial cells; no simulator."""
import tempfile
import torch as th
from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner


def main():
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        for score, isolated in (('group_qchange', True), ('group_calibration', False)):
            th.manual_seed(47)
            mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=score,
                    mixer_input_detach_unimportant=isolated, slot_group_interval=1, slot_importance_samples=8))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_ABLATION_' + score
            args.wandb_run_name, args.seed = 'SYNTHETIC_ABLATION_NOT_EXPERIMENT', 1
            batch['filled'][1, 3] = 1
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            values = learner._rollout(mac, batch)
            chosen = values[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            joint = learner.mixer.values(chosen, batch['state'][:, :-1], 'train', detach_unimportant=isolated)
            grads = th.autograd.grad(joint[..., 1].square().mean(), list(mac.parameters()), allow_unused=True)
            zero = all(g is None or th.count_nonzero(g) == 0 for g in grads)
            assert zero == isolated, 'Wrong auxiliary agent gradient routing for ' + score
            assert (learner.other_optimiser is not None) == isolated
            learner.train(batch, 10, 1)
            assert learner.last_losses.numel() == 2 and th.isfinite(learner.last_losses).all()
            assert learner.mixer.grouping_updates.item() == 1
            if score == 'group_calibration':
                assert learner.mixer.calibration_probe_count.item() > 0
            assert th.equal(learner.mixer.important_slots, learner.target_mixer.important_slots)
            print('PASS: {} detach={} real 5m6m shapes; auxiliary actor gradient; two TD losses; grouping; target partition'.format(score, isolated))


if __name__ == '__main__':
    main()
