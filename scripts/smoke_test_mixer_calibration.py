#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import torch as th

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from utils.mixer_state_importance import (score_state_slot_calibration, state_slot_names,
                                          complete_episode_returns)


def formulas():
    class Mixer:
        state_dim = 2
        def masks(self, s, mode):
            return s.new_ones(1, 2)
        def conditioned_value(self, q, s, group, mask):
            return ((s * mask)[..., :1] + 2 * (s * mask)[..., 1:2]) * q[..., :1]
    q = th.ones(1, 3, 1)
    s = th.tensor([[[1., 1.], [3., 1.], [999., 999.]]])
    valid = th.tensor([[[1.], [1.], [0.]]])
    g = th.tensor([[[1.], [3.], [999.]]])
    score, count, before, after = score_state_slot_calibration(Mixer(), q, s, g, valid)
    assert count == 2 and th.allclose(score, th.tensor([3 / 5, 4 / 5]))
    assert th.equal(before, th.tensor([4., 4.])) and th.equal(after, th.tensor([1., 0.]))
    scaled, *_ = score_state_slot_calibration(Mixer(), q * 100, s, g * 100, valid)
    assert th.allclose(score, scaled)
    assert score_state_slot_calibration(Mixer(), q, s, g, valid * 0) is None
    gain, *_ = score_state_slot_calibration(Mixer(), q, s,
                                           th.tensor([[[3.], [5.], [0.]]]), valid)
    assert th.allclose(gain, th.tensor([-5 / 17, -4 / 17]))  # Keep negative gains.
    rewards = th.tensor([[[1.], [2.], [3.]], [[1.], [2.], [3.]]])
    term = th.tensor([[[0.], [0.], [1.]], [[0.], [0.], [0.]]])
    returns, mask = complete_episode_returns(rewards, term, th.ones_like(term), .5)
    assert th.equal(returns[0, :, 0], th.tensor([2.75, 3.5, 3.]))
    assert mask[0].all() and not mask[1].any()


def main():
    th.set_num_threads(1)
    formulas()
    with tempfile.TemporaryDirectory() as temp:
        for map_name, variant in (('5m_vs_6m', 'group_calibration'), ('8m_vs_9m', 'static_gate')):
            mac, _, batch, logger = make_case('baseline', map_name, production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=variant,
                                      slot_group_interval=1, slot_importance_samples=8))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_' + variant
            args.seed, args.wandb_run_name = 1, 'SYNTHETIC_CALIBRATION_QA'
            batch['filled'][1, 3] = 1  # Include the natural terminal transition.
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            assert mac.agent.hyper_out_w is None
            assert (args.n_agents, args.n_actions, args.state_shape) == (
                (5, 12, 98) if map_name == '5m_vs_6m' else (8, 15, 179))
            assert len(state_slot_names(args, learner.mixer.state_dim)) == args.state_shape
            assert logger._wandb_metric_allowed('mixer_input/important_count')
            learner.train(batch, 10, 1)
            assert th.isfinite(learner.last_losses).all()
            assert len(learner.last_losses) == (2 if variant == 'group_calibration' else 1)
            if variant == 'group_calibration':
                mixer = learner.mixer
                before_partition = mixer.important_slots.clone()
                before_updates = mixer.grouping_updates.clone()
                batch['terminated'].zero_()
                learner.train(batch, 20, 2)
                assert th.equal(before_partition, mixer.important_slots)
                assert th.equal(before_updates, mixer.grouping_updates)
                assert mixer.calibration_probe_count.item() == 0
                mixer.important_slots.zero_()
                mixer.score_initialized.zero_()
                bad = th.arange(mixer.state_dim).float() + 1
                bad[3] = -1
                change = bad.new_zeros(bad.shape)
                result = mixer.apply_scores(change, change, bad)
                assert result['promoted'] == 3  # Lowest badness, not highest.
                learner.target_mixer.sync_partition(mixer)
                assert th.equal(mixer.removal_gain_scores, learner.target_mixer.removal_gain_scores)
                records = list((Path(temp) / 'mixer_input_control' / args.unique_token).rglob('*.json'))
                assert 'normalized_removal_gain' in json.loads(records[0].read_text())
                path = Path(temp) / 'checkpoint'
                path.mkdir()
                learner.save_models(path)
                learner.load_models(path)
                assert th.equal(mixer.removal_gain_scores, bad)
            print('PASS: {} {}; training/heatmaps/checkpoint'.format(map_name, variant))
    print('PASS: MSE difference; padding; signed gains; common scale; MC terminal mask; low-removal-gain ranking')


if __name__ == '__main__':
    main()
