#!/usr/bin/env python3
"""Production 5m6m shapes and ordinary-agent updates; no SC2 launch."""
import copy
import json
from pathlib import Path
import tempfile

import torch as th

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from utils.mixer_state_importance import score_state_slots, state_slot_names
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics


def check_formulas():
    class Mixer:
        state_dim = 2
        mixers = [None]
        def masks(self, states, mode):
            return states.new_ones(1, 2)
        def conditioned_value(self, q, s, group, mask):
            return (2 * s[..., :1] + s[..., 1:2]) + q[..., :1] * 0
    states = th.tensor([[[0., 0.], [2., 0.], [1000., 1000.]]])
    q = th.zeros(1, 3, 1)
    valid = th.tensor([[[1.], [1.], [0.]]])
    y = th.tensor([[[0.], [4.], [0.]]])
    qchange, tderror, count = score_state_slots(Mixer(), q, states, y, valid, 256)
    assert count == 2 and th.equal(qchange, th.tensor([2., 0.]))
    assert th.equal(tderror, th.tensor([4., 0.]))
    # If replacement improves fitting, positive mean TD increase is zero.
    _, tderror, _ = score_state_slots(Mixer(), q, states, th.zeros_like(y), valid, 256)
    assert th.equal(tderror, th.zeros(2))
    assert score_state_slots(Mixer(), q, states, y, th.zeros_like(valid), 256) is None


def main():
    th.set_num_threads(1)
    check_formulas()
    first_group_parameters = None
    with tempfile.TemporaryDirectory(prefix="gomarl-mixer-input-smoke-") as temp:
        for variant in ("static_gate", "group_qchange", "group_tderror"):
            th.manual_seed(17)
            mac, _, batch, logger = make_case("baseline", "5m_vs_6m", production_shapes=True,
                config_overrides=dict(clean_model_type="qmix_minimal", mixer_input_control=variant,
                                      slot_group_interval=1, slot_importance_samples=8,
                                      slot_heatmap_interval=100000))
            args = mac.args
            args.local_results_path = temp
            args.unique_token = "SYNTHETIC_" + variant
            args.wandb_run_name = "SYNTHETIC_QA_NOT_EXPERIMENT_" + variant
            args.seed = 1
            # Production run has state_shape but not fixture-only obs_shape.
            del args.obs_shape
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            if learner.mixer.grouped:
                parameters = {k: v.clone() for k, v in learner.mixer.state_dict().items()}
                if first_group_parameters is None:
                    first_group_parameters = parameters
                else:
                    assert parameters.keys() == first_group_parameters.keys()
                    assert all(th.equal(v, first_group_parameters[k]) for k, v in parameters.items())
            assert args.n_agents == 5 and args.n_actions == 12 and args.state_shape == 98
            assert mac.agent.hyper_out_w is None and mac.agent.fixed_head is not None
            assert not any("hyper_out" in n for n, _ in mac.agent.named_parameters())
            assert len(state_slot_names(args, 98)) == 98
            assert not learner.mixer.important_slots.any()
            before = [p.detach().clone() for p in mac.agent.fixed_head.parameters()]
            learner.train(batch, 10, 1)
            assert any(not th.equal(a, b) for a, b in zip(before, mac.agent.fixed_head.parameters()))
            assert all(th.isfinite(p).all() for p in learner.params)
            assert len(learner.last_losses) == (1 if variant == "static_gate" else 2)
            assert abs(logger.stats["loss_td"][-1][1] - learner.last_losses.mean().item()) < 1e-5
            if learner.mixer.grouped:
                assert th.equal(learner.mixer.important_slots, learner.target_mixer.important_slots)
                assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
                           for m in learner.mixer.mixers)
                qchange = th.zeros(98)
                tderror = th.zeros(98)
                qchange[0], qchange[1] = 10, 1
                tderror[0], tderror[1] = 1, 10
                learner.mixer.important_slots.zero_()
                learner.mixer.score_initialized.zero_()
                migration = learner.mixer.apply_scores(qchange, tderror)
                assert migration["promoted"] == (0 if variant == "group_qchange" else 1)
                assert int(learner.mixer.important_slots.sum()) == 1
            else:
                assert learner.mixer.gate_logits.grad is not None
                assert learner.mixer.gate_logits.grad.abs().sum() > 0
                p = learner.mixer.gate_logits.sigmoid()
                assert th.equal(learner.mixer.masks(batch["state"], "eval")[0],
                                (p > .1).to(p).expand_as(batch["state"]))
                assert th.equal(learner.mixer.masks(batch["state"], "target")[0], p.expand_as(batch["state"]))
            # A deleted state field cannot leak via W1/W2/b1/V. Changing state
            # or membership does not change ordinary decentralized utilities.
            mac.init_hidden(batch.batch_size)
            ordinary = mac.forward(batch, 0, test_mode=True).detach().clone()
            state = batch["state"][:, :1].clone()
            q = th.randn(batch.batch_size, 1, 5, requires_grad=True)
            if learner.mixer.grouped:
                learner.mixer.important_slots.zero_()
                learner.mixer.important_slots[0] = True
            else:
                with th.no_grad():
                    learner.mixer.gate_logits.fill_(-10)
                    learner.mixer.gate_logits[0] = 10
            base = learner.mixer(q, state)
            altered = state.clone()
            altered[..., 1:] += 100
            assert th.equal(base, learner.mixer(q, altered))
            sensitivity = th.autograd.grad(base.sum(), q)[0]
            assert (sensitivity >= 0).all()
            copied = copy.deepcopy(batch)
            copied["state"][:] += 100
            mac.init_hidden(batch.batch_size)
            assert th.equal(ordinary, mac.forward(copied, 0, test_mode=True))
            # Both branch values have independent calibration diagnostics.
            batch["filled"][1, 3] = 1
            summary = ValueDiagnosticSummary()
            collect_value_diagnostics(mac, learner.mixer, batch, args.gamma, summary)
            assert summary.moments["q_tot"][0] and summary.moments["q_mc_bias"][0]
            assert summary.moments["mixer_w1"][0] and summary.moments["mixer_w_final"][0]
            if learner.mixer.grouped:
                assert summary.moments["q_tot_unimportant"][0]
                assert summary.moments["q_mc_bias_unimportant"][0]
                assert summary.moments["mixer_w1_unimportant"][0]
            checkpoint = Path(temp) / variant
            checkpoint.mkdir()
            learner.save_models(checkpoint)
            saved_mask = learner.mixer.important_slots.clone()
            learner.mixer.important_slots.logical_not_()
            learner.load_models(checkpoint)
            assert th.equal(saved_mask, learner.mixer.important_slots)
            assert th.equal(saved_mask, learner.target_mixer.important_slots)
            records = list((Path(temp) / "mixer_input_control" / args.unique_token).rglob("*.json"))
            assert records and all("qchange" in json.loads(p.read_text()) for p in records)
            assert list((Path(temp) / "mixer_input_control" / args.unique_token).rglob("*.png"))
            print("PASS: {} real 5m6m dimensions; fixed actor; TD/gradient paths; masks; ranking; diagnostics; checkpoint; heatmap".format(variant))
    print("PASS: exact importance formulas; masked padding ignored; no valid probes handled")


if __name__ == "__main__":
    main()
