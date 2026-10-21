#!/usr/bin/env python3
"""Synthetic-only Q_tot-level algebra, actor identity and actual TD updates."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch as th

from smoke_test_onefill_gate import case
from smoke_test_gate_score_supervision import equal_state
from modules.mixers.qtot_composition import QTotCompositionMixer
from modules.mixers.onefill_gate import OneFillGateMixer
from learners import REGISTRY as learners
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics


def small(combination="sum", both=False):
    return QTotCompositionMixer(SimpleNamespace(
        state_shape=4, n_agents=2, mixing_embed_dim=8, hypernet_layers=2, hypernet_embed=8,
        mixer_input_initial_probability=.1, mixer_input_gate_temperature=.5,
        mixer_onefill_residual=False, mixer_onefill_residual_scale=.1,
        mixer_qtot_combination=combination, mixer_qtot_secondary_ones=both,
        mixer_qtot_alpha_initial=.1))


def algebra():
    th.manual_seed(73)
    fresh = small()
    th.manual_seed(73)
    legacy = OneFillGateMixer(fresh.base.args)
    assert equal_state(fresh.base.state_dict(), legacy.base.state_dict())
    assert th.equal(fresh.gate_logits, legacy.gate_logits)
    th.manual_seed(74)
    q, s = th.randn(2, 5, 2, requires_grad=True), th.randn(2, 5, 4)
    mask = th.tensor([1., 0., 1., 0.])
    for combination, both in (("sum", True), ("sum", False), ("alpha_sum", False)):
        mixer = small(combination, both)
        x = th.ones_like(s) if both else mask * s + (1 - mask)
        main, sub = mixer.base(q, th.ones_like(s)), mixer.secondary(q, x)
        alpha = mixer.alpha()
        expected = main + sub if alpha is None else alpha * main + (1-alpha) * sub
        actual = mixer.conditioned_value(q, s, mask)
        th.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert (th.autograd.grad(actual.sum(), q, retain_graph=True)[0] >= 0).all()
        assert not any(a is b for a in mixer.base.parameters() for b in mixer.secondary.parameters())
        assert not equal_state(mixer.base.state_dict(), mixer.secondary.state_dict())
        actual.square().mean().backward()
        for branch in (mixer.base, mixer.secondary):
            for name, parameter in branch.named_parameters():
                assert parameter.grad is not None and th.isfinite(parameter.grad).all(), name
            assert branch.hyper_b1.weight.grad.abs().sum() > 0
            assert branch.value[-1].bias.grad.abs().sum() > 0
        mixer(q, s, mode="train").square().mean().backward()
        if both:
            assert not mixer.gate_logits.requires_grad and mixer.gate_logits.grad is None
            rng = th.get_rng_state().clone()
            assert th.equal(mixer(q, s, mode="train"), mixer(q, s + 100, mode="train"))
            assert th.equal(rng, th.get_rng_state())
        else:
            assert mixer.gate_logits.grad.abs().sum() > 0
        if alpha is not None:
            assert abs(alpha.item() - .1) < 1e-7 and mixer.alpha_logit.grad.abs() > 0
        assert not hasattr(mixer, "residual_w1") and not hasattr(mixer, "diagnostic_parameters")
    # Q_tot can be negative; no final abs, 0.1 scaling, centering or weight sum.
    mixer = small()
    with th.no_grad():
        for branch, bias in ((mixer.base, -2), (mixer.secondary, -3)):
            for parameter in branch.parameters():
                parameter.zero_()
            branch.value[-1].bias.fill_(bias)
    assert th.equal(mixer.conditioned_value(q, s, mask), th.full((2, 5, 1), -5.))
    print("PASS: complete Q_tot algebra, W1/W2/b1/b2 gradients, monotonicity and signed outputs")


def production(root):
    actor = None
    for combination, both in (("sum", True), ("sum", False), ("alpha_sum", False)):
        th.manual_seed(75)
        original, batch, logger = case("5m_vs_6m", "td", False, "adam", root)
        args = original.args
        args.mixer_qtot_combination, args.mixer_qtot_secondary_ones = combination, both
        args.mixer_qtot_alpha_initial = .1
        args.unique_token = "SYNTHETIC_QTOT_" + combination + str(both)
        before_actor = copy.deepcopy(original.mac.agent.state_dict())
        if actor is None:
            actor = before_actor
        assert equal_state(actor, before_actor)
        learner = learners["qtot_composition_learner"](original.mac, batch.scheme, logger, args)
        assert equal_state(before_actor, learner.mac.agent.state_dict())
        assert isinstance(learner.mac.agent.fixed_head, th.nn.Sequential)
        assert isinstance(learner.mac.agent.fixed_head[1], th.nn.ELU)
        assert len(learner.mac.agent.fixed_head) == 3
        assert learner.contract["version"] == 3
        before_base = copy.deepcopy(learner.mixer.base.state_dict())
        before_sub = copy.deepcopy(learner.mixer.secondary.state_dict())
        before_gate = learner.mixer.gate_logits.detach().clone()
        learner.train(batch, 10, 1)
        learner.train(batch, 11, 200)
        assert learner.train_steps == 2 and th.isfinite(learner.last_losses).all()
        assert not equal_state(before_base, learner.mixer.base.state_dict())
        assert not equal_state(before_sub, learner.mixer.secondary.state_dict())
        assert not equal_state(before_actor, learner.mac.agent.state_dict())
        assert th.equal(before_gate, learner.mixer.gate_logits) == both
        assert equal_state(learner.mixer.state_dict(), learner.target_mixer.state_dict())
        assert all(p.grad is None for p in learner.target_mixer.parameters())
        assert logger.stats["mixer_branches/qtot_main_mean"]
        summary = ValueDiagnosticSummary()
        rng, hidden = th.get_rng_state().clone(), learner.mac.hidden_states
        collect_value_diagnostics(learner.mac, learner.mixer, batch, args.gamma, summary)
        assert th.equal(rng, th.get_rng_state()) and learner.mac.hidden_states is hidden
        summary.log(logger, 11)
        assert logger.stats["test_value/qtot_main_mean"] and logger.stats["test_value/qtot_secondary_mean"]
        assert "test_value/mixer_w1_mean" not in logger.stats  # no fictitious combined W
        if combination == "alpha_sum":
            assert logger.stats["mixer_branches/alpha_main"]
        learner._snapshot(12, batch["state"][:, :-1], batch["filled"][:, :-1].float())
        snapshots = list((root / "mixer_onefill_gate" / args.unique_token).glob("*_qtot.json"))
        assert snapshots and json.loads(snapshots[-1].read_text())["branches"]["combined"]["count"] > 0
        destination = root / ("checkpoint_" + combination + str(both))
        destination.mkdir()
        learner.save_models(destination)
        original.mac.init_hidden(batch.batch_size)
        resumed = learners["qtot_composition_learner"](copy.deepcopy(original.mac), batch.scheme, logger, args)
        resumed.load_models(destination)
        assert equal_state(learner.mixer.state_dict(), resumed.mixer.state_dict())
        assert equal_state(learner.target_mixer.state_dict(), resumed.target_mixer.state_dict())
        print("PASS: 5m6m", combination, "both_ones=", both, "unchanged MLP head, joint TD, diagnostics and checkpoint")
    for scene in ("MMM2", "corridor"):
        th.manual_seed(76)
        learner, batch, logger = case(scene, "td", False, "adam", root)
        learner.args.unique_token = "SYNTHETIC_ONEFILL_" + scene
        assert type(learner.mixer).__name__ == "OneFillGateMixer"
        assert isinstance(learner.mac.agent.fixed_head, th.nn.Sequential)
        assert isinstance(learner.mac.agent.fixed_head[1], th.nn.ELU)
        assert learner.mac.agent.fixed_head[-1].out_features == learner.args.n_actions
        before = copy.deepcopy(learner.mixer.state_dict())
        learner.train(batch, 10, 1)
        assert th.isfinite(learner.last_losses).all() and not equal_state(before, learner.mixer.state_dict())
        assert not th.equal(before["gate_logits"], learner.mixer.gate_logits)
        print("PASS:", scene, "ordinary onefill_p10 actual map dimensions and historical-head TD update")


def main():
    th.set_num_threads(1)
    algebra()
    with tempfile.TemporaryDirectory(prefix="qtot-composition-qa-") as scratch:
        production(Path(scratch))


if __name__ == "__main__":
    main()
