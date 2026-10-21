#!/usr/bin/env python3
"""Synthetic QA only: deterministic sigmoid interpolation, exact grads and TD."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from smoke_test_onefill_gate import case
from smoke_test_gate_score_supervision import equal_state
from modules.mixers.onefill_gate import OneFillGateMixer
from modules.mixers.soft_onefill_gate import SoftOneFillGateMixer
from learners import REGISTRY
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys, guard_and_route
from ozstar_submit_5m6m_onefill_gate_15jobs import build_plans as legacy_plans
from ozstar_submit_5m6m_soft_onefill_3seeds import build_plans


def small(layers=2):
    return SimpleNamespace(state_shape=4, n_agents=2, mixing_embed_dim=8,
        hypernet_layers=layers, hypernet_embed=8, mixer_input_initial_probability=.1,
        mixer_input_gate_temperature=.5, mixer_onefill_residual=False, mixer_onefill_residual_scale=.1)


def algebra():
    for layers in (1, 2):
        args = small(layers)
        th.manual_seed(157)
        mixer = SoftOneFillGateMixer(args).double()
        th.manual_seed(157)
        legacy = OneFillGateMixer(args).double()
        assert equal_state(mixer.state_dict(), legacy.state_dict()), "No initialization or weight layout change"
        q = th.randn(2, 3, 2, dtype=th.double, requires_grad=True)
        s = th.randn(2, 3, 4, dtype=th.double, requires_grad=True)
        values = mixer.gate_values(s)
        th.testing.assert_close(values, th.full_like(s, .1), rtol=1e-6, atol=1e-8)
        assert th.equal(values[0, 0], values[1, 2]), "Static coordinate gates, not state-conditioned MLP"
        assert th.equal(mixer.controlled_input(s, values), values * s + (1-values))
        assert th.equal(mixer.controlled_input(s, th.zeros_like(s)), th.ones_like(s))
        assert th.equal(mixer.controlled_input(s, th.ones_like(s)), s)
        derivative = th.autograd.grad(mixer.controlled_input(s, values).sum(), s, retain_graph=True)[0]
        assert th.equal(derivative, values)
        logit_gradient = th.autograd.grad(mixer.controlled_input(s, values).sum(), mixer.gate_logits)[0]
        expected = ((s-1) * values * (1-values)).sum((0, 1))
        th.testing.assert_close(logit_gradient, expected)
        rng = th.get_rng_state().clone()
        with patch("torch.rand_like", side_effect=AssertionError("No random hard mask allowed")):
            results = [mixer(q, s, mode=mode) for mode in ("train", "target", "eval", "train")]
        assert all(th.equal(results[0], result) for result in results)
        assert th.equal(rng, th.get_rng_state())
        assert th.equal(results[0], legacy(q, s, mode="eval"))
        assert (th.autograd.grad(results[0].sum(), q, retain_graph=True)[0] >= 0).all()
        # Check exact gate-to-Qtot derivative numerically, not an ST surrogate.
        from torch.func import functional_call
        logits = mixer.gate_logits.detach().clone().requires_grad_()
        assert th.autograd.gradcheck(lambda a: functional_call(mixer, {"gate_logits": a},
            (q.detach(), s.detach()), {"mode": "train"}), (logits,), eps=1e-6, atol=1e-5)
        results[0].square().mean().backward()
        assert mixer.gate_logits.grad.abs().sum() > 0
        for parameter in mixer.base.parameters():
            assert parameter.grad is not None and th.isfinite(parameter.grad).all()
        for branch in (mixer.base.hyper_w1, mixer.base.hyper_w_final, mixer.base.hyper_b1, mixer.base.value):
            assert any(p.grad.abs().sum() > 0 for p in branch.parameters())
        # Full close recovers all-ones; p=.1 does NOT mean full close.
        closed = mixer.conditioned_value(q, s, th.zeros_like(s))
        assert th.equal(closed, mixer.base(q, th.ones_like(s)))
        assert not th.allclose(closed, results[0])
        assert not hasattr(mixer, "threshold") and not hasattr(mixer, "sample_mask")
        try:
            mixer(q, s, mode="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown forward mode accepted")
    for initial in (0, 1, float("nan"), -1):
        args = small()
        args.mixer_input_initial_probability = initial
        try:
            SoftOneFillGateMixer(args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid initial value accepted")
    print("PASS: exact soft one-fill algebra, all-mode determinism, finite-difference gradients and monotonicity")


def training(root):
    for scene in ("5m_vs_6m", "8m_vs_9m"):
        for optimizer in ("adam", "rmsprop"):
            old, batch, logger = case(scene, "td", False, optimizer, root)
            args = old.args
            args.learner = "soft_onefill_gate_learner"
            args.unique_token = "SYNTHETIC_SOFT_" + scene + optimizer
            # Temperature and threshold are irrelevant, even if nonsensical.
            args.mixer_input_gate_temperature = args.mixer_input_gate_threshold = float("nan")
            actor = copy.deepcopy(old.mac.agent.state_dict())
            learner = REGISTRY[args.learner](old.mac, batch.scheme, logger, args)
            assert equal_state(actor, learner.mac.agent.state_dict())
            assert isinstance(learner.mac.agent.fixed_head[1], th.nn.ELU)
            assert len(learner.mac.agent.fixed_head) == 3
            assert learner.contract["sampling"] == "none_deterministic_sigmoid"
            assert "temperature" not in learner.contract
            assert learner.gate_optimiser is None
            assert any(p is learner.mixer.gate_logits for p in learner.main_params)
            initial_gate = learner.mixer.gate_logits.detach().clone()
            initial_mixer = copy.deepcopy(learner.mixer.base.state_dict())
            initial_target = copy.deepcopy(learner.target_mixer.state_dict())
            with patch("torch.rand_like", side_effect=AssertionError("Training sampled a hard gate")):
                learner.train(batch, 10, 1)
            assert learner.last_gate_aux_loss is None and th.isfinite(learner.last_losses).all()
            assert not th.equal(initial_gate, learner.mixer.gate_logits)
            assert not equal_state(initial_mixer, learner.mixer.base.state_dict())
            assert not equal_state(actor, learner.mac.agent.state_dict())
            assert equal_state(initial_target, learner.target_mixer.state_dict())
            learner.train(batch, 11, 200)
            assert equal_state(learner.mixer.state_dict(), learner.target_mixer.state_dict())
            assert all(p.grad is None for p in learner.target_mixer.parameters())
            assert logger.stats["mixer_input/soft_gate_value_mean"]
            assert "mixer_input/expected_open_slots" not in logger.stats
            rng = th.get_rng_state().clone()
            learner._snapshot(12)
            record = next((root / "mixer_soft_onefill_gate" / args.unique_token).glob("*.json"))
            assert json.loads(record.read_text())["gate_value"] == learner.mixer.gate_logits.sigmoid().detach().tolist()
            summary = ValueDiagnosticSummary()
            collect_value_diagnostics(learner.mac, learner.mixer, batch, args.gamma, summary)
            assert th.equal(rng, th.get_rng_state())
            for metric in ("q_tot", "mixer_w1", "mixer_w_final", "mixer_dqtot_dqi"):
                assert summary.moments[metric][0] > 0
            destination = root / ("checkpoint_" + scene + optimizer)
            destination.mkdir()
            learner.save_models(destination)
            learner.mac.init_hidden(batch.batch_size)
            resumed = REGISTRY[args.learner](copy.deepcopy(learner.mac), batch.scheme, logger, args)
            resumed.load_models(destination)
            assert equal_state(learner.mixer.state_dict(), resumed.mixer.state_dict())
            assert equal_state(learner.target_mixer.state_dict(), resumed.target_mixer.state_dict())
            learner.train(batch, 13, 201)
            resumed.train(batch, 13, 201)
            assert equal_state(learner.mixer.state_dict(), resumed.mixer.state_dict())
            assert equal_state(learner.optimiser.state_dict(), resumed.optimiser.state_dict())
            assert equal_state(learner.mac.agent.state_dict(), resumed.mac.agent.state_dict())
            historical = root / ("legacy_" + scene + optimizer)
            historical.mkdir()
            old.save_models(historical)
            try:
                resumed.load_models(historical)
            except ValueError as exc:
                assert "contract" in str(exc)
            else:
                raise AssertionError("Hard-ST checkpoint accepted as a soft experiment")
            print("PASS:", scene, optimizer, "unchanged head, TD gate update, target, diagnostics and resume")
    old, batch, logger = case("5m_vs_6m", "td", False, "adam", root)
    for objective in ("kl", "mc"):
        args = copy.deepcopy(old.args)
        args.mixer_onefill_objective = objective
        try:
            REGISTRY["soft_onefill_gate_learner"](old.mac, batch.scheme, logger, args)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected auxiliary loss accepted")


def plans():
    repo = Path(__file__).resolve().parents[1]
    planned = build_plans(repo)
    assert len(planned) == 3 and {p["seed"] for p in planned} == {1, 2, 3}
    legacy = {p["seed"]: p for p in legacy_plans(repo) if p["label"] == "qmix_onefill_p10"}
    for p in planned:
        assert p["scene"] == "smac_5m6m" and p["map_name"] == "5m_vs_6m"
        assert p["exports"]["CONFIG"] == "clean_hyper" and p["exports"]["MODEL_TYPE"] == "qmix_minimal"
        assert "--time=1-00:00:00" in p["sbatch_args"]
        assert p["job_name"] != legacy[p["seed"]]["job_name"]
        extra = dict(x.split("=", 1) for x in shlex.split(p["exports"]["EXTRA_ARGS"]))
        previous = dict(x.split("=", 1) for x in shlex.split(legacy[p["seed"]]["exports"]["EXTRA_ARGS"]))
        assert extra.pop("learner") == "soft_onefill_gate_learner"
        previous.pop("learner")
        previous.pop("mixer_input_gate_temperature")
        assert extra == previous, "No changes to loss, actor or training recipe"
    routed = copy.deepcopy(planned)
    guard_and_route(routed, Path("/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    validate_config_keys(repo, routed)
    print("PASS: three isolated 5m6m p10 plans, same actor/TD settings, 24h; no jobs submitted")


def main():
    th.set_num_threads(1)
    algebra()
    with tempfile.TemporaryDirectory(prefix="soft-onefill-qa-") as scratch:
        training(Path(scratch))
    plans()


if __name__ == "__main__":
    main()
