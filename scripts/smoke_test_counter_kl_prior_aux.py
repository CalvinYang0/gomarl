#!/usr/bin/env python3
"""Matched initialization, exact KL prior, gradients, real updates and plots."""
import math
from pathlib import Path
import torch as th
from smoke_test_counter_transformer_nine import make_case, check
from modules.agents.counter_transformer_suite import experiment_overrides
from ozstar_submit_counter_transformer_nine import build_plans


def check_prior(prior):
    label = "relation_kl{}aux".format(round(prior * 100))
    original, variant = experiment_overrides("relation_kl80aux"), experiment_overrides(label)
    assert {key for key in original if original[key] != variant[key]} == {
        "clean_model_type", "clean_kl_auxiliary_prior"}
    th.manual_seed(7)
    reference, _, batch, _ = make_case("relation_kl80aux")
    th.manual_seed(7)
    mac, learner, _, logger = make_case(label)
    assert all(th.equal(value, mac.agent.state_dict()[key]) for key, value in reference.agent.state_dict().items())
    capturer = mac.agent.rpg_relation_capturer
    assert capturer.kl_auxiliary_prior == prior
    assert learner.kl80_random_drop_auxiliary and learner.concrete_random_drop_auxiliary
    assert learner.random_drop_auxiliary_coef == learner.mask_parameter_relation_coef == 1
    assert not learner.gate_regularization_active
    assert capturer.kl80_auxiliary_gate.binary_concrete_temperature == .5
    for model in (reference, mac):
        model.set_dynamic_branch_gate_t_env(300000)
        model.agent.rpg_relation_capturer.kl80_auxiliary_enabled = True
        model.init_hidden(batch.batch_size)
    th.manual_seed(12)
    a = reference.forward(batch, t=0)
    th.manual_seed(12)
    b = mac.forward(batch, t=0)
    assert th.equal(a, b)  # Only the regularizer changes, not initial policy/sampler.
    gate = capturer.kl80_auxiliary_gate
    with th.no_grad():
        gate.gate_network[-1].weight.zero_()
        gate.gate_network[-1].bias.fill_(math.log(.6 / .4))
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0)
    loss = capturer.latest_kl80_auxiliary_loss
    expected = .6 * math.log(.6 / prior) + .4 * math.log(.4 / (1 - prior))
    assert abs(loss.item() - expected) < 1e-6
    grad, = th.autograd.grad(loss, gate.gate_network[-1].bias)
    assert grad.min() >= 0 and grad.max() > 0  # Gradient descent moves .6 toward .5/.3.
    capturer.kl80_auxiliary_enabled = False
    print(label + ": matched initialization/sampler and exact KL gradient OK")
    check(label)


if __name__ == "__main__":
    th.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    assert len(build_plans(repo)) == 9
    plans = build_plans(repo, ["relation_kl50aux", "relation_kl30aux"])
    assert all(plan["exports"]["TEST_INTERVAL"] == "10000" for plan in plans)
    for prior in (.5, .3):
        check_prior(prior)
    print("2/2 KL prior controls passed (synthetic episodes, no Slurm/simulator)")
