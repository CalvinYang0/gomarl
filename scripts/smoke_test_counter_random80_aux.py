#!/usr/bin/env python3
"""Fixed Concrete control: matched main network, sampler, isolation, real update."""
import math
import torch as th

from smoke_test_counter_transformer_nine import check, make_case
from modules.agents.counter_transformer_suite import experiment_overrides


def check_fixed_concrete():
    original = experiment_overrides("relation_kl80aux")
    fixed = experiment_overrides("relation_random80")
    assert {key for key in original if original[key] != fixed[key]} == {
        "clean_model_type", "clean_random_drop_auxiliary_keep_probability",
    }
    th.manual_seed(7)
    learned_mac, _, batch, _ = make_case("relation_kl80aux")
    th.manual_seed(7)
    mac, learner, _, logger = make_case("relation_random80")
    a, b = learned_mac.agent.state_dict(), mac.agent.state_dict()
    assert a.keys() == b.keys()
    assert all(th.equal(a[k], b[k]) for k in a if "kl80_auxiliary_gate" not in k)
    capturer = mac.agent.rpg_relation_capturer
    gate = capturer.kl80_auxiliary_gate
    learned_gate = learned_mac.agent.rpg_relation_capturer.kl80_auxiliary_gate
    assert not any(p.requires_grad for p in gate.parameters())
    assert learner.mask_parameter_relation_active and learner.mask_parameter_relation_coef == 1.0
    assert learner.concrete_random_drop_auxiliary and not learner.kl80_random_drop_auxiliary
    assert not learner.gate_regularization_active
    assert learner.random_drop_auxiliary_coef == 1.0
    assert learner.random_drop_auxiliary_keep_probability == 0.8
    assert gate.binary_concrete_temperature == learned_gate.binary_concrete_temperature == 0.5
    with th.no_grad():
        learned_gate.gate_network[-1].bias.fill_(math.log(0.8 / 0.2))
    obs = batch["obs"][:, 0]
    th.manual_seed(12)
    mask, probability = gate(obs, sample=True)
    th.manual_seed(12)
    reference, _ = learned_gate(obs, sample=True)
    assert th.equal(mask, reference)  # Exactly the same sampler at p=.8.
    assert th.allclose(probability, th.full_like(probability, 0.8))
    assert th.allclose(gate(obs * 100, sample=False)[1], probability)
    assert ((mask > 0) & (mask < 1)).any()  # Not hard Bernoulli.
    assert not th.equal(mask, gate(obs, sample=True)[0])

    def forward(auxiliary, test):
        capturer.kl80_auxiliary_enabled = auxiliary
        mac.set_dynamic_branch_gate_t_env(300000)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(123)
        return mac.forward(batch, t=0, test_mode=test).detach()
    assert not th.allclose(forward(False, False), forward(True, False))
    assert th.equal(forward(False, True), forward(True, True))
    capturer.kl80_auxiliary_enabled = False
    frozen = {k: v.clone() for k, v in gate.state_dict().items()}
    learner.train(batch, t_env=300000, episode_num=1)
    assert all(th.equal(v, gate.state_dict()[k]) for k, v in frozen.items())
    assert not logger.stats.get("loss_kl80_random_auxiliary")
    assert capturer.latest_kl80_auxiliary_loss.item() == 0.0
    assert not capturer.kl80_auxiliary_enabled
    assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0
    print("Fixed p=.8, identical Concrete sampler, no auxiliary learning/KL, training-only injection: OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    check_fixed_concrete()
    check("relation_random80")
    print("Fixed random80 auxiliary passed (synthetic episodes; simulator not exercised)")
