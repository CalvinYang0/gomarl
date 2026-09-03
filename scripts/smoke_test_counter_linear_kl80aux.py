#!/usr/bin/env python3
"""Verify a real linear-only path and active-branch loss/mask diagnostics."""
from pathlib import Path
from unittest.mock import patch
import torch as th
from smoke_test_counter_transformer_nine import make_case, check
from modules.agents.counter_transformer_suite import experiment_overrides
from ozstar_submit_counter_transformer_nine import build_plans


def check_linear():
    original = experiment_overrides("relation_kl80aux")
    variant = experiment_overrides("linear_relation_kl80aux")
    assert {k for k in original if original[k] != variant[k]} == {"clean_model_type"}
    th.manual_seed(7)
    mac, learner, batch, logger = make_case("linear_relation_kl80aux")
    capturer = mac.agent.rpg_relation_capturer
    assert isinstance(capturer.dual_linear_encoder, th.nn.Linear)
    assert len(capturer.transformer_layers) == 0
    assert capturer.dual_condition_fuser is None
    assert learner.counter_branch_index == 0
    assert capturer.kl_auxiliary_prior == .8
    observed = []
    hook = capturer.dual_linear_encoder.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone()))
    mac.set_dynamic_branch_gate_t_env(300000)

    def forward(auxiliary, test=False):
        capturer.kl80_auxiliary_enabled = auxiliary
        mac.init_hidden(batch.batch_size)
        th.manual_seed(123)
        return mac.forward(batch, t=0, test_mode=test).detach().clone()

    with patch.object(capturer, "_forward_full_obs_attention_branch", side_effect=AssertionError("Unused Transformer executed")):
        plain = forward(False)
        perturbed = forward(True)
        assert not th.equal(plain, perturbed)
        expected = batch["obs"][:, 0] * capturer.latest_dynamic_branch_gates_graph[0].detach() * capturer.latest_kl80_auxiliary_mask[0]
        assert th.equal(observed[-1], expected)
        assert th.equal(forward(False, True), forward(True, True))
        capturer.kl80_auxiliary_enabled = False
        learner.train(batch, t_env=300000, episode_num=1)
    hook.remove()
    assert capturer.dual_linear_encoder.weight.grad.abs().sum() > 0
    for gate in (capturer.dynamic_branch_gate, capturer.kl80_auxiliary_gate):
        grad = gate.gate_network[-1].bias.grad.reshape(2, -1)
        assert grad[0].abs().sum() > 0
        assert grad[1].abs().sum() == 0  # No loss incorrectly supervises attention.
    assert logger.stats["loss_mask_parameter_relation"][-1][1] > 0
    assert logger.stats["loss_kl80_random_auxiliary"][-1][1] > 0
    assert logger.stats["train_gate/main_linear_mask/valid_slot_count"][-1][1] == 840
    assert "train_gate/main_attention_mask/mean" not in logger.stats
    print("Linear-only execution, actual mask product, test isolation and active-branch gradients OK")


if __name__ == "__main__":
    th.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    plan, = build_plans(repo, ["linear_relation_kl80aux"])
    assert plan["exports"]["TEST_INTERVAL"] == "10000"
    assert len(build_plans(repo)) == 9
    check_linear()
    check("linear_relation_kl80aux")
    print("Single linear + relation + KL80 auxiliary passed (synthetic episodes)")
