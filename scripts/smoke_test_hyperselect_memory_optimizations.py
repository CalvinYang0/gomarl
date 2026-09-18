#!/usr/bin/env python3
"""Regression checks for semantics-preserving HyperSelect memory changes."""
import copy
import logging
from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test_counter_transformer_nine import make_case


LABEL = "relation_advantage_qvalue_augtd_nomasktd"


def rollout(mac, batch, reuse_policy_hidden=False):
    outputs = []
    policy_hidden = []
    mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length):
        override = policy_hidden[t] if reuse_policy_hidden else None
        outputs.append(
            mac.forward(
                batch,
                t=t,
                policy_hidden_override=override,
            )
        )
        if not reuse_policy_hidden:
            policy_hidden.append(
                mac.hidden_states[:, :, : mac.agent.hidden_dim]
            )
    return th.stack(outputs, dim=1), policy_hidden


def two_path_loss(mac, batch, reuse_policy_hidden):
    main, policy_hidden = rollout(mac, batch)
    mac.init_hidden(batch.batch_size)
    mac.set_dynamic_branch_gate_force_open(True)
    full = []
    try:
        for t in range(batch.max_seq_length):
            full.append(
                mac.forward(
                    batch,
                    t=t,
                    policy_hidden_override=(
                        policy_hidden[t] if reuse_policy_hidden else None
                    ),
                )
            )
    finally:
        mac.set_dynamic_branch_gate_force_open(False)
    full = th.stack(full, dim=1)
    return main, full, main.square().mean() + full.square().mean()


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    th.manual_seed(19)
    reference, _, batch, _ = make_case(LABEL)
    optimized = copy.deepcopy(reference)

    th.manual_seed(23)
    reference_main, reference_full, reference_loss = two_path_loss(
        reference, batch, reuse_policy_hidden=False
    )
    reference_loss.backward()

    th.manual_seed(23)
    optimized_main, optimized_full, optimized_loss = two_path_loss(
        optimized, batch, reuse_policy_hidden=True
    )
    optimized_loss.backward()

    assert th.equal(reference_main, optimized_main)
    assert th.equal(reference_full, optimized_full)
    assert th.equal(reference_loss, optimized_loss)
    reference_parameters = dict(reference.agent.named_parameters())
    optimized_parameters = dict(optimized.agent.named_parameters())
    assert reference_parameters.keys() == optimized_parameters.keys()
    for name in reference_parameters:
        left = reference_parameters[name].grad
        right = optimized_parameters[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            assert th.allclose(left, right, rtol=2e-5, atol=2e-6), name

    # Compare the staged multi-path backward with the aggregate-loss backward
    # on the exact same model, replay batch and stochastic masks.
    th.manual_seed(31)
    _, aggregate_learner, train_batch, _ = make_case(LABEL)
    staged_learner = copy.deepcopy(aggregate_learner)
    aggregate_learner.memory_efficient_multi_path = False
    th.manual_seed(37)
    aggregate_learner.train(train_batch, t_env=500000, episode_num=1)
    th.manual_seed(37)
    staged_learner.train(train_batch, t_env=500000, episode_num=1)
    aggregate_parameters = dict(aggregate_learner.mac.agent.named_parameters())
    staged_parameters = dict(staged_learner.mac.agent.named_parameters())
    for name in aggregate_parameters:
        assert th.allclose(
            aggregate_parameters[name],
            staged_parameters[name],
            rtol=3e-5,
            atol=3e-6,
        ), name
    assert len(aggregate_learner.params) == len(staged_learner.params)
    for index, (left, right) in enumerate(
        zip(aggregate_learner.params, staged_learner.params)
    ):
        assert th.allclose(left, right, rtol=3e-5, atol=3e-6), index
    assert all(
        parameter.grad is None or th.isfinite(parameter.grad).all()
        for parameter in staged_learner.params
    )
    print("HyperSelect memory optimizations preserve outputs and gradients")


if __name__ == "__main__":
    main()
