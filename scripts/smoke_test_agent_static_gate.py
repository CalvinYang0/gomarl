#!/usr/bin/env python3
"""Verify independently learned, observation-free masks for every agent."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "agent_static_gate_kl80aux"


def main():
    sacred = yaml.safe_load((ROOT / "src/config/algs/clean_hyper.yaml").read_text())
    overrides = experiment_overrides(LABEL)
    assert not set(overrides) - set(sacred)
    assert overrides["clean_dynamic_branch_gate_static"]
    assert overrides["clean_dynamic_branch_gate_per_agent"]
    assert overrides["clean_mask_parameter_relation_coef"] == 0.0
    assert overrides["clean_random_drop_auxiliary_coef"] == 1.0

    mac, learner, batch, logger = make_case(LABEL)
    capturer = mac.agent.rpg_relation_capturer
    gate = capturer.dynamic_branch_gate
    assert gate.observation_independent
    assert gate.observation_independent_agent_count == mac.args.n_agents
    assert gate.static_logits.shape == (mac.args.n_agents, mac.args.obs_shape)
    assert gate.gate_network is None

    learner.train(batch, t_env=300000, episode_num=1)
    assert not logger.stats.get("loss_mask_parameter_relation")
    assert logger.stats["loss_kl80_random_auxiliary"][-1][1] > 0
    assert gate.static_logits.grad is not None
    assert (gate.static_logits.grad.abs().sum(dim=1) > 0).all()

    # Probabilities may differ by agent, but remain exactly invariant to the
    # observation and timestep for each fixed agent identity.
    mac.set_dynamic_branch_gate_t_env(300000)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0)
    first = mac.latest_dynamic_branch_probabilities_graph.detach().clone()
    mac.forward(batch, t=1)
    second = mac.latest_dynamic_branch_probabilities_graph.detach().clone()
    assert th.equal(first, second)
    assert th.equal(first[:, :1].expand_as(first), first)
    check(LABEL)
    print("Independent per-agent fixed masks + KL80 auxiliary passed on Counter")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(43)
    main()
