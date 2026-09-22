#!/usr/bin/env python3
"""Validate the matched ID-HyperNet control on the two paper SMAC maps."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test_counter_transformer_nine import make_case


LABEL = "hyper_hypermarl_id"
SCENES = ("5m_vs_6m", "MMM2")


def forward_once(mac, batch, obs):
    batch["obs"][:, 0] = obs
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        q_values = mac.forward(batch, t=0, test_mode=True)
    return q_values, mac.agent.latest_condition.clone()


def check(scene):
    th.manual_seed(23)
    mac, learner, batch, logger = make_case(LABEL, scene)
    agent = mac.agent
    assert agent.counter_hyper_condition_source == "agent_id"
    assert agent.counter_id_condition_encoder is not None
    assert agent.counter_transformer_policy_projection is not None

    original = batch["obs"][:, 0].clone()
    q_a, condition_a = forward_once(mac, batch, original)
    q_b, condition_b = forward_once(mac, batch, original + th.randn_like(original))
    # The generated-head condition is identity-only, while observations still
    # affect the shared Transformer policy representation and therefore Q.
    assert th.allclose(condition_a, condition_b)
    assert not th.allclose(condition_a[:, 0], condition_a[:, 1])
    assert not th.allclose(q_a, q_b)

    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=300000, episode_num=2)
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(scene + ": matched ID-HyperNet forward/backward OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    for scene_name in SCENES:
        check(scene_name)
    print("2/2 SMAC ID-HyperNet checks passed")
