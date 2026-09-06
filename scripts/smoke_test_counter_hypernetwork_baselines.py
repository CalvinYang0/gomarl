#!/usr/bin/env python3
"""Matched Counter hypernetwork controls: construction, semantics and TD update."""
import sys
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_test_counter_transformer_nine import make_case


LABELS = (
    "hyper_hypermarl_id",
    "hyper_cash_obs_type",
    "hyper_rpg_relation",
)


def forward_once(mac, batch, obs=None):
    if obs is not None:
        batch["obs"][:, 0] = obs
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        q = mac.forward(batch, t=0, test_mode=True)
    return q, mac.agent.latest_condition.clone()


def check(label):
    th.manual_seed(17)
    mac, learner, batch, logger = make_case(label)
    agent = mac.agent
    source = agent.counter_hyper_condition_source
    assert source == {
        "hyper_hypermarl_id": "agent_id",
        "hyper_cash_obs_type": "obs_agent_type",
        "hyper_rpg_relation": "rpg_relation",
    }[label]
    assert not learner.mask_parameter_relation_active
    assert not learner.temporal_param_auxiliary_active
    assert not learner.random_drop_auxiliary_active
    assert not learner.gate_regularization_active

    original = batch["obs"][:, 0].clone()
    q_a, condition_a = forward_once(mac, batch, original)
    changed = original + th.randn_like(original)
    q_b, condition_b = forward_once(mac, batch, changed)
    if source == "agent_id":
        assert th.allclose(condition_a, condition_b)
        assert not th.allclose(condition_a[:, 0], condition_a[:, 1])
        # Observation still affects Q through the common Transformer input.
        assert not th.allclose(q_a, q_b)
    else:
        assert not th.allclose(condition_a, condition_b)
    if source == "obs_agent_type":
        assert agent.counter_agent_types.tolist() == [0, 1, 2, 2]
        repeated_obs = original[:, :1].expand(-1, agent.n_agents, -1).clone()
        _, typed_condition = forward_once(mac, batch, repeated_obs)
        assert not th.allclose(typed_condition[:, 0], typed_condition[:, 1])
    if source == "rpg_relation":
        assert agent.counter_rpg_condition_encoder is not None
    assert agent.counter_transformer_policy_projection is not None
    assert agent.rpg_relation_capturer.relation_encoder_style == "attention_only"

    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=300000, episode_num=2)
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    assert not logger.stats.get("loss_mask_parameter_relation")
    assert not logger.stats.get("loss_random_drop_td_auxiliary")
    print(label + ": condition semantics + two TD/QMIX updates OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    for name in LABELS:
        check(name)
    print("3/3 matched hypernetwork baselines passed")
