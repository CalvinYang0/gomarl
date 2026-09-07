#!/usr/bin/env python3
"""Matched Transformer-only/QMIX mixer-KL80 auxiliary regression test."""
from pathlib import Path

import torch as th

from ozstar_submit_counter_transformer_nine import build_plans
from smoke_test_counter_transformer_nine import check, make_case


def main():
    th.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    baseline_plan, auxiliary_plan = build_plans(
        repo, ["baseline", "mixer_kl80aux"]
    )
    for key in baseline_plan["exports"]:
        if key not in {"MODEL_TYPE", "RUN_NAME", "GROUP_NAME", "EXTRA_ARGS"}:
            assert baseline_plan["exports"][key] == auxiliary_plan["exports"][key], key

    th.manual_seed(17)
    baseline_mac, baseline_learner, batch, _ = make_case("baseline")
    th.manual_seed(17)
    auxiliary_mac, auxiliary_learner, _, logger = make_case("mixer_kl80aux")
    assert baseline_mac.agent.state_dict().keys() == auxiliary_mac.agent.state_dict().keys()
    assert all(
        th.equal(baseline_mac.agent.state_dict()[key], auxiliary_mac.agent.state_dict()[key])
        for key in baseline_mac.agent.state_dict()
    )
    assert all(
        th.equal(baseline_learner.mixer.state_dict()[key], auxiliary_learner.mixer.state_dict()[key])
        for key in baseline_learner.mixer.state_dict()
    )
    assert not baseline_learner.mixer_kl80_auxiliary_active
    assert auxiliary_learner.mixer_kl80_auxiliary_active
    assert auxiliary_learner.mixer_kl80_gate is not None

    auxiliary_learner.train(batch, t_env=10, episode_num=1)
    assert not any(p.grad is not None for p in auxiliary_learner.mixer_kl80_gate.parameters())
    auxiliary_learner.train(batch, t_env=300000, episode_num=2)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in auxiliary_learner.mixer_kl80_gate.parameters()
    )
    assert logger.stats["loss_mixer_kl80_td_auxiliary"][-1][1] > 0
    assert logger.stats["loss_mixer_kl80_prior"][-1][1] >= -1e-6
    assert logger.stats["mixer_kl80_keep_prior"][-1][1] == 0.8
    probabilities = auxiliary_learner.latest_mixer_kl80_probability
    assert probabilities.shape == (batch.batch_size, batch.max_seq_length - 1, auxiliary_mac.args.n_agents)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    check("mixer_kl80aux")
    print("Transformer-only + QMIX KL80 auxiliary: matched base, update and logs OK")


if __name__ == "__main__":
    main()
