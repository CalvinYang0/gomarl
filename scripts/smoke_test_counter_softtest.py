#!/usr/bin/env python3
"""Verify soft test execution while training remains identical to KL80aux."""
from pathlib import Path
import sys

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import (  # noqa: E402
    PROFILES,
    experiment_overrides,
)
from ozstar_submit_counter_transformer_nine import build_plans  # noqa: E402
from smoke_test_counter_transformer_nine import check, make_case  # noqa: E402


LABEL = "relation_kl80aux_softtest"


def check_soft_test_gate():
    ordinary_overrides = experiment_overrides("relation_kl80aux")
    soft_overrides = experiment_overrides(LABEL)
    assert {
        key for key in ordinary_overrides if ordinary_overrides[key] != soft_overrides[key]
    } == {"clean_model_type"}

    th.manual_seed(7)
    ordinary_mac, _, batch, _ = make_case("relation_kl80aux")
    th.manual_seed(7)
    soft_mac, _, _, _ = make_case(LABEL)
    ordinary_state = ordinary_mac.agent.state_dict()
    soft_state = soft_mac.agent.state_dict()
    assert ordinary_state.keys() == soft_state.keys()
    assert all(th.equal(ordinary_state[key], soft_state[key]) for key in ordinary_state)

    def forward(mac, test_mode):
        mac.set_dynamic_branch_gate_t_env(300000)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(123)
        output = mac.forward(batch, t=0, test_mode=test_mode).detach().clone()
        capturer = mac.agent.rpg_relation_capturer
        return (
            output,
            capturer.latest_dynamic_branch_gates_graph.detach().clone(),
            capturer.latest_dynamic_branch_probabilities_graph.detach().clone(),
        )

    ordinary_train, _, _ = forward(ordinary_mac, False)
    soft_train, _, _ = forward(soft_mac, False)
    assert th.equal(ordinary_train, soft_train)

    ordinary_test, ordinary_gate, ordinary_probability = forward(ordinary_mac, True)
    soft_test, soft_gate, soft_probability = forward(soft_mac, True)
    assert ordinary_test.shape == soft_test.shape
    # The ordinary test path is hard thresholding; the control applies the
    # same learned probabilities directly as deterministic soft weights.
    assert th.all((ordinary_gate == 0) | (ordinary_gate == 1))
    assert th.equal(soft_gate, soft_probability)
    assert th.all((soft_gate >= 0) & (soft_gate <= 1))
    assert ((soft_gate > 0) & (soft_gate < 1)).any()
    assert th.equal(ordinary_probability, soft_probability)
    assert soft_mac.agent.rpg_relation_capturer.latest_kl80_auxiliary_mask is None
    print(
        "Soft-test control preserves training exactly and applies p (not p>0.5) at test",
        flush=True,
    )


if __name__ == "__main__":
    th.set_num_threads(1)
    repo = ROOT
    plan, = build_plans(repo, [LABEL])
    assert plan["exports"]["TEST_INTERVAL"] == "10000"
    assert len(build_plans(repo)) == len(PROFILES) == 9
    check_soft_test_gate()
    check(LABEL)
    print("Soft test-gate control passed (synthetic episodes; simulator not exercised)")
