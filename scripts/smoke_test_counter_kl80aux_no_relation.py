#!/usr/bin/env python3
"""Matched architecture, isolated loss removal, real updates and diagnostic plots."""
import torch as th

from smoke_test_counter_transformer_nine import check, make_case
from modules.agents.counter_transformer_suite import PROFILES, experiment_overrides
from ozstar_submit_counter_transformer_nine import build_plans
from pathlib import Path


def check_matched_control():
    original = experiment_overrides("relation_kl80aux")
    control = experiment_overrides("obs_gate_kl80aux")
    assert {key for key in original if original[key] != control[key]} == {
        "clean_model_type", "clean_mask_parameter_relation_coef",
    }
    assert control["clean_mask_parameter_relation_coef"] == 0.0
    assert original["clean_mask_parameter_relation_coef"] == 1.0
    repo = Path(__file__).resolve().parents[1]
    assert len(build_plans(repo)) == len(PROFILES) == 9
    plans = build_plans(repo, ["relation_kl80aux", "obs_gate_kl80aux"])
    for key in plans[0]["exports"]:
        if key not in {"MODEL_TYPE", "RUN_NAME", "GROUP_NAME", "EXTRA_ARGS"}:
            assert plans[0]["exports"][key] == plans[1]["exports"][key], key
    th.manual_seed(7)
    original_mac, _, batch, _ = make_case("relation_kl80aux")
    th.manual_seed(7)
    control_mac, learner, _, _ = make_case("obs_gate_kl80aux")
    a, b = original_mac.agent.state_dict(), control_mac.agent.state_dict()
    assert a.keys() == b.keys()
    assert all(th.equal(a[key], b[key]) for key in a)
    assert not learner.mask_parameter_relation_active
    assert learner.mask_parameter_relation_coef == 0.0
    assert not learner.temporal_param_auxiliary_active
    assert not learner.gate_regularization_active  # No KL on the primary gate.
    assert learner.random_drop_auxiliary_active and learner.kl80_random_drop_auxiliary
    results = []
    for mac in (original_mac, control_mac):
        mac.set_dynamic_branch_gate_t_env(300000)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(99)
        results.append(mac.forward(batch, t=0, test_mode=False).detach())
    assert th.equal(*results)
    print("Matched initial weights, forward outputs, resource settings; only relation disabled", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    check_matched_control()
    check("obs_gate_kl80aux")
    print("No-relation control passed (synthetic episodes; simulator not exercised)")
