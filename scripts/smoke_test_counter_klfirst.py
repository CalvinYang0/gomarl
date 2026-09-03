#!/usr/bin/env python3
"""Verify the KL-first auxiliary mask ordering on synthetic padded episodes."""
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


LABEL = "relation_kl80aux_klfirst"


def check_matched_ordering():
    ordinary = experiment_overrides("relation_kl80aux")
    kl_first = experiment_overrides(LABEL)
    assert {
        key for key in ordinary if ordinary[key] != kl_first[key]
    } == {"clean_model_type"}

    th.manual_seed(7)
    ordinary_mac, _, batch, _ = make_case("relation_kl80aux")
    th.manual_seed(7)
    kl_first_mac, _, _, _ = make_case(LABEL)
    assert ordinary_mac.agent.state_dict().keys() == kl_first_mac.agent.state_dict().keys()
    assert all(
        th.equal(ordinary_mac.agent.state_dict()[key], kl_first_mac.agent.state_dict()[key])
        for key in ordinary_mac.agent.state_dict()
    )

    def capture(mac, auxiliary, test_mode=False):
        capturer = mac.agent.rpg_relation_capturer
        capturer.kl80_auxiliary_enabled = auxiliary
        aux_masks = []
        main_inputs = []
        aux_hook = capturer.kl80_auxiliary_gate.register_forward_hook(
            lambda _module, _inputs, output: aux_masks.append(output[0].detach().clone())
        )
        main_hook = capturer.dynamic_branch_gate.register_forward_pre_hook(
            lambda _module, inputs: main_inputs.append(inputs[0].detach().clone())
        )
        try:
            mac.set_dynamic_branch_gate_t_env(300000)
            mac.init_hidden(batch.batch_size)
            th.manual_seed(123)
            output = mac.forward(batch, t=0, test_mode=test_mode).detach().clone()
        finally:
            aux_hook.remove()
            main_hook.remove()
        return output, aux_masks[-1], main_inputs[-1]

    ordinary_out, ordinary_aux_mask, ordinary_main_input = capture(ordinary_mac, True)
    first_out, first_aux_mask, first_main_input = capture(kl_first_mac, True)
    raw_obs = batch["obs"][:, 0]
    # The ordinary control always feeds raw obs to the main gate. KL-first
    # feeds the attention branch's auxiliary-dropped obs to that same gate.
    assert th.equal(ordinary_main_input, raw_obs)
    assert th.equal(first_main_input, raw_obs * first_aux_mask[1])
    assert not th.equal(first_main_input, raw_obs)
    # The two orderings consume the stochastic gates in a different order, so
    # their sampled auxiliary masks need not be bit-identical under one seed.
    assert ordinary_aux_mask.shape == first_aux_mask.shape
    # The relation encoder may map the particular synthetic draw to the same
    # output despite a different gate input; ordering is established directly
    # by the captured tensors above.
    assert ordinary_out.shape == first_out.shape

    # Disabling the auxiliary rollout restores the ordinary raw-obs path.
    _, _, disabled_main_input = capture(kl_first_mac, False)
    assert th.equal(disabled_main_input, raw_obs)

    # Semantic evaluation never applies the auxiliary mask, so enabling the
    # auxiliary flag cannot perturb test outputs or the main-gate input.
    test_disabled, _, test_disabled_input = capture(kl_first_mac, False, True)
    test_enabled, _, test_enabled_input = capture(kl_first_mac, True, True)
    assert th.equal(test_disabled, test_enabled)
    assert th.equal(test_disabled_input, raw_obs)
    assert th.equal(test_enabled_input, raw_obs)

    print(
        "KL-first changes only auxiliary order: aux mask -> main gate; main/test raw-obs path preserved",
        flush=True,
    )


if __name__ == "__main__":
    th.set_num_threads(1)
    repo = ROOT
    plan, = build_plans(repo, [LABEL])
    assert plan["exports"]["TEST_INTERVAL"] == "10000"
    assert len(build_plans(repo)) == len(PROFILES) == 9
    check_matched_ordering()
    check(LABEL)
    print("KL-first auxiliary control passed (synthetic episodes; simulator not exercised)")
