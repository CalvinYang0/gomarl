#!/usr/bin/env python3
"""Exercise gate-only KL80 auxiliary TD routing on Counter."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "relation_kl80aux_kltd_gateonly"


def check_auxiliary_gradient_router(learner):
    capturer = learner.mac.agent.rpg_relation_capturer
    main_gate_parameter = next(capturer.dynamic_branch_gate.parameters())
    auxiliary_gate_parameter = next(capturer.kl80_auxiliary_gate.parameters())
    isolated_ids = {
        id(parameter)
        for parameter in learner.kl_auxiliary_td_gate_only_parameters
    }
    assert id(main_gate_parameter) in isolated_ids
    assert id(auxiliary_gate_parameter) in isolated_ids
    non_gate_parameter = next(
        parameter for parameter in learner.params
        if id(parameter) not in isolated_ids
    )

    learner.optimiser.zero_grad()
    main_loss = (
        non_gate_parameter.reshape(-1)[0]
        + 2.0 * main_gate_parameter.reshape(-1)[0]
    )
    # Deliberately connect the auxiliary objective to every parameter class.
    # Correct routing must discard its factor-11 main-network gradient while
    # accumulating its gate gradients on top of the ordinary main pass.
    auxiliary_loss = (
        11.0 * non_gate_parameter.reshape(-1)[0]
        + 3.0 * main_gate_parameter.reshape(-1)[0]
        + 5.0 * auxiliary_gate_parameter.reshape(-1)[0]
    )
    learner._backward_main_and_gate_only_auxiliary(
        main_loss,
        auxiliary_loss,
        gate_parameters=learner.kl_auxiliary_td_gate_only_parameters,
    )
    assert th.allclose(
        non_gate_parameter.grad.reshape(-1)[0],
        non_gate_parameter.new_tensor(1.0),
    )
    assert th.allclose(
        main_gate_parameter.grad.reshape(-1)[0],
        main_gate_parameter.new_tensor(5.0),
    )
    assert th.allclose(
        auxiliary_gate_parameter.grad.reshape(-1)[0],
        auxiliary_gate_parameter.new_tensor(5.0),
    )
    learner.optimiser.zero_grad()


def main():
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    overrides = experiment_overrides(LABEL)
    missing = set(overrides) - set(sacred)
    assert not missing, "{} has unregistered keys: {}".format(
        LABEL, sorted(missing)
    )
    assert overrides["clean_kl_auxiliary_td_gate_only"] is True
    assert overrides["clean_dual_gate_test"] is True
    assert overrides["clean_mask_nomask_gradient_separation"] is False
    assert math.isclose(overrides["clean_main_td_coef"], 1.0)
    assert math.isclose(overrides["clean_nomask_td_auxiliary_coef"], 0.0)
    assert math.isclose(overrides["clean_random_drop_auxiliary_coef"], 1.0)
    assert math.isclose(
        overrides["clean_mask_parameter_relation_coef"], 1.0
    )
    check(LABEL)

    _, learner, batch, logger = make_case(LABEL)
    assert learner.kl_auxiliary_td_gate_only
    assert learner.kl_auxiliary_td_gate_only_parameters
    check_auxiliary_gradient_router(learner)
    learner.train(batch, t_env=300000, episode_num=1)
    assert logger.stats["loss_td"][-1][1] > 0.0
    assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0.0
    assert logger.stats["loss_kl80_random_auxiliary"][-1][1] >= 0.0
    assert logger.stats["loss_mask_parameter_relation"][-1][1] >= 0.0
    print(
        "Counter relation KL80 auxiliary TD gate-only: "
        "routing + learner update OK"
    )


if __name__ == "__main__":
    main()
