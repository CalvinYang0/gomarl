#!/usr/bin/env python3
"""Exercise the gradient-isolated relcoef=.1 Counter profile."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "relation_all4_relcoef01_gradsep"


def check_gradient_router(learner):
    """Prove that cross-connected losses reach only their assigned group."""
    non_gate = learner.gradient_separation_non_gate_parameters[0]
    gate = learner.gradient_separation_gate_parameters[0]
    learner.optimiser.zero_grad()
    # Each loss deliberately contains both parameter classes. Correct routing
    # must ignore the cross-connected term rather than merely detach a pass.
    teacher_loss = non_gate.reshape(-1)[0] + 7.0 * gate.reshape(-1)[0]
    masked_loss = 11.0 * non_gate.reshape(-1)[0] + 3.0 * gate.reshape(-1)[0]
    learner._backward_mask_nomask_gradient_separated(
        teacher_loss, masked_loss
    )
    assert th.allclose(
        non_gate.grad.reshape(-1)[0], non_gate.new_tensor(1.0)
    )
    assert th.allclose(gate.grad.reshape(-1)[0], gate.new_tensor(3.0))
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
    assert overrides["clean_mask_nomask_gradient_separation"] is True
    assert math.isclose(overrides["clean_main_td_coef"], 1.0)
    assert math.isclose(overrides["clean_nomask_td_auxiliary_coef"], 1.0)
    assert math.isclose(
        overrides["clean_mask_parameter_relation_coef"], 0.1
    )
    assert overrides["clean_kl_auxiliary_force_main_open"] is True
    check(LABEL)

    _, learner, batch, logger = make_case(LABEL)
    assert learner.mask_nomask_gradient_separation
    assert learner.gradient_separation_gate_parameters
    assert learner.gradient_separation_non_gate_parameters
    gate_ids = {id(parameter) for parameter in
                learner.gradient_separation_gate_parameters}
    non_gate_ids = {id(parameter) for parameter in
                    learner.gradient_separation_non_gate_parameters}
    assert not gate_ids.intersection(non_gate_ids)
    assert gate_ids.union(non_gate_ids) == {
        id(parameter) for parameter in learner.params
    }
    check_gradient_router(learner)

    learner.train(batch, t_env=300000, episode_num=1)
    assert logger.stats["loss_td"][-1][1] > 0.0
    assert logger.stats["loss_nomask_td_auxiliary"][-1][1] > 0.0
    assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0.0
    assert logger.stats["loss_mask_parameter_relation"][-1][1] >= 0.0

    # Before the 250k gate warm-up ends, the gate is intentionally force-open
    # and its routed loss may have no gate graph. That phase must still train
    # the no-mask teacher without being mistaken for a broken connection.
    _, warmup_learner, warmup_batch, warmup_logger = make_case(LABEL)
    warmup_learner.train(warmup_batch, t_env=0, episode_num=1)
    assert warmup_logger.stats["loss_nomask_td_auxiliary"][-1][1] > 0.0
    print(
        "Counter relation all4 relcoef01 gradient separation: "
        "routing + learner update OK"
    )


if __name__ == "__main__":
    main()
