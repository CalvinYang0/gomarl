#!/usr/bin/env python3
"""Exercise Direct-Q + AugTD + NoMaskTD on 3s5z and corridor."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check
from smoke_test_trans9_multiscene import check_smac_semantics


LABEL = "relation_advantage_qvalue_augtd_nomasktd"
SCENES = ("3s5z_vs_3s6z", "corridor")


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    overrides = experiment_overrides(LABEL, "smac")
    missing = set(overrides) - set(sacred)
    assert not missing, "unregistered keys: {}".format(sorted(missing))
    assert overrides["clean_main_td_coef"] == 0.0
    assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
    assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
    assert overrides["clean_random_drop_auxiliary_identity_warmup"] is True
    assert overrides["clean_kl_auxiliary_force_main_open"] is False
    assert overrides["clean_advantage_margin_auxiliary"] is True
    assert overrides["clean_advantage_objective"] == "action_q"
    assert overrides["clean_mask_parameter_relation_coef"] == 0.0
    assert overrides["clean_dynamic_branch_gate_warmup_steps"] == 250000
    assert overrides["clean_importance_auxiliary_warmup_steps"] == 250000
    assert overrides["clean_advantage_margin_warmup_steps"] == 250000
    assert overrides["clean_dual_gate_test"] is True

    for scene in SCENES:
        check(LABEL, scene)
        check_smac_semantics(scene, LABEL)
    print("Direct-Q + AugTD + NoMaskTD passed on 3s5z and corridor")


if __name__ == "__main__":
    main()
