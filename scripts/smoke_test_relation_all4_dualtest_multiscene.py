#!/usr/bin/env python3
"""Exercise the relation-all4 paired gate evaluation on 3s5z and corridor."""
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


LABEL = "relation_all4_dualtest"
SCENES = ("3s5z_vs_3s6z", "corridor")


def main():
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    config = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    overrides = experiment_overrides(LABEL, "smac")
    assert not set(overrides) - set(config)
    assert overrides["clean_dual_gate_test"] is True
    assert overrides["clean_main_td_coef"] == 1.0
    assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
    assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
    assert overrides["clean_mask_parameter_relation_coef"] == 1.0
    assert overrides["clean_kl_auxiliary_force_main_open"] is True
    for scene in SCENES:
        check(LABEL, scene)
        check_smac_semantics(scene, LABEL)
    print("relation_all4 dual-test passed on 3s5z and corridor")


if __name__ == "__main__":
    main()
