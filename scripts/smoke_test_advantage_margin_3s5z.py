#!/usr/bin/env python3
"""Exercise selected Advantage-mask profiles on one requested SMAC map."""
import logging
import os
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


LABELS = (
    "relation_advantage_margin_kl80aux_gradsep",
    "relation_advantage_margin_kl80aux",
    "relation_advantage_weighted_kl80aux",
)
SCENES = {
    "3s5z": "3s5z_vs_3s6z",
    "corridor": "corridor",
}


def main():
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    scene_key = os.environ.get("TARGET_SCENE", "3s5z")
    if scene_key not in SCENES:
        raise ValueError("TARGET_SCENE must be 3s5z or corridor")
    scene = SCENES[scene_key]
    config = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label in LABELS:
        overrides = experiment_overrides(label, "smac")
        assert not set(overrides) - set(config)
        assert overrides["clean_dual_gate_test"] is True
        assert overrides["clean_main_td_coef"] == 1.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        assert overrides["clean_mask_parameter_relation_coef"] == 1.0
        assert overrides["clean_kl_auxiliary_force_main_open"] is True
        assert overrides["clean_advantage_margin_auxiliary"] is True
        assert overrides["clean_mask_nomask_gradient_separation"] is (
            label.endswith("_gradsep")
        )
        assert overrides["clean_advantage_margin_weight_by_teacher"] is (
            "_weighted_" in label
        )
        check(label, scene)
        check_smac_semantics(scene, label)
    print("Selected Advantage-mask profiles passed on SMAC " + scene_key)


if __name__ == "__main__":
    main()
