#!/usr/bin/env python3
"""Validate HyperSelect on the five paper scenes not already run on Counter."""
import logging
from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test_counter_transformer_nine import check
from smoke_test_trans9_multiscene import check_smac_semantics


LABEL = "relation_advantage_qvalue_augtd_nomasktd"
GRF_SCENES = (
    "academy_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
)
SMAC_SCENES = (
    "3s5z_vs_3s6z",
    "5m_vs_6m",
    "MMM2",
)


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    for scene in GRF_SCENES:
        check(LABEL, scene)
    for scene in SMAC_SCENES:
        check(LABEL, scene)
        check_smac_semantics(scene, LABEL)
    print("HyperSelect passed on all five additional paper scenes")


if __name__ == "__main__":
    main()
