#!/usr/bin/env python3
"""Exercise the fixed-head VDN paper baseline on GRF and SMAC shapes."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test_counter_transformer_nine import make_case
from modules.mixers.vdn import VDNMixer


SCENES = (
    "academy_counterattack_easy",
    "academy_pass_and_shoot_with_keeper",
    "5m_vs_6m",
    "MMM2",
)


def check(scene):
    th.manual_seed(29)
    mac, learner, batch, logger = make_case(
        "baseline",
        scene,
        config_overrides={
            "clean_model_type": "qmix_minimal",
            "mixer": "vdn",
        },
    )
    assert mac.agent.model_type == "qmix_minimal"
    assert isinstance(learner.mixer, VDNMixer)
    assert isinstance(learner.target_mixer, VDNMixer)
    agent_values = th.randn(2, 4, mac.args.n_agents)
    mixed = learner.mixer(agent_values, None)
    assert mixed.shape == (2, 4, 1)
    assert th.allclose(mixed.squeeze(-1), agent_values.sum(dim=2))

    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=300000, episode_num=2)
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(scene + ": fixed-head VDN forward/backward OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    for scene_name in SCENES:
        check(scene_name)
    print("4/4 paper VDN checks passed")
