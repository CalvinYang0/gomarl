#!/usr/bin/env python3
"""Real nine-model learner updates on synthetic padded episodes, no simulator."""
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.clean_controller import CleanMAC
from learners.clean_learner import CleanLearner
from modules.agents.counter_transformer_suite import ALL_PROFILES, PROFILES, experiment_overrides
from utils.logging import Logger


def make_case(label):
    config = {}
    for filename in ("default.yaml", "envs/academy_counterattack_easy.yaml", "algs/clean_hyper.yaml"):
        config.update(yaml.safe_load((ROOT / "src/config" / filename).read_text()))
    config.update(experiment_overrides(label))
    config.update(n_agents=4, n_actions=19, state_shape=30, obs_shape=30,
                  rnn_hidden_dim=16, hypernet_embed=16, clean_condition_dim=16,
                  use_cuda=False, device="cpu", batch_size=2, batch_size_run=2,
                  obs_last_action=False, obs_agent_id=False, learner_log_interval=1)
    args = SimpleNamespace(**config)
    scheme = {
        "obs": {"vshape": 30, "group": "agents"},
        "state": {"vshape": 30},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (19,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    batch = EpisodeBatch(scheme, {"agents": 4}, 2, 5,
                         preprocess={"actions": ("actions_onehot", [OneHot(19)])})
    batch.update({"obs": th.randn(2, 5, 4, 30), "state": th.randn(2, 5, 30),
                  "actions": th.randint(0, 19, (2, 5, 4, 1)),
                  "avail_actions": th.ones(2, 5, 4, 19, dtype=th.int),
                  "reward": th.rand(2, 5, 1), "terminated": th.zeros(2, 5, 1, dtype=th.uint8)})
    batch["terminated"][1, 2] = 1
    batch["filled"][1, 3:] = 0
    logger = Logger(logging.getLogger("transformer-nine-test"))
    mac = CleanMAC(batch.scheme, {"agents": 4}, args)
    learner = CleanLearner(mac, batch.scheme, logger, args)
    return mac, learner, batch, logger


def check(label):
    th.manual_seed(7)
    flags = ALL_PROFILES[label]
    mac, learner, batch, logger = make_case(label)
    capturer = mac.agent.rpg_relation_capturer
    branch = "linear" if flags.get("branch") == "linear" else "attention"
    assert capturer.relation_encoder_style == branch + "_only"
    assert (capturer.dynamic_branch_gate is not None) == bool(flags.get("gate"))
    assert learner.mask_parameter_relation_active == bool(flags.get("relation"))
    assert learner.temporal_param_auxiliary_active == bool(flags.get("temporal"))
    assert learner.random_drop_auxiliary_active == bool(flags.get("aux"))
    assert learner.gate_regularization_active == bool(flags.get("kl"))
    assert learner.mask_parameter_relation_pairing == "fixed"
    assert learner.mask_parameter_relation_mask_source == "probability"
    # Both before warmup and after warmup must execute actual optimiser steps.
    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=300000, episode_num=2)
    assert all(th.isfinite(p).all() for p in mac.parameters())
    if flags.get("relation"):
        assert logger.stats["loss_mask_parameter_relation"][-1][1] > 0
        assert learner.mask_parameter_relation_coef == 1.0
        assert logger.stats["weighted_loss_mask_parameter_relation"][-1][1] == logger.stats["loss_mask_parameter_relation"][-1][1]
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in capturer.dynamic_branch_gate.parameters())
    if flags.get("temporal"):
        assert logger.stats["loss_temporal_parameter"][-1][1] > 0
        assert learner.mask_parameter_relation_temporal_coef == 1.0
        assert logger.stats["weighted_loss_temporal_parameter"][-1][1] == logger.stats["loss_temporal_parameter"][-1][1]
    if flags.get("kl"):
        assert logger.stats["loss_aux"][-1][1] > 0
    if flags.get("aux"):
        assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0
        assert learner.random_drop_auxiliary_coef == 1.0
    if flags.get("aux") == "kl80":
        assert logger.stats["loss_" + learner.kl_auxiliary_tag + "_random_auxiliary"][-1][1] > 0
        assert not capturer.kl80_auxiliary_enabled
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in capturer.kl80_auxiliary_gate.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in capturer.dynamic_branch_gate.parameters())
    if not flags.get("relation"):
        assert not logger.stats.get("loss_mask_parameter_relation")
    # Evaluation capture is exercised through select_actions, including the
    # no-gate baseline. Every model must provide real generated-head vectors.
    mac.init_hidden(batch.batch_size)
    mac.reset_test_gate_probability_trajectory()
    with th.no_grad():
        for t in range(5):
            mac.select_actions(batch, t, 300000, test_mode=True)
    trajectory = mac.pop_test_gate_probability_trajectory()
    assert trajectory is not None
    assert trajectory["generated_parameter_vectors"].shape[0] == 5
    assert set(trajectory["branches"]) == {branch}
    heatmap = th.tensor(trajectory["agent_probability_branches"][branch])
    assert heatmap.shape == (5, 4, 30)
    assert ((heatmap >= 0) & (heatmap <= 1)).all()
    if label == "baseline":
        assert (heatmap == 1).all()
    if flags.get("test_open"):
        assert (mac.agent.latest_dynamic_branch_gates_graph == 1).all()
        assert (heatmap < 1).any()
    # Test actual rendering and W&B buffer keys without uploading dummy data.
    logger.use_wandb = True
    logger.wandb_current_t = 300000
    logger.wandb_current_data = {}
    def capture_image(fig):
        fig.canvas.draw()
        return "rendered"
    logger.wandb_module = SimpleNamespace(Image=capture_image)
    logger.log_test_gate_probability_trajectory(trajectory, 300000)
    assert "test_mask_probability_heatmap_" + branch in logger.wandb_current_data
    if flags.get("aux") == "kl80":
        assert "test_mask_probability_heatmap_auxiliary_{}_{}".format(learner.kl_auxiliary_tag, branch) in logger.wandb_current_data
    if flags.get("aux") == "fixed_concrete":
        assert "test_mask_probability_heatmap_auxiliary_fixed80_attention" in logger.wandb_current_data
        assert "test_mask_probability_heatmap_auxiliary_kl80_attention" not in logger.wandb_current_data
        assert not any(p.requires_grad for p in capturer.kl80_auxiliary_gate.parameters())
    assert any("parameter_pca" in key for key in logger.wandb_current_data)
    assert "test_dynamic_gate_trajectory" in logger.wandb_current_data
    print(label + ": learner forward/backward + flags + PCA + trajectory + heatmap OK", flush=True)


def check_test_open_and_auxiliary_isolation():
    ordinary, _, batch, _ = make_case("kl80")
    opened, _, _, _ = make_case("kl80_test_open")
    with th.no_grad():
        last = ordinary.agent.rpg_relation_capturer.dynamic_branch_gate.gate_network[-1]
        last.weight.zero_()
        last.bias.fill_(math.log(0.2 / 0.8))
    opened.agent.load_state_dict(ordinary.agent.state_dict())
    def forward(mac, test):
        mac.set_dynamic_branch_gate_t_env(300000)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(123)
        return mac.forward(batch, t=0, test_mode=test)
    assert th.allclose(forward(ordinary, False), forward(opened, False))
    ordinary_q, open_q = forward(ordinary, True), forward(opened, True)
    assert not th.allclose(ordinary_q, open_q)
    assert (ordinary.agent.latest_dynamic_branch_gates_graph[1] == 0).all()
    assert (opened.agent.latest_dynamic_branch_gates_graph == 1).all()
    auxiliary, _, _, _ = make_case("relation_kl80aux")
    capturer = auxiliary.agent.rpg_relation_capturer
    before = forward(auxiliary, False).detach()
    with th.no_grad():
        capturer.kl80_auxiliary_gate.gate_network[-1].bias.fill_(-4)
    assert th.allclose(before, forward(auxiliary, False))  # Not on main path.
    capturer.kl80_auxiliary_enabled = True
    assert not th.allclose(before, forward(auxiliary, False))
    assert th.allclose(forward(auxiliary, True), forward(auxiliary, True))
    print("KL80 test-open affects test only; independent auxiliary affects auxiliary only: OK")


if __name__ == "__main__":
    th.set_num_threads(1)
    for profile in PROFILES:
        check(profile)
    check_test_open_and_auxiliary_isolation()
    print("9/9 passed (synthetic episodes; simulator not exercised)")
