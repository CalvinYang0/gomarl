#!/usr/bin/env python3
"""Synthetic QA, never upload as experimental evidence; no SC2 launch."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch as th

from smoke_test_counter_transformer_nine import make_case
from utils.hyper_obs_importance import (
    HyperObsImportanceSession, PREFIX, analyze, head_jacobian, hyper_q, replay_samples, supported,
)
from utils.logging import Logger


ROOT = Path(__file__).resolve().parents[1]


def check_real_path(scene):
    th.manual_seed(43)
    mac, learner, batch, logger = make_case("linear_baseline", scene, production_shapes=True)
    assert supported(mac.agent)
    batch["filled"][1, 3] = 1  # Real post-terminal observation snapshot.
    batch["avail_actions"][0, 1, 0] = 0
    batch["avail_actions"][0, 1, 0, 0] = 1  # Dead agent.
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        for t in range(batch.max_seq_length - 1):
            real_q = mac.forward(batch, t=t, test_mode=True)
            policy_hidden = mac.hidden_states[..., :mac.agent.hidden_dim]
            pure_q = hyper_q(mac.agent, batch["obs"][:, t].reshape(-1, mac.args.obs_shape),
                             policy_hidden.reshape(-1, mac.agent.hidden_dim)).reshape_as(real_q)
            assert th.allclose(real_q, pure_q, atol=1e-6)
    # Check exact parameter derivatives using a finite perturbation.
    encoder = mac.agent.rpg_relation_capturer.dual_linear_encoder
    obs = batch["obs"][0, 0]
    changed = obs.clone()
    changed[:, 8] += .2  # Affine generator: larger step avoids float32 cancellation.
    with th.no_grad():
        for name, jacobian in zip(("hyper_bottleneck_w", "hyper_bottleneck_b", "hyper_out_w", "hyper_out_b"), head_jacobian(mac.agent)):
            layer = getattr(mac.agent, name)
            finite = (layer(encoder(changed)) - layer(encoder(obs))) / .2
            assert th.allclose(finite, jacobian[:, 8].expand_as(finite), atol=2e-5)
    saved_hidden, saved_condition = mac.hidden_states, mac.agent.latest_condition
    parameter = next(mac.agent.parameters())
    parameter.grad = th.ones_like(parameter)
    saved_grad = parameter.grad.clone()
    saved_params = {name: value.clone() for name, value in mac.agent.state_dict().items()}
    saved_batch = {key: batch[key].clone() for key in ("obs", "actions", "avail_actions", "filled")}
    modes = [(module, module.training) for module in mac.agent.modules()]
    rng = th.get_rng_state().clone()
    samples = replay_samples(mac, batch, 2)
    assert not ((samples["episode"] == 0) & (samples["agent"] == 0) & (samples["timestep"] == 1)).any()
    assert not ((samples["episode"] == 1) & (samples["timestep"] > 2)).any()
    with th.no_grad():
        result = analyze(mac.agent, samples, 30)
    assert result["n_probe_states"] <= 30
    assert mac.hidden_states is saved_hidden and mac.agent.latest_condition is saved_condition
    assert th.equal(parameter.grad, saved_grad) and th.equal(rng, th.get_rng_state())
    assert all(module.training == mode for module, mode in modes)
    assert all(th.equal(value, saved_params[name]) for name, value in mac.agent.state_dict().items())
    assert all(th.equal(batch[key], value) for key, value in saved_batch.items())
    print("PASS:", scene, "pure Q == production Q; exact Jacobian; masks; state/RNG/grad/parameter isolation")


def known_feature_and_output():
    mac, _, batch, logger = make_case("linear_baseline", "5m_vs_6m", production_shapes=True)
    assert supported(mac.agent)
    names = list(mac.agent.rpg_relation_capturer.semantic_names)
    important = names.index("enemy_0_health")
    with th.no_grad():
        encoder = mac.agent.rpg_relation_capturer.dual_linear_encoder
        encoder.weight.zero_()
        encoder.bias.zero_()
        encoder.weight[0, important] = 1
        for name in ("hyper_bottleneck_w", "hyper_bottleneck_b", "hyper_out_w", "hyper_out_b"):
            getattr(mac.agent, name).weight.zero_()
            getattr(mac.agent, name).bias.zero_()
        mac.agent.hyper_out_b.weight[6, 0] = 1
        mac.agent.hyper_out_b.weight[7, 0] = -1
        mac.agent.hyper_out_b.bias[7] = .8
        batch["obs"][0, :, :, important] = .1
        batch["obs"][1, :, :, important] = .9
        batch["avail_actions"][:] = 0
        batch["avail_actions"][..., 6:8] = 1
        batch["filled"][1, 3] = 1
    samples = replay_samples(mac, batch, 2)
    result = analyze(mac.agent, samples, 50)
    assert abs(result["head_jacobian_l2"][important] - 2 ** .5) < 1e-6
    assert abs(result["gap_gradient_abs_mean"][important] - 2) < 1e-6
    assert all(value == 0 for i, value in enumerate(result["gap_gradient_abs_mean"]) if i != important)
    ablation = {row["group"]: row for row in result["group_mean_replacement"]}
    assert ablation["enemy_health"]["action_flip_rate"] > 0
    assert ablation["ally_geometry"]["action_flip_rate"] == 0
    args = mac.args
    args.test_hyper_obs_importance = True
    args.test_hyper_obs_importance_interval = 1000000
    args.test_hyper_obs_importance_episodes = 2
    args.test_hyper_obs_importance_samples = 50
    args.seed, args.unique_token = 43, "SYNTHETIC_QA_NOT_EXPERIMENT"
    args.local_results_path = str(ROOT / "output/hyper-obs-importance-smoke")
    args.wandb_run_name = "SYNTHETIC_ONLY_KNOWN_FEATURE"
    logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, 1000000, {}
    logger.wandb_module = SimpleNamespace(Image=lambda path: path)
    logger.wandb = SimpleNamespace(log=lambda *args, **kwargs: None)
    session = HyperObsImportanceSession(args, mac, logger)
    session.begin(500000)
    assert session.parts is None
    session.begin(1000000)
    session.consume(batch)
    session.consume(batch)  # Episode limit must prevent double collection.
    assert session.collected == 2
    session.finish()
    assert logger.stats[PREFIX + "failed"][-1][1] == 0
    for key in ("head_sensitivity", "decision_sensitivity", "group_ablation"):
        assert Path(logger.wandb_current_data[PREFIX + key]).is_file()
    directory = Path(logger.wandb_current_data[PREFIX + "head_sensitivity"]).parent
    saved = json.loads((directory / "importance.json").read_text())
    assert saved["run_name"] == "SYNTHETIC_ONLY_KNOWN_FEATURE"
    assert all(Logger._wandb_metric_allowed(key) for key in logger.wandb_current_data)
    # Equal Qs must not create action flips solely from top-k tie breaking.
    with th.no_grad():
        mac.agent.hyper_out_b.weight.zero_()
        mac.agent.hyper_out_b.bias.zero_()
    tied = analyze(mac.agent, samples, 50)
    assert all(row["action_flip_rate"] == 0 for row in tied["group_mean_replacement"])
    # Malformed diagnostic data is reported, never an uncaught training error.
    session.begin(2000000)
    session.consume(None)
    session.finish()
    assert logger.stats[PREFIX + "failed"][-1][1] == 1
    print("PASS: known health feature isolated; ablation effects; scheduling/limits; 3 PNGs + JSON/CSV; W&B keys")
    print("Synthetic QA only:", directory)


if __name__ == "__main__":
    th.set_num_threads(1)
    for scene in ("5m_vs_6m", "8m_vs_9m"):
        check_real_path(scene)
    for label in ("linear_id_baseline", "linear_ones_baseline", "linear_timestep_baseline", "linear_bayesg_kl80_keep"):
        mac, _, _, _ = make_case(label, "5m_vs_6m")
        assert not supported(mac.agent), label
    known_feature_and_output()
