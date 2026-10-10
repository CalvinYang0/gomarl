#!/usr/bin/env python3
"""Production-model diagnostics on synthetic episodes; no simulator/uploads."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from smoke_test_counter_transformer_nine import make_case
from utils.policy_importance import analyze_batch, write_outputs, PolicyImportanceSession, PREFIX, probe_indices
from utils.visualization_requirements import preflight_visualizations, record_visualization_inventory
from utils.logging import Logger


def check_model(label, scene="5m_vs_6m", overrides=None):
    th.manual_seed(59)
    mac, learner, batch, logger = make_case(label, scene, production_shapes=True,
                                          config_overrides=overrides)
    batch["filled"][1, 3] = 1
    batch["avail_actions"][0, 1, 0] = 0
    batch["avail_actions"][0, 1, 0, 0] = 1
    mac.init_hidden(batch.batch_size)
    mac.set_dynamic_branch_gate_t_env(1000000)
    expected_q = []
    with th.no_grad():
        for timestep in range(batch.max_seq_length - 1):
            expected_q.append(mac.forward(batch, t=timestep, test_mode=True).clone())
    hidden = mac.hidden_states
    condition = getattr(mac.agent, "latest_condition", None)
    parameter = next(mac.agent.parameters())
    parameter.grad = th.ones_like(parameter)
    original_grad = parameter.grad.clone()
    weights = {key: value.clone() for key, value in mac.agent.state_dict().items()}
    observations = batch["obs"].clone()
    modes = [module.training for module in mac.agent.modules()]
    rng, py_rng, np_rng = th.get_rng_state(), random.getstate(), np.random.get_state()
    with th.no_grad():
        result = analyze_batch(mac, batch, 1000000, 16)
    assert result and 0 < len(result["probes"]) <= 16
    for row in result["probes"]:
        q = expected_q[row["timestep"]][row["episode"], row["agent"]]
        available = batch["avail_actions"][row["episode"], row["timestep"], row["agent"]].bool()
        ranked = q.masked_fill(~available, -th.inf)
        first = int(ranked.argmax())
        ranked[first] = -th.inf
        second = int(ranked.argmax())
        assert (row["top1"], row["top2"]) == (first, second)
        assert abs(row["gap"] - (q[first] - q[second]).item()) < 1e-6
    assert not any(row["episode"] == 0 and row["agent"] == 0 and row["timestep"] == 1 for row in result["probes"])
    assert not any(row["episode"] == 1 and row["timestep"] > 2 for row in result["probes"])
    assert mac.hidden_states is hidden and getattr(mac.agent, "latest_condition", None) is condition
    assert th.equal(parameter.grad, original_grad)
    assert all(th.equal(value, weights[key]) for key, value in mac.agent.state_dict().items())
    assert th.equal(observations, batch["obs"])
    assert modes == [module.training for module in mac.agent.modules()]
    assert th.equal(th.get_rng_state(), rng) and random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert any(row["connected_probes"] > 0 for row in result["parameters"].values())
    if label == "linear_global_state_baseline":
        assert any(sum(row["state_gradient"]) > 0 for row in result["probes"])
        with tempfile.TemporaryDirectory(prefix="gomarl-state-importance-") as temp:
            files, _ = write_outputs([result], ["obs_{}".format(i) for i in range(mac.args.obs_shape)],
                                     mac.n_agents, Path(temp), "SYNTHETIC QA - NOT EXPERIMENTAL DATA")
            assert len(files) == 3 and files["state_sensitivity"].is_file()
    if overrides:
        assert all(sum(row["state_gradient"]) == 0 for row in result["probes"])
    print("PASS:", scene, label, overrides or "", "acting/replay Q agreement; bounded probes; alive/terminal masks; original model/batch/grad/cache/RNG unchanged")
    return mac, batch, logger


def session_and_requirements(mac, batch, logger):
    with tempfile.TemporaryDirectory(prefix="gomarl-policy-importance-") as temp:
        args = mac.args
        args.test_policy_importance = True
        args.test_policy_importance_interval = 1000000
        args.test_policy_importance_episodes = 2
        args.test_policy_importance_samples = 16
        args.seed, args.name, args.unique_token = 1, "SYNTHETIC_ONLY_IMPORTANCE", "synthetic"
        args.wandb_run_name = args.name
        args.local_results_path = temp
        args.test_visualizations_required = True
        logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, 1000000, {}
        logger.wandb_module = SimpleNamespace(Image=lambda path: path)
        logger.wandb = SimpleNamespace(log=lambda *args, **kwargs: None)
        session = PolicyImportanceSession(args, mac, logger)
        session.begin(999999)
        session.consume(batch)
        assert session.parts is None
        session.begin(1000000)
        session.consume(batch)
        session.consume(batch)
        assert session.collected == 2
        session.finish()
        assert session.error is None and len(session.files) == 2
        for key, path in session.files.items():
            assert path.is_file() and Logger._wandb_metric_allowed(PREFIX + key)
        saved = json.loads((next(iter(session.files.values())).parent / "importance.json").read_text())
        assert saved["n_probes"] <= 16 and saved["scope"].endswith("NOT hyper-only")
        videos = SimpleNamespace(enabled=True, due=True, collected=10, rendered=10, limit=10)
        hyper = SimpleNamespace(enabled=False, due=False, error=None)
        record_visualization_inventory(args, logger, videos, session, hyper)
        inventory = next((Path(temp) / "test_visualization_inventory").rglob("*.json"))
        assert json.loads(inventory.read_text())["cloud_upload_verified"] is False
        videos.inventory = [{"error": "synthetic W&B media recording failure"}]
        try:
            record_visualization_inventory(args, logger, videos, session, hyper)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Rendered but failed media logging cannot count as complete")
        videos.inventory = []
        videos.rendered = 9
        try:
            record_visualization_inventory(args, logger, videos, session, hyper)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Required missing video must fail, not silently continue")
        session.begin(2000000)
        session.consume(None)
        session.finish()
        try:
            record_visualization_inventory(args, logger, videos, session, hyper)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed importance must block a required-output milestone")
        with patch("utils.visualization_requirements.check_video_dependencies", side_effect=ImportError("imageio")):
            try:
                preflight_visualizations(args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Missing encoder must fail BEFORE training")
        with patch("utils.visualization_requirements.check_video_dependencies"):
            preflight_visualizations(args)
        args.test_policy_importance = False
        try:
            preflight_visualizations(args)
        except ValueError:
            pass
        else:
            raise AssertionError("Mandatory feature cannot silently be disabled")
        args.test_visualizations_required = False
        preflight_visualizations(args)
        args.test_visualizations_required = True
        args.test_policy_importance = True
        args.test_nepisode, args.batch_size_run = 10, 8
        with patch("utils.visualization_requirements.check_video_dependencies"):
            try:
                preflight_visualizations(args)
            except ValueError:
                pass
            else:
                raise AssertionError("Batch-rounded normal test count cannot supply ten videos")
    print("PASS: schedule/episode/probe limits; two real PNGs + JSON/CSV; W&B keys; output manifest; missing dependencies/media fail; explicit debug opt-out")


def main():
    th.set_num_threads(1)
    for label in ("linear_baseline", "linear_id_baseline", "linear_ones_baseline", "linear_health_baseline",
                  "linear_timestep_baseline", "linear_global_state_baseline",
                  "linear_obs_entity_id_baseline", "linear_bayesg_kl80_keep", "baseline"):
        mac, batch, logger = check_model(label)
    for method in ("vdn", "qmix"):
        mac, batch, logger = check_model("baseline", overrides={"clean_model_type": "qmix_minimal", "mixer": method})
    check_model("linear_baseline", "MMM2")
    session_and_requirements(mac, batch, logger)
    masks = th.ones(1, 2, 10, dtype=th.bool)
    covered = {row[2] for offset in range(10) for row in probe_indices(masks, 3, offset)}
    assert covered == set(range(10))
    print("PASS: all current SMAC experiment families have common policy importance coverage")


if __name__ == "__main__":
    main()
