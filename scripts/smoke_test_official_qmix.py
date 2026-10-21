#!/usr/bin/env python3
"""Pinned-source integrity, both map shapes, targets, losses and Adam updates.

No SC2 process, W&B upload, Slurm submission, or legacy code mutation.
--verify-upstream additionally checks the immutable online source bytes.
"""
import argparse
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ROOT / "src/third_party/pymarl2_qmix"


def algorithm_hash(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

    def canonical(value):
        if isinstance(value, ast.AST):
            # Python 3.12 adds empty type_params to ClassDef/FunctionDef.
            return [type(value).__name__, [[k, canonical(v)] for k, v in ast.iter_fields(value)
                                          if k != "type_params"]]
        if isinstance(value, list):
            return [canonical(v) for v in value]
        return value

    return hashlib.sha256(json.dumps(canonical(tree), separators=(",", ":")).encode()).hexdigest()


def verify_sources(online=False):
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    assert provenance["commit"] == "8ccac7c5aa134422a2e3009be735d23cdf8ce2f8"
    for name, entry in provenance["files"].items():
        data = (PACKAGE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["adapted_sha256"], name
        if name.endswith(".py"):
            assert algorithm_hash(data.decode()) == entry["algorithm_ast_sha256"], name
        if online:
            from urllib.request import urlopen
            upstream = urlopen(entry["upstream_url"], timeout=30).read()
            assert hashlib.sha256(upstream).hexdigest() == entry["upstream_sha256"], name
            if name.endswith(".py"):
                assert algorithm_hash(upstream.decode()) == entry["algorithm_ast_sha256"], name
    print("PASS: pinned upstream checksums and unchanged algorithm bodies" + (" (online verified)" if online else ""))


def make_case(map_name):
    import torch as th
    import yaml
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers import REGISTRY as controllers
    from envs.starcraft import StarCraft2Env
    from learners import REGISTRY as learners
    from utils.logging import Logger
    config = {}
    for name in ("default.yaml", "envs/sc2.yaml", "algs/official_pymarl2_qmix.yaml"):
        config.update(yaml.safe_load((ROOT / "src/config" / name).read_text()))
    config["env_args"]["map_name"] = map_name
    env = StarCraft2Env(**config["env_args"])
    info = env.get_env_info()
    env.close()  # No reset or simulator launch.
    config.update(n_agents=info["n_agents"], n_actions=info["n_actions"],
                  obs_shape=info["obs_shape"], state_shape=info["state_shape"],
                  use_cuda=False, device="cpu", use_wandb=False, learner_log_interval=1)
    args = SimpleNamespace(**config)
    scheme = dict(
        obs=dict(vshape=args.obs_shape, group="agents"),
        state=dict(vshape=args.state_shape),
        actions=dict(vshape=(1,), group="agents", dtype=th.long),
        avail_actions=dict(vshape=(args.n_actions,), group="agents", dtype=th.int),
        reward=dict(vshape=(1,)), terminated=dict(vshape=(1,), dtype=th.uint8),
    )
    batch = EpisodeBatch(scheme, {"agents": args.n_agents}, 2, 5,
                         preprocess={"actions": ("actions_onehot", [OneHot(args.n_actions)])})
    batch.update(dict(obs=th.randn(2, 5, args.n_agents, args.obs_shape),
                      state=th.randn(2, 5, args.state_shape),
                      actions=th.randint(0, args.n_actions, (2, 5, args.n_agents, 1)),
                      avail_actions=th.ones(2, 5, args.n_agents, args.n_actions, dtype=th.int),
                      reward=th.rand(2, 5, 1), terminated=th.zeros(2, 5, 1, dtype=th.uint8)))
    batch["terminated"][1, 2] = 1
    batch["filled"][1, 3:] = 0
    logger = Logger(logging.getLogger("official-qmix-test"))
    mac = controllers[args.mac](batch.scheme, {"agents": args.n_agents}, args)
    learner = learners[args.learner](mac, batch.scheme, logger, args)
    return args, mac, learner, batch, logger


def check_case(map_name):
    import torch as th
    import torch.nn.functional as F
    from third_party.pymarl2_qmix.agent import NRNNAgent
    from third_party.pymarl2_qmix.learner import NQLearner
    from third_party.pymarl2_qmix.rl_utils import build_td_lambda_targets
    th.manual_seed(37)
    args, mac, learner, batch, logger = make_case(map_name)
    assert type(mac.agent) is NRNNAgent
    assert type(learner).train is NQLearner.train
    assert set(dict(mac.agent.named_children())) == {"fc1", "rnn", "fc2"}
    assert mac.agent.fc2.in_features == 64 and mac.agent.fc2.out_features == args.n_actions
    assert mac.agent.fc1.in_features == args.obs_shape + args.n_actions + args.n_agents
    assert len(learner.params) == len(list(mac.parameters())) + len(list(learner.mixer.parameters()))

    # Constructor through the production registry has exactly upstream RNG use.
    th.manual_seed(91)
    reference = NRNNAgent(mac.agent.fc1.in_features, args)
    reference_rng = th.get_rng_state().clone()
    th.manual_seed(91)
    registered = type(mac)(batch.scheme, {"agents": args.n_agents}, args)
    assert th.equal(reference_rng, th.get_rng_state())
    assert all(th.equal(v, registered.agent.state_dict()[k]) for k, v in reference.state_dict().items())

    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        inputs = mac._build_inputs(batch, 0)
        h0 = th.zeros(batch.batch_size * args.n_agents, 64)
        h = mac.agent.rnn(F.relu(mac.agent.fc1(inputs.reshape(-1, inputs.shape[-1]))), h0)
        expected = mac.agent.fc2(h).reshape(batch.batch_size, args.n_agents, args.n_actions)
        assert th.equal(mac.forward(batch, 0), expected)
        # State is a mixer input, never an input to the decentralized agent.
        changed = copy.deepcopy(batch)
        changed["state"][:] = th.randn_like(changed["state"]) * 100
        mac.init_hidden(batch.batch_size)
        assert th.equal(mac.forward(changed, 0), expected)

    # Independently compute the upstream masked half-MSE, including termination.
    ref_mac, ref_target, ref_mixer, ref_target_mixer = (
        copy.deepcopy(mac), copy.deepcopy(learner.target_mac),
        copy.deepcopy(learner.mixer), copy.deepcopy(learner.target_mixer))
    ref_mac.init_hidden(batch.batch_size)
    ref_target.init_hidden(batch.batch_size)
    out = th.stack([ref_mac.forward(batch, t) for t in range(5)], 1)
    with th.no_grad():
        target_out = th.stack([ref_target.forward(batch, t) for t in range(5)], 1)
        select = out.detach().masked_fill(batch["avail_actions"] == 0, -9999999).argmax(3, keepdim=True)
        target_q = ref_target_mixer(th.gather(target_out, 3, select).squeeze(3), batch["state"])
        terminated = batch["terminated"][:, :-1].float()
        valid = batch["filled"][:, :-1].float().clone()
        valid[:, 1:] *= 1 - terminated[:, :-1]
        targets = build_td_lambda_targets(batch["reward"][:, :-1], terminated, valid,
                                          target_q, args.n_agents, args.gamma, args.td_lambda)
    chosen = th.gather(out[:, :-1], 3, batch["actions"][:, :-1]).squeeze(3)
    joint = ref_mixer(chosen, batch["state"][:, :-1])
    expected_loss = (0.5 * (joint - targets).square() * valid).sum() / valid.sum()
    ref_params = list(ref_mac.parameters()) + list(ref_mixer.parameters())
    ref_opt = th.optim.Adam(ref_params, lr=args.lr)
    ref_opt.zero_grad()
    expected_loss.backward()
    th.nn.utils.clip_grad_norm_(ref_params, args.grad_norm_clip)
    ref_opt.step()
    learner.train(batch, t_env=10, episode_num=1)
    assert abs(logger.stats["loss_td"][-1][1] - expected_loss.item()) < 1e-6
    for actual, expected_parameter in zip(learner.params, ref_params):
        th.testing.assert_close(actual, expected_parameter, rtol=0, atol=0)
        assert actual.grad is not None and th.isfinite(actual.grad).all()
    assert all(th.equal(a, b) for a, b in zip(learner.target_mac.parameters(), ref_target.parameters()))
    learner.train(batch, t_env=20, episode_num=200)
    assert all(th.equal(a, b) for a, b in zip(mac.parameters(), learner.target_mac.parameters()))
    assert all(th.equal(a, b) for a, b in zip(learner.mixer.parameters(), learner.target_mixer.parameters()))
    print("PASS: {} single-linear head, exact initialization, half-MSE/Adam update and target copy".format(map_name))


def check_plans():
    import yaml
    from ozstar_submit_official_qmix_6jobs import build_plans, route_runtime
    from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
    plans = build_plans(ROOT)
    assert len(plans) == 6
    assert {(p["map_name"], p["seed"]) for p in plans} == {
        (m, s) for m in ("5m_vs_6m", "8m_vs_9m") for s in (1, 2, 3)}
    route_runtime(plans, Path("/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    validate_config_keys(ROOT, plans)
    for p in plans:
        assert p["exports"]["CONFIG"] == "official_pymarl2_qmix"
        assert "clean_" not in p["exports"]["EXTRA_ARGS"]
        assert p["exports"]["BATCH_SIZE_RUN"] == "8"
        assert p["exports"]["T_MAX"] == "10050000"
        assert "baseline_10m_s" not in p["job_name"]
    upstream = yaml.safe_load((PACKAGE / "upstream_qmix.yaml").read_text())
    actual = yaml.safe_load((ROOT / "src/config/algs/official_pymarl2_qmix.yaml").read_text())
    for key, value in upstream.items():
        if key not in {"mac", "agent", "learner", "name"}:
            assert actual[key] == value, key
    print("PASS: six isolated plans; official hyperparameters; no clean-model overrides")


def check_training_loop():
    """Exercise the actual training loop with synthetic episodes, never SC2."""
    import importlib
    import tempfile
    import torch as th
    from unittest.mock import patch
    loop = importlib.import_module("run.run")
    args, _, _, sample, logger = make_case("5m_vs_6m")
    args.t_max, args.test_interval = 520, 256
    args.seed, args.unique_token = 37, "synthetic-official-qmix-smoke"
    args.log_interval = 1000
    args.checkpoint_path = ""
    created = []

    class SyntheticRunner:
        def __init__(self, args, logger):
            self.args, self.logger = args, logger
            self.batch_size, self.t_env, self.closed = 8, 0, False
            self.episode_limit = 4
            created.append(self)

        def get_env_info(self):
            return dict(n_agents=args.n_agents, n_actions=args.n_actions,
                        obs_shape=args.obs_shape, state_shape=args.state_shape, episode_limit=4)

        def setup(self, scheme, groups, preprocess, mac):
            self.mac = mac

        def run(self, test_mode=False):
            batch = copy.deepcopy(sample[[0, 1, 0, 1, 0, 1, 0, 1]])
            self.mac.init_hidden(8)
            for t in range(4):
                chosen = self.mac.select_actions(batch, t, self.t_env, test_mode=test_mode)
                batch["actions"][:, t] = chosen.unsqueeze(-1)
            if not test_mode:
                self.t_env += 32
            else:
                self.logger.log_stat("test_battle_won_mean", 0.5, self.t_env)
            return batch

        def close_env(self):
            self.closed = True

    with tempfile.TemporaryDirectory(prefix="official-qmix-loop-") as scratch:
        args.local_results_path = scratch
        with patch.dict(loop.r_REGISTRY, {"parallel": SyntheticRunner}):
            loop.run_sequential(args, logger)
    assert created[0].closed and created[0].t_env == 544
    assert logger.stats["loss_td"] and logger.stats["test_battle_won_mean"]
    print("PASS: production run loop + replay + ordinary evaluation (synthetic episodes only)")


def report_environment():
    """Launch-time evidence only: inspect versions and source/map bytes."""
    import importlib.metadata
    import os
    import smac
    record = dict(python=sys.version, smac_module=smac.__file__, packages={})
    for package in ("torch", "numpy", "smac", "pysc2", "protobuf"):
        try:
            record["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][package] = "not available as package metadata"
    root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII"))
    record["sc2_root"] = str(root.resolve())
    record["sc2_installed_builds"] = sorted(p.name for p in (root / "Versions").glob("Base*"))
    record["maps"] = {}
    for name in ("5m_vs_6m", "8m_vs_9m"):
        record["maps"][name] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (root / "Maps").rglob(name + ".SC2Map")}
    from pysc2 import run_configs
    # run_configs resolves the executable/version but does not launch SC2.
    record["sc2_selected_version"] = repr(run_configs.get().version)
    record["project_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "src/config/algs/official_pymarl2_qmix.yaml", "src/learners/official_qmix_learner.py",
            "src/envs/starcraft/StarCraft2Env.py", "src/runners/parallel_runner.py", "src/run/run.py")}
    print("OFFICIAL_QMIX_ENVIRONMENT=" + json.dumps(record, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--verify-upstream", action="store_true")
    parser.add_argument("--report-environment", action="store_true")
    options = parser.parse_args()
    verify_sources(options.verify_upstream)
    if options.report_environment:
        report_environment()
    if options.sources_only:
        return
    import torch as th
    th.set_num_threads(1)
    check_plans()
    for name in ("5m_vs_6m", "8m_vs_9m"):
        check_case(name)
    check_training_loop()


if __name__ == "__main__":
    main()
