#!/usr/bin/env python3
"""Real four-model updates on synthetic episodes; never launch SC2/upload."""
from copy import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from smoke_test_counter_transformer_nine import make_case
from controllers.slot_group_controller import SlotGroupMAC
from learners.slot_group_learner import SlotGroupLearner
from modules.agents.slot_group_hyper_agent import VARIANTS
from utils.policy_importance import analyze_batch
from utils.slot_input_control import score_slots, log_slot_snapshot
from utils.value_diagnostics import collect_value_diagnostics, ValueDiagnosticSummary
from utils.logging import Logger


def check_real_startup(args, logger):
    """Exercise real runner/scheme/MAC/learner wiring; stub only environment.

    Stop after learner construction, before SC2 rollout or data uploads.
    Do not add the helper-only args.obs_shape that hid the original failure.
    """
    entry = importlib.import_module("run.run")
    startup_args = copy(args)
    startup_args.runner = "synthetic_slot_startup"
    startup_args.buffer_size = 2
    startup_args.use_wandb = False
    assert not hasattr(startup_args, "obs_shape")
    reached = []

    class StartupComplete(Exception):
        pass

    class MetadataRunner:
        def __init__(self, args, logger):
            pass

        def get_env_info(self):
            return dict(n_agents=5, n_actions=12, obs_shape=55,
                        state_shape=98, episode_limit=4)

        def setup(self, scheme, groups, preprocess, mac):
            assert scheme["obs"]["vshape"] == mac.agent.obs_dim == 55

    def construct(mac, scheme, logger, args):
        assert not hasattr(args, "obs_shape")
        learner = SlotGroupLearner(mac, scheme, logger, args)
        if mac.agent.is_grouped:
            assert learner.mixer.obs_dim == mac.agent.obs_dim == 55
            assert learner.mixer.small[0].state_dim == 5 * 55 + 55
        reached.append(True)
        raise StartupComplete()

    with patch.dict(entry.r_REGISTRY, synthetic_slot_startup=MetadataRunner), \
         patch.dict(entry.le_REGISTRY, slot_group_learner=construct):
        try:
            entry.run_sequential(startup_args, logger)
        except StartupComplete:
            pass
        else:
            raise AssertionError("Real startup must reach learner construction")
    assert reached == [True]


def check_model(model, directory):
    th.manual_seed(71)
    baseline, _, batch, logger = make_case("linear_baseline", "5m_vs_6m", production_shapes=True)
    args = copy(baseline.args)
    args.clean_model_type = model
    args.agent, args.mac, args.learner = "slot_group_hyper", "slot_group_mac", "slot_group_learner"
    args.local_results_path, args.unique_token, args.seed = directory, "synthetic", 1
    args.name = args.wandb_run_name = "SYNTHETIC_QA_" + model
    args.slot_group_interval, args.slot_importance_samples = 2, 32
    # Production runner assigns n_agents/n_actions/state_shape, not obs_shape.
    del args.obs_shape
    check_real_startup(args, logger)
    mac = SlotGroupMAC(batch.scheme, {"agents": args.n_agents}, args)
    learner = SlotGroupLearner(mac, batch.scheme, logger, args)
    assert mac.agent.obs_dim == 55 and args.n_agents == 5 and args.n_actions == 12
    # Same first-head architecture as matched baseline when all inputs open.
    shared = {k: v for k, v in baseline.agent.state_dict().items() if k in mac.agent.state_dict()}
    mac.agent.load_state_dict(shared, strict=False)
    learner._update_targets()
    mac.init_hidden(2)
    baseline.init_hidden(2)
    if mac.agent.slot_variant == "reverse_hierarchy":
        assert th.allclose(mac.forward(batch, 0, test_mode=True), baseline.forward(batch, 0, test_mode=True))
    if not mac.agent.is_grouped:
        gate = mac.agent.rpg_relation_capturer.dynamic_branch_gate
        assert th.allclose(th.sigmoid(gate.static_logits), th.full((55,), .1))
        a, p = gate(batch["obs"][:, 0], sample=False)
        b, p2 = gate(batch["obs"][:, 1] * 99, sample=False)
        assert th.equal(a, b) and th.equal(p, p2)
        assert th.equal(a, (p > .1).to(a))
        assert a.sum() == 0
        with th.no_grad():
            gate.static_logits[0] = th.logit(th.tensor(.11))
            gate.static_logits[1] = th.logit(th.tensor(.09))
        mask, _ = gate(batch["obs"][:, 0], sample=False)
        assert (mask[..., 0] == 1).all() and (mask[..., 1] == 0).all()
    else:
        assert mac.agent.important_slots.all() == ("reverse" in model)
        masks = mac.agent.group_masks(batch["obs"])
        assert th.equal(masks.sum(0), th.ones(55))
        # Fixed hidden state: changing the other group's Obs cannot affect
        # this head. Raw GRU still observes the original full input.
        obs, hidden = batch["obs"][:, 0], batch["obs"].new_ones(2, 5, 64)
        q1 = mac.agent.head_q(obs * masks[0], hidden)[0]
        q2 = mac.agent.head_q((obs + 3 * masks[1]) * masks[0], hidden)[0]
        assert th.equal(q1, q2)
        assert mac.agent.fc1.in_features == baseline.agent.fc1.in_features
    learner.train(batch, 10, 1)
    assert learner.train_steps == 1 and int(mac.agent.grouping_updates) == 0
    if not mac.agent.is_grouped:
        assert mac.agent.rpg_relation_capturer.dynamic_branch_gate.static_logits.grad.abs().sum() > 0
    else:
        # Exercise migration semantics deterministically, not an efficacy experiment.
        old = mac.agent.important_slots.clone()
        scores = th.arange(1, 56).float()
        change = mac.agent.apply_slot_scores(scores, th.zeros(55))
        if "reverse" in model:
            assert change["promoted"] is None and change["demoted"] == 0
            assert mac.agent.important_slots.sum() == 54
        else:
            assert change["promoted"] == 54 and change["demoted"] is None
            assert mac.agent.important_slots.sum() == 1
        assert not mac.agent.slot_score_initialized[old != mac.agent.important_slots].any()
        learner._sync_partition()
        assert th.equal(mac.agent.important_slots, learner.target_mac.agent.important_slots)
        # Independent group gradients; both losses reach shared recurrent trunk.
        _, groups, observations = learner._rollout(mac, batch)
        utilities = groups[:, :-1].gather(-1, batch["actions"][:, :-1].unsqueeze(2).expand(-1, -1, 2, -1, -1)).squeeze(-1)
        local = learner.mixer.group_values(utilities, observations[:, :-1], mac.agent.group_masks(utilities))
        trunk = mac.agent.rnn.weight_hh
        for g in (0, 1):
            grad = th.autograd.grad(local[..., g].square().mean(), trunk, retain_graph=True)[0]
            assert grad.abs().sum() > 0
        other_grad = th.autograd.grad(local[..., 0].sum(), next(mac.agent.other_generators.parameters()),
                                     retain_graph=True, allow_unused=True)[0]
        assert other_grad is None or other_grad.count_nonzero() == 0
        leaf = utilities.detach().requires_grad_(True)
        joint = learner.mixer(leaf, batch["state"][:, :-1], observations[:, :-1].detach(), mac.agent.group_masks(leaf))
        derivative = th.autograd.grad(joint.sum(), leaf)[0]
        assert (derivative >= 0).all()
    if mac.agent.is_grouped:
        with th.no_grad():
            online, group_q, _ = learner._rollout(mac, batch)
            expected_double = online.masked_fill(batch["avail_actions"] == 0, -th.inf).argmax(-1, keepdim=True)
            other_actions = group_q[:, :, 1].argmax(-1, keepdim=True)
            assert not th.equal(expected_double, other_actions)
        target_weights = {k: v.clone() for k, v in learner.target_mac.agent.named_parameters()}
    learner.train(batch, 20, 2)
    assert learner.train_steps == 2
    expected_losses = 2 if "dualtd" in model else 1
    assert len(learner.last_losses) == expected_losses and th.isfinite(learner.last_losses).all()
    if mac.agent.is_grouped:
        assert int(mac.agent.grouping_updates) == 2
        assert th.equal(learner.last_double_actions, expected_double)
        assert all(th.equal(v, target_weights[k]) for k, v in learner.target_mac.agent.named_parameters())
    assert not any("kl" in key or "aux" in key for key in logger.stats)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mac.agent.rnn.parameters())
    mac.init_hidden(2)
    rng = th.get_rng_state().clone()
    state = {k: v.clone() for k, v in mac.agent.state_dict().items()}
    measured = score_slots(mac, batch, 32)
    assert measured is not None and th.isfinite(measured[0]).all() and measured[2] <= 32
    assert th.equal(th.get_rng_state(), rng)
    assert all(th.equal(v, state[k]) for k, v in mac.agent.state_dict().items())
    assert score_slots(mac, batch, 32)[2] == 30  # 4+2 valid transitions, five live agents.
    hidden_before = mac.hidden_states
    summary = ValueDiagnosticSummary()
    collect_value_diagnostics(mac, learner.mixer, batch, args.gamma, summary)
    assert mac.hidden_states is hidden_before and summary.moments["q_tot"][0] == 6
    assert all(p.grad is not None for p in (mac.agent.rnn.weight_hh,))
    # Diagnostic replay through a shadow model must support the new architecture.
    result = analyze_batch(mac, batch, 1000000, 8)
    assert result and any(v["connected_probes"] for v in result["parameters"].values())
    assert all(sum(row["state_gradient"]) == 0 for row in result["probes"])
    mac.init_hidden(2)
    mac.reset_test_gate_probability_trajectory()
    with th.no_grad():
        for t in range(4):
            mac.select_actions(batch, t, 1000000, test_mode=True)
    trajectory = mac.pop_test_gate_probability_trajectory()
    assert trajectory is not None and trajectory["generated_parameter_vectors"].shape[0] == 4
    assert set(trajectory["branches"]) == ({"important", "unimportant"} if mac.agent.is_grouped else {"linear"})
    logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, 1000000, {}
    logger.wandb = SimpleNamespace(id='SYNTHETIC_HEAD_GROUP_QA', summary={})
    logger.wandb_module = SimpleNamespace(Image=lambda path: path, save=lambda *a, **k: None)
    path = log_slot_snapshot(mac.agent, args, logger, 1000000, 2)
    assert path.with_suffix(".png").is_file()
    snapshot = json.loads(path.read_text())
    assert len(snapshot["slot_names"]) == 55 and len(snapshot["masks_or_probability"]) == 2
    assert "slot_input/partition_probability_heatmap" in logger.wandb_current_data
    if mac.agent.is_grouped:
        assert logger.wandb_current_data['slot_input/grouping_membership']['nrows'] == 55
        assert logger.wandb.summary['slot_input/grouping_t_env'] == 1000000
        assert snapshot['source_run_id'] == 'SYNTHETIC_HEAD_GROUP_QA'
    assert Logger._wandb_metric_allowed("slot_input/important_count")
    # Training schedule and partition must survive a checkpoint.
    checkpoint = Path(directory) / model
    checkpoint.mkdir()
    learner.save_models(str(checkpoint))
    learner.train_steps = 0
    learner.load_models(str(checkpoint))
    assert learner.train_steps == 2
    assert th.equal(mac.agent.important_slots, learner.target_mac.agent.important_slots)
    print("PASS:", model, "real startup without args.obs_shape; optimizer/targets; no KL; grouping; diagnostics; heatmap; checkpoint", flush=True)


def main():
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="gomarl-slot-controls-") as directory:
        for model in VARIANTS:
            check_model(model, directory)
    print("4/4 synthetic code preflights passed; SC2 training and scientific efficacy NOT tested")


if __name__ == "__main__":
    main()
