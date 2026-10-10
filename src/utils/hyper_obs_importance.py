"""Hyper-path-only diagnostics for the ungated SMAC Linear-Obs baseline.

Not causal credit attribution: main GRU history and real action availability
are held fixed. No optimizer step, rollout or random sampling is performed.
The exact head Jacobian is observation-independent for this affine generator.
"""
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import torch as th
import torch.nn.functional as F


BLOCKS = ("hyper_bottleneck_w", "hyper_bottleneck_b", "hyper_out_w", "hyper_out_b")
MODEL = "smac_single_linear_suite_baseline_hypercond"
PREFIX = "test_hyper_obs_importance/"


def supported(agent):
    capturer = getattr(agent, "rpg_relation_capturer", None)
    flags = getattr(capturer, "counter_transformer_profile", {})
    return (
        getattr(agent, "model_type", None) == MODEL
        and {k: v for k, v in flags.items() if k not in {"label", "domain"}} == {"branch": "linear"}
        and capturer.relation_encoder_style == "linear_only"
        and capturer.dynamic_branch_gate is None
        and capturer.output_dim == capturer.relation_dim
        and isinstance(capturer.dual_linear_encoder, th.nn.Linear)
        and all(isinstance(getattr(agent, name, None), th.nn.Linear) for name in BLOCKS)
        and getattr(agent, "counter_transformer_policy_projection", None) is None
        and isinstance(agent.rnn, th.nn.GRUCell)
    )


def hyper_q(agent, obs, hidden):
    """Pure execution of the production two-layer generated head, no caches."""
    condition = agent.rpg_relation_capturer.dual_linear_encoder(obs)
    n, h = obs.shape[0], agent.hidden_dim
    w1 = agent.hyper_bottleneck_w(condition).reshape(n, h, h)
    b1 = agent.hyper_bottleneck_b(condition).reshape(n, 1, h)
    w2 = agent.hyper_out_w(condition).reshape(n, h, agent.n_actions)
    b2 = agent.hyper_out_b(condition).reshape(n, 1, agent.n_actions)
    middle = F.elu(th.bmm(hidden.reshape(n, 1, h), w1) + b1)
    return (th.bmm(middle, w2) + b2).squeeze(1)


def head_jacobian(agent):
    """Exact d[W1,b1,W2,b2]/d(obs), including the Obs encoder."""
    if not supported(agent):
        raise ValueError("Head importance requires the ungated SMAC Linear Obs baseline")
    encoder = agent.rpg_relation_capturer.dual_linear_encoder.weight.detach()
    return [getattr(agent, name).weight.detach() @ encoder for name in BLOCKS]


def field_group(name):
    if name.startswith("self_move_"):
        return "self_movement"
    match = re.match(r"(enemy|ally)_\d+_(.+)", name)
    side, field = match.groups() if match else ("self", name[5:] if name.startswith("self_") else name)
    if field in {"distance", "relative_x", "relative_y"}:
        field = "geometry"
    elif field.startswith("unit_type_"):
        field = "unit_type"
    elif field.startswith("last_action_"):
        field = "last_action"
    return side + "_" + field


def replay_samples(mac, batch, episodes):
    """Replay only the shared GRU. Never mutate MAC hidden state or caches."""
    count = min(episodes, batch.batch_size)
    filled = batch["filled"][:count, :, 0].bool()
    terminated = batch["terminated"][:count, :, 0].bool()
    pieces = []
    h = batch["obs"].new_zeros(count, mac.n_agents, mac.agent.hidden_dim)
    active = th.ones(count, dtype=th.bool, device=h.device)
    with th.no_grad():
        for t in range(batch.max_seq_length - 1):
            inputs = mac._build_inputs(batch, t)[:count]
            h = mac.agent.rnn(F.relu(mac.agent.fc1(inputs.reshape(count * mac.n_agents, -1))),
                              h.reshape(count * mac.n_agents, -1)).reshape_as(h)
            avail = batch["avail_actions"][:count, t].bool()
            valid = (active & filled[:, t] & filled[:, t + 1]).unsqueeze(1)
            valid = valid.expand(-1, mac.n_agents) & (avail.sum(-1) > 1)
            if valid.any():
                episode_ids, agent_ids = valid.nonzero(as_tuple=True)
                pieces.append({
                    "obs": batch["obs"][:count, t][valid].detach().cpu(),
                    "hidden": h[valid].detach().cpu(),
                    "avail": avail[valid].detach().cpu(),
                    "agent": agent_ids.cpu(), "episode": episode_ids.cpu(),
                    "timestep": th.full_like(agent_ids.cpu(), t),
                })
            active &= ~terminated[:, t]
    return {key: th.cat([piece[key] for piece in pieces]) for key in pieces[0]} if pieces else None


def analyze(agent, samples, max_samples=256):
    if not supported(agent):
        raise ValueError("Unsupported model for hyper-only Obs importance")
    if max_samples < agent.n_agents:
        raise ValueError("Probe budget must be at least the number of agents")
    if not all(th.isfinite(samples[key]).all() for key in ("obs", "hidden")):
        raise ValueError("Non-finite diagnostic observations or hidden states")
    names = list(agent.rpg_relation_capturer.semantic_names)
    groups = list(dict.fromkeys(field_group(name) for name in names))
    # Stratify by agent and use deterministic evenly spaced real states.
    indices = []
    cap = max(1, max_samples // agent.n_agents)
    for i in range(agent.n_agents):
        candidates = (samples["agent"] == i).nonzero().flatten()
        if candidates.numel():
            positions = th.linspace(0, candidates.numel() - 1, min(cap, candidates.numel())).long()
            indices.extend(candidates[positions].tolist())
    if not indices:
        raise ValueError("No live agent states with at least two available actions")
    device = next(agent.parameters()).device
    obs = samples["obs"][indices].to(device).detach().requires_grad_(True)
    hidden = samples["hidden"][indices].to(device).detach()
    avail = samples["avail"][indices].to(device)
    ids = samples["agent"][indices]
    # Scale using the whole recorded test set, not only the bounded probe set.
    std = samples["obs"].std(0, unbiased=False).to(device)
    reference = samples["obs"].mean(0).to(device)
    with th.enable_grad():
        q = hyper_q(agent, obs, hidden)
        if not th.isfinite(q).all():
            raise ValueError("Non-finite generated-head Q values")
        ranked = q.detach().masked_fill(~avail, -th.inf)
        first = ranked.argmax(-1)  # Same first-index tie breaking as greedy execution.
        ranked.scatter_(1, first.unsqueeze(1), -th.inf)
        second = ranked.argmax(-1)
        pair = th.stack([first, second], dim=1)
        margin = q.gather(1, pair[:, :1]) - q.gather(1, pair[:, 1:])
        gradient = th.autograd.grad(margin.sum(), obs)[0].detach()
    scaled = gradient.abs() * std
    per_agent = []
    for i in range(agent.n_agents):
        rows = scaled[ids.to(device) == i]
        per_agent.append(rows.mean(0).cpu().numpy() if rows.numel() else np.full(len(names), np.nan))
    jacobians = head_jacobian(agent)
    if not th.isfinite(gradient).all() or not all(th.isfinite(j).all() for j in jacobians):
        raise ValueError("Non-finite head or decision sensitivity")
    block_rms = th.stack([j.square().mean(0).sqrt() for j in jacobians])
    raw_l2 = th.cat(jacobians).square().sum(0).sqrt()
    ablations = []
    with th.no_grad():
        for group in groups:
            columns = [j for j, name in enumerate(names) if field_group(name) == group]
            changed = obs.detach().clone()
            changed[:, columns] = reference[columns]
            altered_q = hyper_q(agent, changed, hidden)
            altered_action = altered_q.masked_fill(~avail, -th.inf).argmax(-1)
            altered_margin = (altered_q.gather(1, pair[:, :1]) - altered_q.gather(1, pair[:, 1:])).squeeze(1)
            ablations.append({
                "group": group, "features": len(columns),
                "action_flip_rate": (altered_action != pair[:, 0]).float().mean().item(),
                "original_pair_margin_abs_change": (altered_margin - margin.detach().squeeze(1)).abs().mean().item(),
            })
    return {
        "scope": "hypernetwork-only; GRU history/hidden and real avail_actions fixed",
        "model": agent.model_type, "slot_names": names, "groups": groups,
        "n_observed_states": len(samples["obs"]), "n_probe_states": len(indices),
        "agent_probe_counts": [(ids == i).sum().item() for i in range(agent.n_agents)],
        "obs_mean": reference.cpu().tolist(), "obs_std": std.cpu().tolist(),
        "head_jacobian_l2": raw_l2.cpu().tolist(),
        "head_block_rms": block_rms.cpu().tolist(),
        "head_block_rms_std_scaled": (block_rms * std).cpu().tolist(),
        "gap_gradient_abs_mean": gradient.abs().mean(0).cpu().tolist(),
        "gap_gradient_std_scaled_mean": scaled.mean(0).cpu().tolist(),
        "agent_gap_gradient_std_scaled_mean": np.stack(per_agent).tolist(),
        "group_mean_replacement": ablations,
        "probe_episode": samples["episode"][indices].tolist(),
        "probe_timestep": samples["timestep"][indices].tolist(),
        "probe_agent": ids.tolist(),
        "probe_gap_gradient_std_scaled": scaled.cpu().tolist(),
        "limitations": [
            "Affine head Jacobian is constant across observations at a fixed checkpoint.",
            "Parameter sensitivity depends on parameterization; not a cross-model importance score.",
            "Gradients are local continuous sensitivities, including binary fields.",
            "Mean replacement may be off-manifold; an action flip is not a win-rate estimate.",
            "Entity slot indices are observer-local; ally_0 is not always absolute agent 0.",
            "Only first requested ordinary test episodes; bounded probes are not all evaluation states.",
        ],
    }


def write_outputs(result, directory, title):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # No NaN JSON: agents without probes remain explicitly null.
    def clean(value):
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return None if isinstance(value, float) and not np.isfinite(value) else value
    (directory / "importance.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    with (directory / "features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "group", "obs_std", "head_jacobian_l2", "gap_gradient_abs_mean", "gap_gradient_std_scaled_mean"])
        for i, name in enumerate(result["slot_names"]):
            writer.writerow([name, field_group(name)] + [result[key][i] for key in (
                "obs_std", "head_jacobian_l2", "gap_gradient_abs_mean", "gap_gradient_std_scaled_mean")])
    import matplotlib.pyplot as plt
    files = {}
    for key, data, columns, label in (
        ("head_sensitivity", np.asarray(result["head_block_rms_std_scaled"]).T,
         ["W1", "b1", "W2", "b2"], "RMS parameter sensitivity x empirical obs std"),
        ("decision_sensitivity", np.asarray(result["agent_gap_gradient_std_scaled_mean"]).T,
         ["agent " + str(i) for i in range(len(result["agent_probe_counts"]))],
         "mean |d(top1-top2 Q)/d(hyper obs)| x empirical obs std"),
    ):
        fig, axis = plt.subplots(figsize=(max(7, len(columns) * 1.0), max(6, len(result["slot_names"]) * .19)), constrained_layout=True)
        try:
            pixels = axis.imshow(data, aspect="auto", interpolation="nearest", cmap="magma", vmin=0)
            axis.set_yticks(range(len(result["slot_names"])), result["slot_names"], fontsize=6)
            axis.set_xticks(range(len(columns)), columns)
            axis.set_title(title + "\n" + key.replace("_", " ") + " (hyper path only)", fontsize=10)
            fig.colorbar(pixels, ax=axis, label=label)
            path = directory / (key + ".png")
            fig.savefig(path, dpi=140)
            files[key] = path
        finally:
            plt.close(fig)
    rows = result["group_mean_replacement"]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(rows) * .3)), constrained_layout=True)
    try:
        labels = [r["group"] + " (" + str(r["features"]) + ")" for r in rows]
        axes[0].barh(labels, [100 * r["action_flip_rate"] for r in rows])
        axes[0].set_xlabel("greedy action flips (%)")
        axes[1].barh(labels, [r["original_pair_margin_abs_change"] for r in rows])
        axes[1].set_xlabel("mean absolute change of original top1-top2 gap")
        for axis in axes:
            axis.invert_yaxis()
        fig.suptitle(title + "\nGroup replaced by test mean; hidden/action mask fixed; NOT win-rate loss", fontsize=10)
        path = directory / "group_ablation.png"
        fig.savefig(path, dpi=140)
        files["group_ablation"] = path
    finally:
        plt.close(fig)
    return files


class HyperObsImportanceSession:
    def __init__(self, args, mac, logger):
        self.args, self.mac, self.logger = args, mac, logger
        self.enabled = bool(getattr(args, "test_hyper_obs_importance", False)) and supported(mac.agent)
        self.interval = int(getattr(args, "test_hyper_obs_importance_interval", 1000000))
        self.episodes = int(getattr(args, "test_hyper_obs_importance_episodes", 10))
        self.max_samples = int(getattr(args, "test_hyper_obs_importance_samples", 256))
        if self.enabled and (self.interval <= 0 or self.episodes <= 0 or self.max_samples < mac.n_agents):
            raise ValueError("Invalid hyper Obs importance diagnostic settings")
        self.last = 0
        self.parts = None
        if args.env == "sc2":
            logger.log_stat(PREFIX + "enabled", int(self.enabled), 0)
        if getattr(args, "test_hyper_obs_importance", False):
            logger.console_logger.info(
                "Hyper Obs importance: %s (supported path: ungated SMAC Linear Obs only)",
                "enabled" if self.enabled else "skipped for this model",
            )

    def begin(self, t_env):
        self.due = self.enabled and t_env - self.last >= self.interval
        self.parts = [] if self.due else None
        self.t_env, self.collected, self.error = t_env, 0, None

    def consume(self, batch):
        if self.parts is None or self.error or self.collected >= self.episodes:
            return
        try:
            count = min(batch.batch_size, self.episodes - self.collected)
            part = replay_samples(self.mac, batch, count)
            if part is not None:
                part["episode"] += self.collected
                self.parts.append(part)
            self.collected += count
        except Exception as exc:
            self.error = str(exc)

    def finish(self):
        if self.parts is None:
            return
        self.last = self.t_env
        try:
            if self.error:
                raise RuntimeError(self.error)
            if not self.parts:
                raise ValueError("No valid test states for hyper Obs importance")
            samples = {key: th.cat([p[key] for p in self.parts]) for key in self.parts[0]}
            result = analyze(self.mac.agent, samples, self.max_samples)
            result.update(t_env=self.t_env, episodes=self.collected, seed=self.args.seed,
                          map_name=self.args.env_args.get("map_name", "SMAC"),
                          run_name=getattr(self.args, "wandb_run_name", None) or self.args.name)
            identity = sha256(result["run_name"].encode()).hexdigest()[:8]
            directory = Path(self.args.local_results_path) / "hyper_obs_importance" / self.args.unique_token / (
                "seed_{}_{}".format(self.args.seed, identity)) / "step_{:09d}".format(self.t_env)
            title = "{} seed={} t_env={}".format(result["map_name"], self.args.seed, self.t_env)
            if result["run_name"].startswith("SYNTHETIC"):
                title = "SYNTHETIC QA - NOT EXPERIMENTAL DATA\n" + title
            files = write_outputs(result, directory, title)
            for key, path in files.items():
                if self.logger.use_wandb:
                    self.logger._update_wandb_buffer(PREFIX + key, self.logger.wandb_module.Image(str(path)), self.t_env)
            self.logger.log_stat(PREFIX + "samples", result["n_probe_states"], self.t_env)
            self.logger.log_stat(PREFIX + "episodes", self.collected, self.t_env)
            self.logger.log_stat(PREFIX + "failed", 0, self.t_env)
            self.logger.console_logger.info("Hyper Obs importance saved: %s", directory)
        except Exception as exc:
            self.error = str(exc)
            self.logger.log_stat(PREFIX + "failed", 1, self.t_env)
            self.logger.console_logger.warning("Hyper Obs importance diagnostic failed: %s", exc)
        finally:
            self.parts = None
