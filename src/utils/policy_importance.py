"""Read-only current-decision sensitivities for every SMAC policy.

Reconstruct a separate controller from the same config and weights. Replay real
test histories, holding incoming recurrent states fixed at each decision. These
are local derivatives, NOT credit allocation, causal utility or win-rate loss.
"""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch as th

PREFIX = "test_policy_importance/"


def valid_states(batch):
    filled = batch["filled"][..., 0].bool()
    terminal = batch["terminated"][..., 0].bool()
    mask = filled[:, :-1] & filled[:, 1:]
    if mask.shape[1] > 1:
        mask[:, 1:] &= (~terminal[:, :-2]).cumprod(1).bool()
    return mask.unsqueeze(-1) & (batch["avail_actions"][:, :-1].sum(-1) > 1)


def probe_indices(mask, budget, agent_offset=0):
    """Deterministic agent-stratified probes; never sample winning episodes."""
    selected = []
    agents = [i for i in range(mask.shape[-1]) if mask[..., i].any()]
    if agents:
        offset = agent_offset % len(agents)
        agents = agents[offset:] + agents[:offset]
    for position, agent in enumerate(agents):
        states = mask[..., agent].nonzero()
        remaining = max(0, budget - len(selected))
        slots = len(agents) - position
        count = min(len(states), (remaining + slots - 1) // slots)
        for index in th.linspace(0, len(states) - 1, count).long().tolist() if count else []:
            episode, timestep = states[index].tolist()
            selected.append((episode, timestep, agent))
    return selected


def analyze_batch(mac, batch, t_env, budget, agent_offset=0):
    """No original MAC/cache/gradient/batch/RNG mutation; no optimizer call."""
    mask = valid_states(batch)
    probes = probe_indices(mask, budget, agent_offset)
    if not probes:
        return None
    device = next(mac.agent.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with th.random.fork_rng(devices=cuda_devices), th.enable_grad():
            shadow = type(mac)(batch.scheme, {"agents": mac.n_agents}, deepcopy(mac.args))
            shadow.agent.to(device)
            shadow.load_state(mac)
            for target, source in zip(shadow.agent.modules(), mac.agent.modules()):
                target.training = source.training
            if hasattr(shadow, "set_dynamic_branch_gate_t_env"):
                shadow.set_dynamic_branch_gate_t_env(t_env)
            copied = deepcopy(batch)
            copied.to(device)
            obs = copied["obs"].detach().requires_grad_(True)
            state = copied["state"].detach().requires_grad_(True)
            copied.data.transition_data["obs"] = obs
            copied.data.transition_data["state"] = state
            params = [(name, p) for name, p in shadow.agent.named_parameters() if p.requires_grad]
            scores = {name: dict(elements=p.numel(), gradient_abs_sum=0.,
                                 parameter_gradient_abs_sum=0., connected_probes=0)
                      for name, p in params}
            rows = []
            shadow.init_hidden(copied.batch_size)
            last_t = max(t for _, t, _ in probes)
            for t in range(last_t + 1):
                selected = [(ep, i) for ep, step, i in probes if step == t]
                with th.set_grad_enabled(bool(selected)):
                    q = shadow.forward(copied, t=t, test_mode=True)
                if not th.isfinite(q).all():
                    raise ValueError("Non-finite policy Q in importance replay")
                for number, (episode, agent) in enumerate(selected):
                    available = copied["avail_actions"][episode, t, agent].bool()
                    ranked = q[episode, agent].detach().masked_fill(~available, -th.inf)
                    first = int(ranked.argmax())
                    ranked[first] = -th.inf
                    second = int(ranked.argmax())
                    gap = q[episode, agent, first] - q[episode, agent, second]
                    derivatives = th.autograd.grad(gap, [obs, state] + [p for _, p in params],
                                                  allow_unused=True,
                                                  retain_graph=number + 1 < len(selected))
                    local = derivatives[0]
                    global_input = derivatives[1]
                    observation_score = (local[episode, t, agent].abs() if local is not None
                                         else th.zeros_like(obs[episode, t, agent]))
                    state_score = (global_input[episode, t].abs() if global_input is not None
                                   else th.zeros_like(state[episode, t]))
                    for (name, parameter), gradient in zip(params, derivatives[2:]):
                        if gradient is not None:
                            if not th.isfinite(gradient).all():
                                raise ValueError("Non-finite parameter sensitivity: " + name)
                            scores[name]["gradient_abs_sum"] += gradient.detach().abs().sum().item()
                            scores[name]["parameter_gradient_abs_sum"] += (
                                parameter.detach() * gradient.detach()).abs().sum().item()
                            scores[name]["connected_probes"] += 1
                    if not th.isfinite(observation_score).all() or not th.isfinite(state_score).all():
                        raise ValueError("Non-finite input sensitivity")
                    rows.append(dict(episode=episode, timestep=t, agent=agent,
                                     top1=first, top2=second, gap=gap.detach().item(),
                                     obs_gradient=observation_score.detach().cpu().tolist(),
                                     state_gradient=state_score.detach().cpu().tolist()))
                # No backpropagation through previous decisions; caches belong
                # exclusively to the shadow controller, never the acting MAC.
                if shadow.hidden_states is not None:
                    shadow.hidden_states = shadow.hidden_states.detach()
            valid_obs = copied["obs"][:, :-1].detach()[mask.to(device)]
            return dict(parameters=scores, probes=rows,
                        obs_sum=valid_obs.double().sum(0).cpu().tolist(),
                        obs_square_sum=valid_obs.double().square().sum(0).cpu().tolist(),
                        observed_states=len(valid_obs))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def write_outputs(parts, names, n_agents, directory, title):
    directory.mkdir(parents=True, exist_ok=True)
    probes = [row for part in parts for row in part["probes"]]
    count = len(probes)
    scores = deepcopy(parts[0]["parameters"])
    for part in parts[1:]:
        for name, row in part["parameters"].items():
            for key in ("gradient_abs_sum", "parameter_gradient_abs_sum", "connected_probes"):
                scores[name][key] += row[key]
    for row in scores.values():
        row["mean_abs_gradient"] = row["gradient_abs_sum"] / (count * row["elements"])
        row["mean_abs_parameter_times_gradient"] = row["parameter_gradient_abs_sum"] / (count * row["elements"])
    observed = sum(part["observed_states"] for part in parts)
    mean = np.sum([p["obs_sum"] for p in parts], axis=0) / observed
    std = np.sqrt(np.maximum(0, np.sum([p["obs_square_sum"] for p in parts], axis=0) / observed - mean ** 2))
    fields = np.full((len(names), n_agents), np.nan)
    for agent in range(n_agents):
        rows = [row["obs_gradient"] for row in probes if row["agent"] == agent]
        if rows:
            fields[:, agent] = np.mean(rows, axis=0) * std
    result = dict(scope="current policy decision, incoming recurrent state/history fixed; NOT hyper-only",
                  parameters=scores, probes=probes, obs_names=names, obs_std=std.tolist(),
                  n_probes=count, n_observed_states=observed,
                  limitations=["Local continuous sensitivities, NOT causal importance or credit allocation.",
                               "Parameterization-dependent; zero derivative does not prove an input/parameter useless.",
                               "Only current own Obs derivatives shown; global-state derivatives saved separately.",
                               "No mixer/TD gradients; no gradients through the incoming recurrent history."])
    (directory / "importance.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (directory / "parameters.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "elements", "connected_probes", "mean_abs_gradient", "mean_abs_parameter_times_gradient"])
        for name, row in scores.items():
            writer.writerow([name] + [row[key] for key in ("elements", "connected_probes", "mean_abs_gradient", "mean_abs_parameter_times_gradient")])
    import matplotlib.pyplot as plt
    files = {}
    entries = sorted(scores, key=lambda name: scores[name]["mean_abs_parameter_times_gradient"], reverse=True)
    fig, axis = plt.subplots(figsize=(12, max(4, len(entries) * .25)), constrained_layout=True)
    try:
        axis.barh(entries, [scores[name]["mean_abs_parameter_times_gradient"] for name in entries])
        axis.invert_yaxis()
        axis.tick_params(axis="y", labelsize=6)
        axis.set_xlabel("mean |parameter x d(top1-top2 Q)/d(parameter)|; local sensitivity")
        axis.set_title(title + "\nLearned POLICY parameter blocks (not mixer/credit allocation)", fontsize=10)
        files["parameter_sensitivity"] = directory / "parameter_sensitivity.png"
        fig.savefig(files["parameter_sensitivity"], dpi=130)
    finally:
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, max(5, len(names) * .18)), constrained_layout=True)
    try:
        pixels = axis.imshow(np.ma.masked_invalid(fields), aspect="auto", cmap="magma", vmin=0)
        axis.set_yticks(range(len(names)), names, fontsize=6)
        axis.set_xticks(range(n_agents), ["agent " + str(i) for i in range(n_agents)], fontsize=7)
        axis.set_title(title + "\nCurrent own Obs sensitivity: whole policy, NOT hyper-only", fontsize=10)
        fig.colorbar(pixels, ax=axis, label="mean |d(Q gap)/d(obs)| x observed obs std")
        files["observation_sensitivity"] = directory / "observation_sensitivity.png"
        fig.savefig(files["observation_sensitivity"], dpi=130)
    finally:
        plt.close(fig)
    # State is NOT a policy input in standard decentralized VDN/QMIX. Only
    # produce this extra figure when the acting policy actually depends on it.
    state_gradients = np.asarray([row["state_gradient"] for row in probes])
    if np.any(state_gradients > 0):
        state_fields = np.full((state_gradients.shape[1], n_agents), np.nan)
        for agent in range(n_agents):
            rows = [row["state_gradient"] for row in probes if row["agent"] == agent]
            if rows:
                state_fields[:, agent] = np.mean(rows, axis=0)
        fig, axis = plt.subplots(figsize=(8, max(5, len(state_fields) * .14)), constrained_layout=True)
        try:
            pixels = axis.imshow(np.ma.masked_invalid(state_fields), aspect="auto", cmap="magma", vmin=0)
            axis.set_yticks(range(len(state_fields)), ["state_{}".format(i) for i in range(len(state_fields))], fontsize=6)
            axis.set_xticks(range(n_agents), ["agent " + str(i) for i in range(n_agents)], fontsize=7)
            axis.set_title(title + "\nCurrent global state sensitivity: POLICY path only, raw input indices", fontsize=10)
            fig.colorbar(pixels, ax=axis, label="mean |d(Q gap)/d(state)|; raw units, not Obs-std-scaled")
            files["state_sensitivity"] = directory / "state_sensitivity.png"
            fig.savefig(files["state_sensitivity"], dpi=130)
        finally:
            plt.close(fig)
    return files, result


class PolicyImportanceSession:
    def __init__(self, args, mac, logger):
        self.args, self.mac, self.logger = args, mac, logger
        self.enabled = args.env == "sc2" and bool(getattr(args, "test_policy_importance", True))
        self.interval = int(getattr(args, "test_policy_importance_interval", 1000000))
        self.episodes = int(getattr(args, "test_policy_importance_episodes", 10))
        self.budget = int(getattr(args, "test_policy_importance_samples", 64))
        if self.enabled and (self.interval <= 0 or self.episodes <= 0 or self.budget < mac.n_agents):
            raise ValueError("Invalid policy importance diagnostic settings")
        self.last_milestone = 0
        self.parts = None
        if args.env == "sc2":
            logger.log_stat(PREFIX + "enabled", int(self.enabled), 0)

    def begin(self, t_env):
        self.milestone = int(t_env) // self.interval if self.enabled else 0
        self.due = self.enabled and self.milestone > self.last_milestone
        self.parts = [] if self.due else None
        self.t_env, self.collected, self.used = t_env, 0, 0
        self.error, self.files = None, {}

    def consume(self, batch):
        if self.parts is None or self.error or self.collected >= self.episodes:
            return
        try:
            count = min(batch.batch_size, self.episodes - self.collected)
            budget = max(1, (self.budget - self.used) * count // (self.episodes - self.collected))
            part = analyze_batch(self.mac, batch[:count], self.t_env, budget, self.collected)
            if part:
                for row in part["probes"]:
                    row["episode"] += self.collected
                self.used += len(part["probes"])
                self.parts.append(part)
            self.collected += count
        except Exception as exc:
            self.error = str(exc)

    def finish(self):
        if self.parts is None:
            return
        self.last_milestone = self.milestone
        try:
            if self.error or not self.parts:
                raise RuntimeError(self.error or "No live states for importance probes")
            capturer = getattr(self.mac.agent, "rpg_relation_capturer", None)
            names = list(getattr(capturer, "semantic_names", ()))
            obs_dim = len(self.parts[0]["obs_sum"])
            if len(names) != obs_dim:
                names = ["obs_{}".format(i) for i in range(obs_dim)]
            name = getattr(self.args, "wandb_run_name", None) or self.args.name
            identity = sha256(name.encode()).hexdigest()[:8]
            directory = Path(self.args.local_results_path) / "policy_importance" / self.args.unique_token / (
                "seed_{}_{}".format(self.args.seed, identity)) / "step_{:09d}".format(self.t_env)
            title = "{} seed={} t_env={}".format(self.args.env_args.get("map_name"), self.args.seed, self.t_env)
            if name.startswith("SYNTHETIC"):
                title = "SYNTHETIC QA - NOT EXPERIMENTAL DATA\n" + title
            self.files, result = write_outputs(self.parts, names, self.mac.n_agents, directory, title)
            for key, path in self.files.items():
                if self.logger.use_wandb:
                    self.logger._update_wandb_buffer(PREFIX + key, self.logger.wandb_module.Image(str(path)), self.t_env)
            self.logger.log_stat(PREFIX + "samples", result["n_probes"], self.t_env)
            self.logger.log_stat(PREFIX + "episodes", self.collected, self.t_env)
            self.logger.log_stat(PREFIX + "failed", 0, self.t_env)
            self.logger.console_logger.info("Policy importance saved: %s", directory)
        except Exception as exc:
            self.error = str(exc)
            self.logger.log_stat(PREFIX + "failed", 1, self.t_env)
            self.logger.console_logger.error("Policy importance failed: %s", exc)
        finally:
            self.parts = None
