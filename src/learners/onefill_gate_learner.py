"""Opt-in stochastic state gates and two-branch mixer weight ablations."""
import copy
import json
import math
from pathlib import Path

import torch as th

from .mixer_input_learner import MixerInputLearner
from modules.mixers.onefill_gate import OneFillGateMixer, bernoulli_gate_kl
from utils.mixer_state_importance import complete_episode_returns, state_slot_names
from utils.rl_utils import build_td_lambda_targets


class OneFillGateLearner(MixerInputLearner):
    mixer_class = OneFillGateMixer
    # Reuse only optimizer construction, MAC rollout, device and target-copy
    # helpers. The old learner's probing, losses and checkpoint API are bypassed.
    def __init__(self, mac, scheme, logger, args):
        del scheme
        if mac.agent.model_type != "qmix_minimal" or mac.agent.hyper_out_w is not None or not args.double_q:
            raise ValueError("One-fill experiments require ordinary qmix_minimal and Double-Q")
        if (args.mixer_gate_aux_score != "none" or args.mixer_gate_l1_coef or args.mixer_gate_variance_coef or
                args.mixer_input_detach_unimportant):
            raise ValueError("Do not mix one-fill experiments with legacy objectives/routing")
        self.mac, self.logger, self.args = mac, logger, args
        self.objective = str(args.mixer_onefill_objective)
        if self.objective not in {"td", "kl", "mc"}:
            raise ValueError("Unknown one-fill objective")
        if args.mixer_onefill_residual and self.objective != "td":
            raise ValueError("Residual ablation must not silently add KL/MC")
        self.mixer = self.mixer_class(args)
        self.target_mac, self.target_mixer = copy.deepcopy(mac), copy.deepcopy(self.mixer)
        self.names = state_slot_names(args, self.mixer.state_dim)
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.gate_params = [self.mixer.gate_logits] if self.objective == "mc" else []
        self.main_params = [p for p in self.params if not self.gate_params or p is not self.mixer.gate_logits]
        self.optimiser = self._make_optimiser(self.main_params)
        self.gate_optimiser = self._make_optimiser(self.gate_params) if self.gate_params else None
        self.kl_prior = float(args.mixer_onefill_kl_prior)
        self.kl_ratio = float(args.clean_adaptive_auxiliary_target_ratio)
        self.kl_decay = float(args.clean_adaptive_auxiliary_ema_decay)
        self.kl_eps = float(args.clean_adaptive_auxiliary_eps)
        self.kl_max = float(args.clean_adaptive_auxiliary_max_coef)
        self.mc_coef = float(args.mixer_onefill_mc_coef)
        if (not 0 < self.kl_prior < 1 or not 0 < self.kl_decay < 1 or
                any(not math.isfinite(x) or x <= 0 for x in (self.kl_ratio, self.kl_eps, self.kl_max, self.mc_coef))):
            raise ValueError("Invalid one-fill objective settings")
        self.image_interval = int(args.slot_heatmap_interval)
        if self.image_interval <= 0:
            raise ValueError("Invalid one-fill diagnostic interval")
        self.train_steps = self.last_target_update_episode = 0
        self.last_image_t, self.log_stats_t = -self.image_interval, -args.learner_log_interval - 1
        self.ema_td = self.ema_kl = None
        self.last_gate_aux_loss = self.last_double_actions = self.last_losses = None
        self.contract = dict(version=1, fill_value=self.mixer.fill_value, sampling="hard_binary_concrete_st",
            target="soft_probability", evaluation="probability_diagnostic_only", objective=self.objective,
            residual=self.mixer.residual, residual_scale=self.mixer.residual_scale,
            initial_probability=float(args.mixer_input_initial_probability), temperature=self.mixer.temperature,
            kl_prior=self.kl_prior, kl_ratio=self.kl_ratio, kl_decay=self.kl_decay, kl_eps=self.kl_eps, kl_max=self.kl_max,
            mc_coef=self.mc_coef, image_interval=self.image_interval, state_dim=self.mixer.state_dim,
            n_agents=args.n_agents, mixing_embed_dim=args.mixing_embed_dim,
            hypernet_layers=args.hypernet_layers, hypernet_embed=args.hypernet_embed,
            gamma=args.gamma, td_lambda=args.td_lambda, target_update_interval=args.target_update_interval,
            optimizer=args.optimizer, lr=args.lr, optim_alpha=args.optim_alpha, optim_eps=args.optim_eps,
            grad_norm_clip=args.grad_norm_clip)
        # Historical default contracts/state_dicts remain loadable. New
        # architectures cannot silently load a centered-residual checkpoint.
        if self.mixer.combination != "centered" or self.mixer.secondary_ones:
            self.contract.update(version=2, combination=self.mixer.combination,
                secondary_ones=self.mixer.secondary_ones,
                alpha_initial=float(args.mixer_residual_alpha_initial))

    def _kl_coefficient(self, td, kl):
        # Same loss-value EMA scaling as direct KL80, not gradient norm matching.
        td, kl = float(td.detach()), float(kl.detach())
        if kl <= self.kl_eps:
            return 0.
        if self.ema_td is None:
            self.ema_td, self.ema_kl = td, kl
        else:
            self.ema_td = self.kl_decay * self.ema_td + (1 - self.kl_decay) * td
            self.ema_kl = self.kl_decay * self.ema_kl + (1 - self.kl_decay) * kl
        return min(self.kl_max, self.kl_ratio * self.ema_td / max(self.ema_kl, self.kl_eps))

    def _mc_gate_loss(self, chosen, states, rewards, terminated, valid):
        returns, complete = complete_episode_returns(rewards, terminated, valid, self.args.gamma)
        indices = complete.reshape(-1).bool()
        if not indices.any():
            return None, 0
        q = chosen.detach().reshape(-1, self.args.n_agents)[indices].unsqueeze(0)
        s = states.detach().reshape(-1, self.mixer.state_dim)[indices].unsqueeze(0)
        g = returns.detach().reshape(-1, 1)[indices].unsqueeze(0)
        # Freeze theta, NOT the forward graph: dQ/dinput still reaches gate.
        frozen = [p for p in self.mixer.parameters() if p is not self.mixer.gate_logits]
        flags = [p.requires_grad for p in frozen]
        try:
            for p in frozen:
                p.requires_grad_(False)
            prediction = self.mixer(q, s, mode="train")
        finally:
            for p, flag in zip(frozen, flags):
                p.requires_grad_(flag)
        return self.mc_coef * (prediction - g).square().mean(), int(indices.sum())

    def train(self, batch, t_env, episode_num):
        self.mac.agent.train()
        rewards, terminated = batch["reward"][:, :-1], batch["terminated"][:, :-1].float()
        valid = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
        valid[:, 1:] *= (1 - terminated[:, :-1]).cumprod(1)
        online = self._rollout(self.mac, batch)
        chosen = online[:, :-1].gather(-1, batch["actions"][:, :-1].long()).squeeze(-1)
        states = batch["state"][:, :-1]
        joint = self.mixer(chosen, states, mode="train", detach_gate=self.objective == "mc")
        with th.no_grad():
            target_q = self._rollout(self.target_mac, batch)
            selected = online.detach().masked_fill(batch["avail_actions"] == 0, -th.inf).argmax(-1, keepdim=True)
            self.last_double_actions = selected.clone()
            evaluated = target_q.gather(-1, selected).squeeze(-1)
            target_joint = self.target_mixer(evaluated, batch["state"], mode="target")
            targets = build_td_lambda_targets(rewards, terminated * valid, valid, target_joint,
                self.args.n_agents, self.args.gamma, self.args.td_lambda)
        error = joint - targets
        selected_error = error[valid.bool()]
        # A fully empty batch must not advance optimizer momentum/schedules.
        if not selected_error.numel():
            return
        td = selected_error.square().mean()
        kl, coefficient = td.new_zeros(()), 0.
        if self.objective == "kl":
            kl = bernoulli_gate_kl(self.mixer.gate_logits, self.kl_prior)
            coefficient = self._kl_coefficient(td, kl)
        loss = td + coefficient * kl
        self.optimiser.zero_grad(set_to_none=True)
        gate_grads, mc_count = None, 0
        self.last_gate_aux_loss = None
        if self.gate_optimiser is not None:
            self.gate_optimiser.zero_grad(set_to_none=True)
            mc_loss, mc_count = self._mc_gate_loss(chosen, states, rewards, terminated, valid)
            if mc_loss is not None:
                if not th.isfinite(mc_loss):
                    raise RuntimeError("Non-finite direct gate MC loss")
                gate_grads = th.autograd.grad(mc_loss, self.gate_params)
                self.last_gate_aux_loss = mc_loss.detach()
        if not th.isfinite(loss):
            raise RuntimeError("Non-finite one-fill TD/KL loss")
        loss.backward()
        norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        if not th.isfinite(norm):
            raise RuntimeError("Non-finite one-fill main gradient")
        gate_norm = td.new_zeros(())
        if gate_grads is not None:
            for p, gradient in zip(self.gate_params, gate_grads):
                p.grad = gradient
            gate_norm = th.nn.utils.clip_grad_norm_(self.gate_params, self.args.grad_norm_clip)
            if not th.isfinite(gate_norm):
                raise RuntimeError("Non-finite direct gate MC gradient")
        self.optimiser.step()
        if gate_grads is not None:
            self.gate_optimiser.step()
        self.train_steps += 1
        self.last_losses = td.detach().reshape(1)
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            metrics = dict(loss_td=td.item(), grad_norm=norm.item(),
                td_error_abs=selected_error.abs().mean().item())
            p = self.mixer.gate_logits.sigmoid().detach()
            metrics.update({"mixer_input/loss_total": loss.item(), "mixer_input/train_updates": self.train_steps})
            if not self.mixer.secondary_ones:
                metrics.update({"mixer_input/gate_probability_mean": p.mean().item(),
                    "mixer_input/expected_open_slots": p.sum().item()})
            if self.mixer.alpha_logit is not None:
                metrics["mixer_branches/alpha_main"] = self.mixer.alpha().detach().item()
                metrics["mixer_branches/alpha_secondary"] = 1 - metrics["mixer_branches/alpha_main"]
            if self.objective == "kl":
                metrics.update({"mixer_input/kl": kl.item(), "mixer_input/kl_coefficient": coefficient,
                    "mixer_input/kl_weighted": coefficient * kl.item(),
                    "mixer_input/kl_weighted_to_td": coefficient * kl.item() / max(td.item(), self.kl_eps)})
            if self.objective == "mc":
                metrics.update({"mixer_input/gate_mc_complete_count": mc_count,
                    "mixer_input/gate_mc_available": int(gate_grads is not None),
                    "mixer_input/gate_mc_grad_norm": gate_norm.item()})
                if self.last_gate_aux_loss is not None:
                    metrics["mixer_input/gate_mc_loss"] = self.last_gate_aux_loss.item()
            for name, value in metrics.items():
                self.logger.log_stat(name, value, t_env)
            self.log_stats_t = t_env
        if t_env - self.last_image_t >= self.image_interval:
            self._snapshot(t_env, states.detach(), valid.detach())
            self.last_image_t = t_env

    def _snapshot(self, t_env, states=None, valid=None):
        import matplotlib.pyplot as plt
        p = self.mixer.gate_logits.sigmoid().detach().cpu()
        directory = Path(self.args.local_results_path) / "mixer_onefill_gate" / self.args.unique_token
        directory.mkdir(parents=True, exist_ok=True)
        stem = "step_{:09d}_update_{:07d}".format(int(t_env), self.train_steps)
        path, image_path = directory / (stem + ".json"), directory / (stem + ".png")
        record = dict(t_env=int(t_env), train_updates=self.train_steps,
            run_name=self.args.wandb_run_name, state_slot_names=self.names,
            keep_probability=None if self.mixer.secondary_ones else p.tolist(),
            contract=self.contract, combination=self.mixer.combination,
            alpha_main=None if self.mixer.alpha_logit is None else self.mixer.alpha().detach().item(),
            note="No operative gate: both branches all ones" if self.mixer.secondary_ones else
                 "Probability only; no field importance, hard threshold or grouping")
        if self.mixer.residual and states is not None:
            record["weights"] = self._weight_snapshot(t_env, directory, stem, states, valid)
        path.write_text(json.dumps(record, indent=2, allow_nan=False))
        if self.mixer.secondary_ones:
            if self.logger.use_wandb:
                self.logger.wandb_module.save(str(path), base_path=str(directory))
            return
        fig, ax = plt.subplots(figsize=(max(14, len(self.names) * .15), 3), constrained_layout=True)
        try:
            pixel = ax.imshow(p.unsqueeze(0).numpy(), aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax.set_yticks([0], labels=["keep probability"])
            ax.set_xticks(range(len(self.names)), labels=self.names, rotation=90, fontsize=6)
            title = "{} / {} / seed={} / step={} / closed fills {:g}; no threshold".format(
                self.args.env_args.get("map_name"), self.objective, self.args.seed, t_env, self.mixer.fill_value)
            if self.mixer.residual:
                title += " / all-ones base / " + self.mixer.combination
            if self.args.wandb_run_name.startswith("SYNTHETIC"):
                title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
            ax.set_title(title)
            fig.colorbar(pixel, ax=ax)
            fig.savefig(image_path, dpi=130)
        finally:
            plt.close(fig)
        if self.logger.use_wandb:
            self.logger._update_wandb_buffer("mixer_input/onefill_probability_heatmap",
                self.logger.wandb_module.Image(str(image_path)), t_env)
            self.logger.wandb_module.save(str(path), base_path=str(directory))

    def _weight_snapshot(self, t_env, directory, stem, states, valid):
        """Small, labeled parameter diagnostics; no perturbation or update."""
        import matplotlib.pyplot as plt
        with th.no_grad():
            selected = states[valid.squeeze(-1).bool()] if valid is not None else states.reshape(-1, self.mixer.state_dim)
            selected = selected[:256].unsqueeze(0)
            if not selected.shape[1]:
                return dict(sample_count=0)
            parts = self.mixer.diagnostic_components(selected)
        columns = ("raw_main", "raw_secondary", "main_contribution", "secondary_contribution", "combined", "effective")
        names = ("Main raw", "Secondary raw", "Main contribution", "Secondary contribution", "Combined pre-abs", "Effective W >= 0")
        fig, axes = plt.subplots(2, len(columns), figsize=(21, 7), constrained_layout=True)
        record = dict(sample_count=selected.shape[1], input_mode="soft_probability_diagnostic",
            selection="first up to 256 valid replay transitions; not importance or training-mask expectation",
            alpha_main=None if self.mixer.alpha_logit is None else self.mixer.alpha().detach().item(), layers={})
        try:
            for row, layer in enumerate(("w1", "w_final")):
                shape = (self.args.n_agents, self.args.mixing_embed_dim) if layer == "w1" else (self.args.mixing_embed_dim, 1)
                record["layers"][layer] = {}
                means = {key: parts[layer][key].flatten(0, 1).detach().cpu().mean(0).reshape(shape) for key in columns}
                raw_limit = max(means[key].abs().max().item() for key in columns[:2])
                contribution_limit = max(means[key].abs().max().item() for key in columns[2:5])
                for col, (key, name) in enumerate(zip(columns, names)):
                    values = parts[layer][key].flatten(0, 1).detach().cpu()
                    mean, std = means[key], values.std(0, unbiased=False).reshape(shape)
                    norm = values.norm(dim=-1).mean().item()
                    record["layers"][layer][key] = dict(mean=mean.tolist(), std=std.tolist(),
                        mean_frobenius_norm=norm)
                    ax = axes[row, col]
                    # Raw heads share a scale; weighted contributions share a
                    # second scale. A tiny contribution must LOOK tiny.
                    limit = max(raw_limit if col < 2 else contribution_limit if col < 5 else mean.abs().max().item(), 1e-8)
                    im = ax.imshow(mean.numpy(), aspect="auto", cmap="viridis" if key == "effective" else "RdBu_r",
                        vmin=0 if key == "effective" else -limit, vmax=limit)
                    ax.set_title(layer + " / " + name + "\nmean norm={:.3g}".format(norm), fontsize=10)
                    ax.set_xlabel("hidden" if layer == "w1" else "output")
                    ax.set_ylabel("agent" if layer == "w1" else "hidden")
                    ax.set_xticks(sorted(set(range(0, shape[1], max(1, shape[1] // 4))) | {shape[1] - 1}))
                    ax.set_yticks(list(range(shape[0])) if layer == "w1" else
                        sorted(set(range(0, shape[0], max(1, shape[0] // 4))) | {shape[0] - 1}))
                    fig.colorbar(im, ax=ax, shrink=.7)
            alpha = record["alpha_main"]
            title = "{} / {} / seed={} / step={} / {} / soft-p replay diagnostics".format(
                self.args.env_args.get("map_name"), self.mixer.combination, self.args.seed, t_env,
                "both inputs ones; gate N/A" if self.mixer.secondary_ones else "closed fills {:g}".format(self.mixer.fill_value))
            if alpha is not None:
                title += " / main alpha={:.4f}, secondary={:.4f}".format(alpha, 1 - alpha)
            if self.args.wandb_run_name.startswith("SYNTHETIC"):
                title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
            fig.suptitle(title, fontsize=12)
            image_path = directory / (stem + "_weights.png")
            fig.savefig(image_path, dpi=110)
        finally:
            plt.close(fig)
        if self.logger.use_wandb:
            self.logger._update_wandb_buffer("mixer_branches/weight_components_heatmap",
                self.logger.wandb_module.Image(str(image_path)), t_env)
        return record

    def save_models(self, path):
        path = Path(path)
        self.mac.save_models(str(path))
        (path / "target_mac").mkdir(exist_ok=True)
        self.target_mac.save_models(str(path / "target_mac"))
        th.save(dict(contract=self.contract, mixer=self.mixer.state_dict(),
            target_mixer=self.target_mixer.state_dict(), optimiser=self.optimiser.state_dict(),
            gate_optimiser=None if self.gate_optimiser is None else self.gate_optimiser.state_dict(),
            train_steps=self.train_steps, last_target_update_episode=self.last_target_update_episode,
            last_image_t=self.last_image_t, log_stats_t=self.log_stats_t, ema_td=self.ema_td, ema_kl=self.ema_kl),
            str(path / "onefill_training.th"))

    def load_models(self, path):
        path = Path(path)
        saved = th.load(str(path / "onefill_training.th"), map_location="cpu", weights_only=False)
        if saved["contract"] != self.contract:
            raise ValueError("One-fill checkpoint contract differs; use a fresh run")
        self.mac.load_models(str(path))
        self.target_mac.load_models(str(path / "target_mac"))
        self.mixer.load_state_dict(saved["mixer"])
        self.target_mixer.load_state_dict(saved["target_mixer"])
        self.optimiser.load_state_dict(saved["optimiser"])
        if self.gate_optimiser is not None:
            self.gate_optimiser.load_state_dict(saved["gate_optimiser"])
        for key in ("train_steps", "last_target_update_episode", "last_image_t", "log_stats_t", "ema_td", "ema_kl"):
            setattr(self, key, saved[key])
