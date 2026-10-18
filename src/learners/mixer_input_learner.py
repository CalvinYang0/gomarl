"""One ordinary agent network; only mixer state inputs are controlled."""
import copy
import math
from pathlib import Path

import torch as th
from torch.optim import Adam, RMSprop

from modules.mixers.state_input_control import StateInputControlMixer
from utils.mixer_state_importance import (log_state_snapshot, score_state_slots,
                                          score_state_slot_calibration, complete_episode_returns,
                                          score_state_slot_variance, state_slot_names)
from utils.rl_utils import build_td_lambda_targets
from utils.mixer_gate_regularization import gate_variance_coefficients, gate_penalties


class MixerInputLearner:
    def __init__(self, mac, scheme, logger, args):
        del scheme
        if mac.agent.model_type != "qmix_minimal" or mac.agent.hyper_out_w is not None:
            raise ValueError("Mixer controls require the ordinary fixed qmix_minimal action head")
        if not getattr(args, "double_q", True):
            raise ValueError("Mixer controls require Double-Q")
        self.mac, self.logger, self.args = mac, logger, args
        self.target_mac = copy.deepcopy(mac)
        self.mixer = StateInputControlMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        state_slot_names(args, self.mixer.state_dim)
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.detach_unimportant = bool(getattr(args, "mixer_input_detach_unimportant", False))
        if self.detach_unimportant and not self.mixer.grouped:
            raise ValueError("Unimportant gradient isolation requires two mixer groups")
        self.main_params = (list(mac.parameters()) + list(self.mixer.mixers[0].parameters())
                            if self.detach_unimportant else self.params)
        self.other_params = list(self.mixer.mixers[1].parameters()) if self.detach_unimportant else []
        self.optimiser = self._make_optimiser(self.main_params)
        self.other_optimiser = self._make_optimiser(self.other_params) if self.other_params else None
        self.interval = int(getattr(args, "slot_group_interval", 1000))
        self.budget = int(getattr(args, "slot_importance_samples", 256))
        self.image_interval = int(getattr(args, "slot_heatmap_interval", 100000))
        if min(self.interval, self.budget, self.image_interval) <= 0:
            raise ValueError("Invalid mixer diagnostic schedule")
        self.gate_l1_coef = float(getattr(args, "mixer_gate_l1_coef", 0))
        self.gate_variance_coef = float(getattr(args, "mixer_gate_variance_coef", 0))
        if any(not math.isfinite(c) or c < 0 for c in (self.gate_l1_coef, self.gate_variance_coef)):
            raise ValueError("Gate penalty coefficients must be finite and nonnegative")
        if (self.gate_l1_coef or self.gate_variance_coef) and self.mixer.variant != "static_gate":
            raise ValueError("Gate penalties require static_gate")
        self.gate_variance_interval = int(getattr(args, "mixer_gate_variance_interval", 100))
        if self.gate_variance_coef and (self.gate_variance_interval <= 0 or self.budget < 2):
            raise ValueError("Invalid gate variance schedule/budget")
        self.train_steps = self.last_target_update_episode = 0
        self.last_image_t = -self.image_interval
        self.log_stats_t = -args.learner_log_interval - 1
        self.last_losses = self.last_double_actions = None
        self.last_gate_penalties = None

    def _make_optimiser(self, params):
        args = self.args
        if args.optimizer == "adam":
            return Adam(params, lr=args.lr)
        if args.optimizer == "rmsprop":
            return RMSprop(params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        raise ValueError("Unsupported optimizer")

    def _rollout(self, mac, batch):
        mac.init_hidden(batch.batch_size)
        return th.stack([mac.forward(batch, t) for t in range(batch.max_seq_length)], 1)

    def train(self, batch, t_env, episode_num):
        self.mac.agent.train()
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
        if mask.shape[1] > 1:
            mask[:, 1:] *= (1 - terminated[:, :-1]).cumprod(1)
        important = self._rollout(self.mac, batch)
        chosen = important[:, :-1].gather(-1, batch["actions"][:, :-1].long()).squeeze(-1)
        states = batch["state"][:, :-1]
        joint = self.mixer.values(chosen, states, "train", detach_unimportant=self.detach_unimportant)
        with th.no_grad():
            target_q = self._rollout(self.target_mac, batch)
            selected = important.detach().masked_fill(batch["avail_actions"] == 0, -th.inf).argmax(-1, keepdim=True)
            self.last_double_actions = selected.clone()
            evaluated = target_q.gather(-1, selected).squeeze(-1)
            target_joint = self.target_mixer.values(evaluated, batch["state"], "target")
            targets = build_td_lambda_targets(rewards, terminated * mask, mask, target_joint,
                                             self.args.n_agents, self.args.gamma, self.args.td_lambda)
        error = joint - targets
        expanded = mask.expand_as(error)
        losses = (error.square() * expanded).sum((0, 1)) / expanded.sum((0, 1)).clamp(min=1)
        # Average two independent TD losses, not TD on an averaged Q_tot.
        td_loss = losses.mean()
        if self.gate_variance_coef and self.train_steps % self.gate_variance_interval == 0:
            estimate = gate_variance_coefficients(self.mixer, chosen, states, mask, self.budget)
            self.mixer.gate_variance_probe_update.fill_(self.train_steps)
            self.mixer.gate_variance_probability_at_probe.copy_(self.mixer.gate_logits.sigmoid().detach())
            if estimate is None:
                self.mixer.gate_variance_probe_count.zero_()
                self.mixer.gate_variance_difference.zero_()
                self.mixer.gate_original_variance.zero_()
                self.mixer.gate_masked_variance.zero_()
            else:
                difference, original, masked, count = estimate
                self.mixer.gate_variance_difference.copy_(difference)
                self.mixer.gate_original_variance.copy_(original)
                self.mixer.gate_masked_variance.copy_(masked)
                self.mixer.gate_variance_probe_count.fill_(count)
        l1_penalty, variance_penalty = gate_penalties(self.mixer, self.gate_l1_coef, self.gate_variance_coef)
        loss = td_loss + self.gate_l1_coef * l1_penalty + self.gate_variance_coef * variance_penalty
        self.last_gate_penalties = th.stack((l1_penalty.detach(), variance_penalty.detach()))
        if not th.isfinite(loss):
            raise RuntimeError("Non-finite mixer-input TD loss")
        self.optimiser.zero_grad()
        if self.other_optimiser is not None:
            self.other_optimiser.zero_grad()
        loss.backward()
        norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        other_norm = (th.nn.utils.clip_grad_norm_(self.other_params, self.args.grad_norm_clip)
                      if self.other_params else norm.new_zeros(()))
        if not th.isfinite(norm) or not th.isfinite(other_norm):
            raise RuntimeError("Non-finite mixer-input gradient norm")
        self.optimiser.step()
        if self.other_optimiser is not None:
            self.other_optimiser.step()
        self.train_steps += 1
        self.last_losses = losses.detach().clone()
        regroup = self.mixer.grouped and self.train_steps % self.interval == 0
        image_due = t_env - self.last_image_t >= self.image_interval
        if regroup or image_due:
            evaluated_partition = self.mixer.important_slots.detach().clone() if self.mixer.grouped else None
            # Reconstruct utilities AFTER the update so probes evaluate one
            # coherent current checkpoint. Freeze them and the original TD
            # targets for every perturbation; scoring never re-selects actions.
            with th.no_grad():
                probe_q = self._rollout(self.mac, batch)[:, :-1]
                probe_chosen = probe_q.gather(-1, batch["actions"][:, :-1].long()).squeeze(-1)
            measured = score_state_slots(self.mixer, probe_chosen, states, targets, mask, self.budget)
            migration = None
            if measured is not None:
                qchange, tderror, count = measured
                self.mixer.qchange_scores.copy_(qchange)
                self.mixer.tderror_scores.copy_(tderror)
                self.logger.log_stat("mixer_input/probe_count", count, t_env)
                removal_gain = None
                variance_difference = None
                if self.mixer.variant == "group_calibration":
                    returns, calibration_mask = complete_episode_returns(rewards, terminated, mask, self.args.gamma)
                    estimate = score_state_slot_calibration(self.mixer, probe_chosen, states,
                        returns, calibration_mask, self.budget,
                        perturbation=self.mixer.calibration_perturbation, reference_valid=mask)
                    if estimate is None:
                        self.mixer.calibration_probe_count.zero_()
                        self.logger.log_stat("mixer_input/no_complete_calibration_probes", 1, t_env)
                    else:
                        removal_gain, count, original_error, masked_error = estimate
                        self.mixer.calibration_probe_count.fill_(count)
                        self.mixer.removal_gain_scores.copy_(removal_gain)
                        self.mixer.original_mc_mse.copy_(original_error)
                        self.mixer.masked_mc_mse.copy_(masked_error)
                        self.logger.log_stat("mixer_input/calibration_probe_count", count, t_env)
                        self.logger.log_stat("mixer_input/normalized_removal_gain_mean", removal_gain.mean().item(), t_env)
                if self.mixer.variant == "group_variance":
                    estimate = score_state_slot_variance(self.mixer, probe_chosen, states, mask, self.budget)
                    if estimate is None:
                        self.mixer.variance_probe_count.zero_()
                        self.logger.log_stat("mixer_input/no_valid_variance_probes", 1, t_env)
                    else:
                        variance_difference, count, original_var, masked_var = estimate
                        self.mixer.variance_probe_count.fill_(count)
                        self.mixer.variance_difference_scores.copy_(variance_difference)
                        self.mixer.original_qtot_variance.copy_(original_var)
                        self.mixer.masked_qtot_variance.copy_(masked_var)
                        self.logger.log_stat("mixer_input/variance_probe_count", count, t_env)
                        self.logger.log_stat("mixer_input/qtot_variance_difference_mean", variance_difference.mean().item(), t_env)
                if regroup:
                    ready = ((self.mixer.variant != "group_calibration" or removal_gain is not None) and
                             (self.mixer.variant != "group_variance" or variance_difference is not None))
                    if ready:
                        migration = self.mixer.apply_scores(qchange, tderror, removal_gain, variance_difference)
                    # Partition ownership is structural, not lagged parameters.
                    self.target_mixer.sync_partition(self.mixer)
            else:
                self.logger.log_stat("mixer_input/no_valid_probes", 1, t_env)
            log_state_snapshot(self.mixer, self.args, self.logger, t_env, self.train_steps,
                               migration, images=image_due or regroup, evaluated_partition=evaluated_partition)
            if image_due:
                self.last_image_t = t_env
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            if self.gate_l1_coef or self.gate_variance_coef:
                self.logger.log_stat("mixer_input/loss_total", loss.item(), t_env)
                self.logger.log_stat("mixer_input/gate_l1", l1_penalty.item(), t_env)
                self.logger.log_stat("mixer_input/gate_variance_penalty", variance_penalty.item(), t_env)
                self.logger.log_stat("mixer_input/gate_l1_weighted", self.gate_l1_coef * l1_penalty.item(), t_env)
                self.logger.log_stat("mixer_input/gate_variance_weighted", self.gate_variance_coef * variance_penalty.item(), t_env)
                if self.gate_variance_coef:
                    self.logger.log_stat("mixer_input/gate_variance_probe_count", self.mixer.gate_variance_probe_count.item(), t_env)
                p = self.mixer.gate_logits.sigmoid().detach()
                self.logger.log_stat("mixer_input/gate_probability_mean", p.mean().item(), t_env)
                self.logger.log_stat("mixer_input/gate_eval_open_count", (p > self.mixer.threshold).sum().item(), t_env)
            self.logger.log_stat("loss_td_important", losses[0].item(), t_env)
            if len(losses) == 2:
                self.logger.log_stat("loss_td_unimportant", losses[1].item(), t_env)
            self.logger.log_stat("grad_norm", norm.item(), t_env)
            if self.detach_unimportant:
                self.logger.log_stat("mixer_input/grad_norm_unimportant", other_norm.item(), t_env)
                self.logger.log_stat("mixer_input/unimportant_agent_gradient_detached", 1, t_env)
            self.logger.log_stat("td_error_abs", (error.abs() * expanded).sum().item() / expanded.sum().clamp(min=1).item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), str(Path(path) / "mixer.th"))
        th.save(self.optimiser.state_dict(), str(Path(path) / "opt.th"))
        if self.other_optimiser is not None:
            th.save(self.other_optimiser.state_dict(), str(Path(path) / "opt_unimportant.th"))
        th.save(dict(train_steps=self.train_steps, last_target_update_episode=self.last_target_update_episode,
                     last_image_t=self.last_image_t, detach_unimportant=self.detach_unimportant,
                     append_mask=self.mixer.append_mask, variant=self.mixer.variant,
                     calibration_perturbation=self.mixer.calibration_perturbation,
                     gate_l1_coef=self.gate_l1_coef, gate_variance_coef=self.gate_variance_coef),
                str(Path(path) / "mixer_input_schedule.th"))

    def load_models(self, path):
        schedule = th.load(str(Path(path) / "mixer_input_schedule.th"), map_location="cpu")
        if bool(schedule.get("detach_unimportant", False)) != self.detach_unimportant:
            raise ValueError("Checkpoint gradient routing differs; use a fresh run")
        if bool(schedule.get("append_mask", self.mixer.grouped)) != self.mixer.append_mask:
            raise ValueError("Checkpoint state input encoding differs; use a fresh run")
        if schedule.get("variant", self.mixer.variant) != self.mixer.variant:
            raise ValueError("Checkpoint importance criterion differs; use a fresh run")
        if schedule.get("calibration_perturbation", "mask_zero") != self.mixer.calibration_perturbation:
            raise ValueError("Checkpoint calibration perturbation differs; use a fresh run")
        for key in ("gate_l1_coef", "gate_variance_coef"):
            if float(schedule.get(key, 0)) != getattr(self, key):
                raise ValueError("Checkpoint gate regularization differs; use a fresh run")
        self.mac.load_models(path)
        self.mixer.load_state_dict(th.load(str(Path(path) / "mixer.th"), map_location="cpu"))
        self.optimiser.load_state_dict(th.load(str(Path(path) / "opt.th"), map_location="cpu"))
        if self.other_optimiser is not None:
            self.other_optimiser.load_state_dict(th.load(str(Path(path) / "opt_unimportant.th"), map_location="cpu"))
        for key, value in schedule.items():
            if key not in {"append_mask", "variant", "calibration_perturbation"}:
                setattr(self, key, value)
        self._update_targets()
