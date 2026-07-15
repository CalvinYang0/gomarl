import copy
import os
from contextlib import nullcontext

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets


class CleanLearner:
    def __init__(self, mac, scheme, logger, args):
        del scheme
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger
        self.params = [
            parameter
            for name, parameter in self.mac.named_parameters()
            if not name.endswith("semantic_probe_scale")
        ]
        self.relation_mixer_gate = None
        self.target_relation_mixer_gate = None
        self.latest_relation_gate_mean = None
        self.latest_relation_gate_std = None
        self.use_amp = bool(getattr(args, "use_amp", False)) and bool(getattr(args, "use_cuda", False)) and th.cuda.is_available()
        self.amp_scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)

        if bool(getattr(args, "clean_relation_mixer_gate", False)):
            cond_dim = int(getattr(args, "clean_condition_dim", args.hypernet_embed))
            self.relation_mixer_gate = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cond_dim, 1),
            )
            self.target_relation_mixer_gate = copy.deepcopy(self.relation_mixer_gate)
            self.params += list(self.relation_mixer_gate.parameters())

        mixer_name = getattr(args, "mixer", "qmix")
        if mixer_name == "qmix":
            self.mixer = QMixer(args)
            self.target_mixer = copy.deepcopy(self.mixer)
            self.params += list(self.mixer.parameters())
        elif mixer_name == "vdn":
            self.mixer = VDNMixer()
            self.target_mixer = copy.deepcopy(self.mixer)
        elif mixer_name in {"none", None}:
            self.mixer = None
            self.target_mixer = None
        else:
            raise ValueError("Unsupported mixer={} for CleanLearner.".format(mixer_name))

        if getattr(args, "optimizer", "adam") == "adam":
            self.optimiser = Adam(self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
            )

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.semantic_router_audit_interval = int(
            getattr(args, "clean_semantic_router_audit_interval", 250000)
        )
        self.last_semantic_router_audit_t = -self.semantic_router_audit_interval
        self.latest_semantic_counterfactual_stats = {}

    def _amp_context(self):
        return th.cuda.amp.autocast(enabled=True) if self.use_amp else nullcontext()

    @staticmethod
    def _semantic_router(mac):
        capturer = getattr(getattr(mac, "agent", None), "rpg_relation_capturer", None)
        if capturer is None or not bool(getattr(capturer, "semantic_router_active", False)):
            return None
        return capturer

    def _sync_semantic_router_to_target(self, router):
        target_router = self._semantic_router(self.target_mac)
        if router is not None and target_router is not None:
            target_router.copy_semantic_router_from(router)

    def _counterfactual_td_loss(self, batch, actions, targets, td_mask):
        mac_out = []
        relation_conditions = [] if self.relation_mixer_gate is not None else None
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out.append(self.mac.forward(batch, t=t))
            if relation_conditions is not None:
                relation_conditions.append(self.mac.latest_condition)
        mac_out = th.stack(mac_out, dim=1)
        chosen_q = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        if self.mixer is not None:
            if relation_conditions is not None:
                relation_conditions = th.stack(relation_conditions, dim=1)
                chosen_q = self._apply_relation_gate(
                    chosen_q, relation_conditions[:, :-1], target=False
                )
            chosen_q = self.mixer(chosen_q, batch["state"][:, :-1])
        error = (chosen_q - targets.detach()) * td_mask
        return error.pow(2).sum() / td_mask.sum().clamp(min=1.0)

    def _audit_semantic_counterfactual(self, batch, actions, targets, td_mask, router, t_env):
        if router is None or not router.semantic_router_needs_counterfactual():
            return False
        if bool(router.semantic_route_frozen.item()):
            return False
        if t_env - self.last_semantic_router_audit_t < self.semantic_router_audit_interval:
            return False

        scores = []
        stats = {}
        try:
            with th.no_grad():
                for group_index, group_name in enumerate(router.semantic_names):
                    router.set_semantic_route_override(group_index, True)
                    token_loss = self._counterfactual_td_loss(
                        batch, actions, targets, td_mask
                    )
                    router.set_semantic_route_override(group_index, False)
                    bias_loss = self._counterfactual_td_loss(
                        batch, actions, targets, td_mask
                    )
                    score = bias_loss - token_loss
                    scores.append(score)
                    stats[
                        f"semantic_cf_{group_name}_bias_minus_token_loss"
                    ] = score.detach()
        finally:
            router.clear_semantic_route_override()

        router.update_semantic_router(t_env, th.stack(scores))
        self._sync_semantic_router_to_target(router)
        self.latest_semantic_counterfactual_stats = stats
        self.last_semantic_router_audit_t = t_env
        return True

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        semantic_router = self._semantic_router(self.mac)
        generated_parameter_graphs = []

        with self._amp_context():
            mac_out = []
            teacher_mac_out = []
            relation_conditions = [] if self.relation_mixer_gate is not None else None
            aux_losses = []
            aux_stat_values = {}
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                mac_out.append(self.mac.forward(batch, t=t))
                if (
                    semantic_router is not None
                    and semantic_router.semantic_router_needs_parameter_graph()
                ):
                    generated_parameter_graph = getattr(
                        self.mac, "latest_generated_interaction_head_graph", None
                    )
                    if generated_parameter_graph is not None:
                        generated_parameter_graphs.append(generated_parameter_graph)
                if relation_conditions is not None:
                    condition = getattr(self.mac, "latest_condition", None)
                    if condition is None:
                        raise RuntimeError("clean_relation_mixer_gate=True requires a relation-conditioned clean model.")
                    relation_conditions.append(condition)
                aux_loss = getattr(self.mac, "latest_aux_loss", None)
                if aux_loss is not None:
                    aux_losses.append(aux_loss)
                for stat_name, stat_value in getattr(self.mac, "latest_aux_stats", {}).items():
                    if stat_value is None:
                        continue
                    if stat_name.startswith("semantic_route_"):
                        continue
                    if not th.is_tensor(stat_value):
                        stat_value = th.as_tensor(stat_value, device=batch.device, dtype=th.float32)
                    aux_stat_values.setdefault(stat_name, []).append(stat_value.detach().float())
                teacher_q = getattr(self.mac, "latest_teacher_q", None)
                if teacher_q is not None:
                    teacher_mac_out.append(teacher_q)
            mac_out = th.stack(mac_out, dim=1)
            teacher_mac_out = (
                th.stack(teacher_mac_out, dim=1)
                if len(teacher_mac_out) == batch.max_seq_length
                else None
            )
            if relation_conditions is not None:
                relation_conditions = th.stack(relation_conditions, dim=1)

            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            with th.no_grad():
                target_mac_out = []
                target_relation_conditions = [] if self.target_relation_mixer_gate is not None else None
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_mac_out.append(self.target_mac.forward(batch, t=t))
                    if target_relation_conditions is not None:
                        condition = getattr(self.target_mac, "latest_condition", None)
                        if condition is None:
                            raise RuntimeError("clean_relation_mixer_gate=True requires a relation-conditioned clean model.")
                        target_relation_conditions.append(condition)
                target_mac_out = th.stack(target_mac_out, dim=1)
                if target_relation_conditions is not None:
                    target_relation_conditions = th.stack(target_relation_conditions, dim=1)

                mac_out_detach = mac_out.detach().clone()
                mask_value = th.finfo(mac_out_detach.dtype).min if mac_out_detach.is_floating_point() else -9999999
                mac_out_detach[avail_actions == 0] = mask_value
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
                target_max_agent_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

                if self.target_mixer is not None:
                    if self.target_relation_mixer_gate is not None:
                        target_max_agent_qvals = self._apply_relation_gate(
                            target_max_agent_qvals, target_relation_conditions, target=True
                        )
                    target_max_qvals = self.target_mixer(target_max_agent_qvals, batch["state"])
                else:
                    target_max_qvals = target_max_agent_qvals

                targets = build_td_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

            if self.mixer is not None:
                if self.relation_mixer_gate is not None:
                    chosen_action_qvals = self._apply_relation_gate(
                        chosen_action_qvals, relation_conditions[:, :-1], target=False
                    )
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

            td_error = chosen_action_qvals - targets.detach()
            td_mask = mask.expand_as(td_error)
            masked_td_error = td_error * td_mask
            td_loss = (masked_td_error.pow(2).sum()) / td_mask.sum().clamp(min=1.0)
            aux_loss = th.stack(aux_losses).mean() if aux_losses else td_loss.new_zeros(())
            teacher_td_loss = td_loss.new_zeros(())
            if teacher_mac_out is not None:
                teacher_chosen_qvals = th.gather(teacher_mac_out[:, :-1], dim=3, index=actions).squeeze(3)
                if self.mixer is not None:
                    teacher_chosen_qvals = self.mixer(teacher_chosen_qvals, batch["state"][:, :-1])
                teacher_td_error = teacher_chosen_qvals - targets.detach()
                teacher_masked_td_error = teacher_td_error * td_mask
                teacher_td_loss = (teacher_masked_td_error.pow(2).sum()) / td_mask.sum().clamp(min=1.0)
            loss = td_loss + aux_loss + float(getattr(self.args, "clean_relation_teacher_td_coef", 0.0)) * teacher_td_loss

        parameter_sensitivity_score = None
        if (
            semantic_router is not None
            and semantic_router.semantic_router_needs_parameter_graph()
            and generated_parameter_graphs
        ):
            # Hutchinson-style random projections estimate how strongly each
            # semantic scale changes generated head parameters without forming
            # the full parameter Jacobian.
            generated_parameter_projection = th.stack(
                [
                    (
                        generated
                        * generated.new_empty(generated.shape)
                        .bernoulli_(0.5)
                        .mul_(2.0)
                        .sub_(1.0)
                    ).sum()
                    / (generated.numel() ** 0.5)
                    for generated in generated_parameter_graphs
                ]
            ).mean()
            parameter_sensitivity_score = th.autograd.grad(
                generated_parameter_projection,
                semantic_router.semantic_probe_scale,
                retain_graph=True,
                allow_unused=True,
            )[0]

        self.optimiser.zero_grad()
        if semantic_router is not None and semantic_router.semantic_probe_scale is not None:
            semantic_router.semantic_probe_scale.grad = None
        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optimiser)
            semantic_gradient = (
                None
                if semantic_router is None or semantic_router.semantic_probe_scale is None
                else semantic_router.semantic_probe_scale.grad.detach()
                / float(self.amp_scaler.get_scale())
            )
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.amp_scaler.step(self.optimiser)
            self.amp_scaler.update()
        else:
            loss.backward()
            semantic_gradient = (
                None
                if semantic_router is None or semantic_router.semantic_probe_scale is None
                else semantic_router.semantic_probe_scale.grad.detach().clone()
            )
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        if semantic_router is not None:
            if semantic_router.semantic_router_mode in {
                "observer_consistency",
                "temporal_stability",
            }:
                semantic_router.update_semantic_router(t_env)
            elif semantic_router.semantic_router_mode in {
                "gradient_importance",
                "gradient_consistency",
            } and semantic_gradient is not None:
                semantic_router.update_semantic_router(t_env, semantic_gradient)
            elif (
                semantic_router.semantic_router_needs_parameter_graph()
                and parameter_sensitivity_score is not None
            ):
                semantic_router.update_semantic_router(
                    t_env, parameter_sensitivity_score.detach()
                )
            if semantic_router.semantic_probe_scale is not None:
                semantic_router.semantic_probe_scale.grad = None
            self._sync_semantic_router_to_target(semantic_router)

        self._audit_semantic_counterfactual(
            batch, actions, targets, td_mask, semantic_router, t_env
        )

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            if aux_losses:
                self.logger.log_stat("loss_aux", aux_loss.item(), t_env)
            for stat_name, values in aux_stat_values.items():
                if values:
                    self.logger.log_stat(stat_name, th.stack(values).mean().item(), t_env)
            if semantic_router is not None:
                for stat_name, stat_value in semantic_router.semantic_router_stats().items():
                    self.logger.log_stat(stat_name, float(stat_value.item()), t_env)
                for stat_name, stat_value in self.latest_semantic_counterfactual_stats.items():
                    self.logger.log_stat(stat_name, float(stat_value.item()), t_env)
            if teacher_mac_out is not None:
                self.logger.log_stat("loss_teacher_td", teacher_td_loss.item(), t_env)
            if self.latest_relation_gate_mean is not None:
                self.logger.log_stat("relation_gate_mean", self.latest_relation_gate_mean, t_env)
                self.logger.log_stat("relation_gate_std", self.latest_relation_gate_std, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = td_mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                masked_td_error.abs().sum().item() / max(mask_elems, 1.0),
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * td_mask).sum().item()
                / max(mask_elems, 1.0),
                t_env,
            )
            self.logger.log_stat(
                "target_mean", (targets * td_mask).sum().item() / max(mask_elems, 1.0),
                t_env,
            )
            self.log_stats_t = t_env

    def _apply_relation_gate(self, agent_qs, relation_conditions, target=False):
        gate_net = self.target_relation_mixer_gate if target else self.relation_mixer_gate
        gate_logits = gate_net(relation_conditions).squeeze(-1)
        gates = F.softmax(gate_logits, dim=-1) * self.args.n_agents
        if not target:
            self.latest_relation_gate_mean = gates.mean().item()
            self.latest_relation_gate_std = gates.std(unbiased=False).item()
        return agent_qs * gates

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.relation_mixer_gate is not None:
            self.target_relation_mixer_gate.load_state_dict(self.relation_mixer_gate.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.relation_mixer_gate is not None:
            self.relation_mixer_gate.cuda()
            self.target_relation_mixer_gate.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.relation_mixer_gate is not None:
            th.save(self.relation_mixer_gate.state_dict(), "{}/relation_mixer_gate.th".format(path))
        if self.use_amp:
            th.save(self.amp_scaler.state_dict(), "{}/amp_scaler.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.relation_mixer_gate is not None:
            self.relation_mixer_gate.load_state_dict(
                th.load("{}/relation_mixer_gate.th".format(path), map_location=lambda storage, loc: storage)
            )
            self.target_relation_mixer_gate.load_state_dict(self.relation_mixer_gate.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        amp_scaler_path = "{}/amp_scaler.th".format(path)
        if self.use_amp and os.path.exists(amp_scaler_path):
            self.amp_scaler.load_state_dict(th.load(amp_scaler_path, map_location=lambda storage, loc: storage))
