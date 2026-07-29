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
            for name, parameter in self.mac.agent.named_parameters()
            if not name.endswith(("semantic_probe_scale", "semantic_route_probe"))
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
        self.semantic_parameter_probe_timesteps = max(
            1, int(getattr(args, "clean_semantic_parameter_probe_timesteps", 4))
        )
        self.semantic_observation_probe_timesteps = max(
            1, int(getattr(args, "clean_semantic_observation_probe_timesteps", 8))
        )
        self.semantic_binary_audit_batch_size = max(
            1, int(getattr(args, "clean_semantic_binary_audit_batch_size", 8))
        )
        self.semantic_binary_rehearsal_updates = max(
            1, int(getattr(args, "clean_semantic_binary_rehearsal_updates", 4))
        )
        self.semantic_binary_rehearsal_remaining = 0
        self.semantic_binary_audit_pending = False
        self.last_semantic_router_audit_t = -self.semantic_router_audit_interval
        self.latest_semantic_counterfactual_stats = {}
        self.last_semantic_route_version_logged = -1

    def _amp_context(self):
        return th.cuda.amp.autocast(enabled=True) if self.use_amp else nullcontext()

    @staticmethod
    def _semantic_router(mac):
        capturer = getattr(getattr(mac, "agent", None), "rpg_relation_capturer", None)
        if capturer is None or not bool(getattr(capturer, "semantic_router_active", False)):
            return None
        return capturer

    @staticmethod
    def _uniform_probe_times(sequence_length, probe_count):
        sequence_length = max(1, int(sequence_length))
        probe_count = min(max(1, int(probe_count)), sequence_length)
        if probe_count == 1:
            return {sequence_length // 2}
        return {
            round(index * (sequence_length - 1) / (probe_count - 1))
            for index in range(probe_count)
        }

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
                current_route = router.semantic_token_route.detach().clone()
                router.clear_semantic_route_override()
                baseline_loss = self._counterfactual_td_loss(
                    batch, actions, targets, td_mask
                )
                for group_index, group_name in enumerate(router.semantic_names):
                    current_token_branch = bool(current_route[group_index].item())
                    router.set_semantic_route_override(
                        group_index, not current_token_branch
                    )
                    alternative_loss = self._counterfactual_td_loss(
                        batch, actions, targets, td_mask
                    )
                    if current_token_branch:
                        token_loss = baseline_loss
                        bias_loss = alternative_loss
                    else:
                        token_loss = alternative_loss
                        bias_loss = baseline_loss
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

    def _audit_semantic_gradient_importance(
        self, batch, actions, targets, td_mask, router, t_env
    ):
        if router is None or not router.semantic_router_needs_independent_audit():
            return False
        if t_env - self.last_semantic_router_audit_t < self.semantic_router_audit_interval:
            return False

        audit_gradient = None
        audit_loss = None
        try:
            router.set_semantic_full_input_audit(True)
            if router.semantic_probe_scale.grad is not None:
                router.semantic_probe_scale.grad = None
            audit_loss = self._counterfactual_td_loss(
                batch, actions, targets, td_mask
            )
            audit_gradient = th.autograd.grad(
                audit_loss,
                router.semantic_probe_scale,
                allow_unused=True,
            )[0]
        finally:
            router.set_semantic_full_input_audit(False)
            if router.semantic_probe_scale.grad is not None:
                router.semantic_probe_scale.grad = None

        if audit_gradient is None:
            return False
        router.update_semantic_router(t_env, audit_gradient.detach())
        self._sync_semantic_router_to_target(router)
        self.latest_semantic_counterfactual_stats = {
            "semantic_audit_td_loss": audit_loss.detach(),
            "semantic_audit_gradient_abs_mean": audit_gradient.detach().abs().mean(),
        }
        self.last_semantic_router_audit_t = t_env
        return True

    def _binary_audit_due(self, router, t_env):
        return (
            router is not None
            and router.semantic_router_needs_binary_audit()
            and t_env - self.last_semantic_router_audit_t
            >= self.semantic_router_audit_interval
        )

    def _start_semantic_binary_rehearsal(self, router, t_env):
        if not self.semantic_binary_audit_pending and self._binary_audit_due(
            router, t_env
        ):
            self.semantic_binary_audit_pending = True
            self.semantic_binary_rehearsal_remaining = (
                self.semantic_binary_rehearsal_updates
            )
        return self.semantic_binary_audit_pending

    def _set_binary_rehearsal_route(self, enabled):
        for mac in (self.mac, self.target_mac):
            router = self._semantic_router(mac)
            if router is not None and router.semantic_router_needs_binary_audit():
                router.set_semantic_full_input_audit(enabled)

    def _binary_audit_batch(self, batch):
        audit_batch_size = min(
            batch.batch_size, self.semantic_binary_audit_batch_size
        )
        return batch[:audit_batch_size]

    def _generated_parameter_snapshots(self, batch):
        probe_times = self._uniform_probe_times(
            batch.max_seq_length, self.semantic_parameter_probe_timesteps
        )
        snapshots = []
        self.mac.init_hidden(batch.batch_size)
        try:
            with th.no_grad():
                for timestep in range(batch.max_seq_length):
                    self.mac.agent.capture_semantic_parameter_graph = (
                        timestep in probe_times
                    )
                    self.mac.forward(batch, t=timestep)
                    if timestep not in probe_times:
                        continue
                    generated = getattr(
                        self.mac,
                        "latest_generated_interaction_head_graph",
                        None,
                    )
                    if generated is not None:
                        snapshots.append(generated.detach().float().reshape(-1))
        finally:
            self.mac.agent.capture_semantic_parameter_graph = False
        return snapshots

    @staticmethod
    def _parameter_intervention_score(baseline, alternative):
        if len(baseline) != len(alternative) or not baseline:
            return None
        squared_difference = baseline[0].new_zeros(())
        squared_baseline = baseline[0].new_zeros(())
        element_count = 0
        for baseline_tensor, alternative_tensor in zip(baseline, alternative):
            if baseline_tensor.shape != alternative_tensor.shape:
                return None
            squared_difference = squared_difference + (
                alternative_tensor - baseline_tensor
            ).pow(2).sum()
            squared_baseline = squared_baseline + baseline_tensor.pow(2).sum()
            element_count += baseline_tensor.numel()
        denominator = (squared_baseline / max(element_count, 1)).sqrt().clamp(
            min=1e-8
        )
        return (
            squared_difference / max(element_count, 1)
        ).sqrt() / denominator

    @staticmethod
    def _semantic_group_rms(batch, router):
        semantic_dim = len(router.semantic_names)
        observations = batch["obs"][:, :-1, :, :semantic_dim].detach().float()
        field_ids = router.semantic_field_ids.to(observations.device)
        rms_values = []
        for group_index in range(router.semantic_field_count):
            group_values = observations[..., field_ids == group_index]
            if group_values.numel() == 0:
                rms_values.append(observations.new_zeros(()))
            else:
                rms_values.append(group_values.pow(2).mean().sqrt())
        return th.stack(rms_values)

    def _audit_semantic_binary(
        self, batch, actions, targets, td_mask, router, t_env
    ):
        if router is None or not router.semantic_router_needs_binary_audit():
            return False
        if not self.semantic_binary_audit_pending:
            return False
        if self.semantic_binary_rehearsal_remaining > 0:
            return False

        audit_batch = self._binary_audit_batch(batch)
        audit_size = audit_batch.batch_size
        audit_actions = actions[:audit_size]
        audit_targets = targets[:audit_size]
        audit_td_mask = td_mask[:audit_size]
        group_scores = []
        baseline_value = None
        try:
            router.set_semantic_full_input_audit(True)
            router.set_semantic_binary_audit_group(None)
            if router.mlp_binary_audit_mode == "td_loss":
                with th.no_grad():
                    baseline_value = self._counterfactual_td_loss(
                        audit_batch,
                        audit_actions,
                        audit_targets,
                        audit_td_mask,
                    )
            else:
                baseline_parameters = self._generated_parameter_snapshots(
                    audit_batch
                )
                if not baseline_parameters:
                    return False
                baseline_value = th.stack(
                    [parameter.pow(2).mean() for parameter in baseline_parameters]
                ).mean().sqrt()

            for group_index in range(router.semantic_field_count):
                router.set_semantic_binary_audit_group(group_index)
                if router.mlp_binary_audit_mode == "td_loss":
                    with th.no_grad():
                        alternative_loss = self._counterfactual_td_loss(
                            audit_batch,
                            audit_actions,
                            audit_targets,
                            audit_td_mask,
                        )
                    score = (alternative_loss - baseline_value) / baseline_value.abs().clamp(
                        min=1e-8
                    )
                else:
                    alternative_parameters = self._generated_parameter_snapshots(
                        audit_batch
                    )
                    score = self._parameter_intervention_score(
                        baseline_parameters, alternative_parameters
                    )
                    if score is None:
                        return False
                group_scores.append(score.detach())
        finally:
            router.set_semantic_full_input_audit(False)

        group_scores = th.stack(group_scores)
        field_ids = router.semantic_field_ids.to(group_scores.device)
        slot_scores = group_scores.index_select(0, field_ids)
        group_rms = self._semantic_group_rms(audit_batch, router).to(
            group_scores.device
        )
        normalized_scores = group_scores / group_rms.clamp(min=1e-6)
        centered_scores = group_scores - group_scores.mean()
        centered_rms = group_rms - group_rms.mean()
        rms_correlation = (
            centered_scores.mul(centered_rms).sum()
            / (
                centered_scores.pow(2).sum().sqrt()
                * centered_rms.pow(2).sum().sqrt()
            ).clamp(min=1e-8)
        )

        route_update_skipped = (
            router.mlp_binary_audit_mode == "td_loss"
            and group_scores.max().item() <= 1e-8
        )
        if not route_update_skipped:
            router.update_semantic_router(t_env, slot_scores)
            self._sync_semantic_router_to_target(router)
        self.latest_semantic_counterfactual_stats = {
            "semantic_binary_audit_baseline": baseline_value.detach(),
            "semantic_binary_audit_score_mean": group_scores.mean(),
            "semantic_binary_audit_score_max": group_scores.max(),
            "semantic_binary_audit_input_rms_mean": group_rms.mean(),
            "semantic_binary_audit_rms_normalized_score_mean": (
                normalized_scores.mean()
            ),
            "semantic_binary_audit_score_rms_correlation": rms_correlation,
            "semantic_binary_audit_rehearsal_updates": group_scores.new_tensor(
                float(self.semantic_binary_rehearsal_updates)
            ),
            "semantic_binary_audit_route_update_skipped": (
                group_scores.new_tensor(float(route_update_skipped))
            ),
        }
        self.last_semantic_router_audit_t = t_env
        self.semantic_binary_audit_pending = False
        return True

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        semantic_router = self._semantic_router(self.mac)
        binary_rehearsal_active = self._start_semantic_binary_rehearsal(
            semantic_router, t_env
        )
        if binary_rehearsal_active:
            self._set_binary_rehearsal_route(True)
        generated_parameter_graphs = []
        parameter_probe_times = set()
        observation_probe_times = set()
        if (
            semantic_router is not None
            and semantic_router.semantic_router_needs_parameter_graph()
        ):
            parameter_probe_times = self._uniform_probe_times(
                batch.max_seq_length, self.semantic_parameter_probe_timesteps
            )
        if (
            semantic_router is not None
            and semantic_router.semantic_router_needs_observation_score()
        ):
            observation_probe_times = self._uniform_probe_times(
                batch.max_seq_length, self.semantic_observation_probe_timesteps
            )

        with self._amp_context():
            mac_out = []
            teacher_mac_out = []
            relation_conditions = [] if self.relation_mixer_gate is not None else None
            aux_losses = []
            aux_stat_values = {}
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                self.mac.agent.capture_semantic_parameter_graph = (
                    t in parameter_probe_times
                )
                if semantic_router is not None:
                    semantic_router.capture_semantic_observation_score = (
                        t in observation_probe_times
                    )
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
            self.mac.agent.capture_semantic_parameter_graph = False
            if semantic_router is not None:
                semantic_router.capture_semantic_observation_score = False
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
        if semantic_router is not None and semantic_router.semantic_route_probe is not None:
            semantic_router.semantic_route_probe.grad = None
        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optimiser)
            semantic_gradient = (
                None
                if (
                    semantic_router is None
                    or semantic_router.semantic_probe_scale is None
                    or semantic_router.semantic_probe_scale.grad is None
                )
                else semantic_router.semantic_probe_scale.grad.detach()
                / float(self.amp_scaler.get_scale())
            )
            semantic_route_gradient = (
                None
                if (
                    semantic_router is None
                    or semantic_router.semantic_route_probe is None
                    or semantic_router.semantic_route_probe.grad is None
                )
                else semantic_router.semantic_route_probe.grad.detach()
                / float(self.amp_scaler.get_scale())
            )
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.amp_scaler.step(self.optimiser)
            self.amp_scaler.update()
        else:
            loss.backward()
            semantic_gradient = (
                None
                if (
                    semantic_router is None
                    or semantic_router.semantic_probe_scale is None
                    or semantic_router.semantic_probe_scale.grad is None
                )
                else semantic_router.semantic_probe_scale.grad.detach().clone()
            )
            semantic_route_gradient = (
                None
                if (
                    semantic_router is None
                    or semantic_router.semantic_route_probe is None
                    or semantic_router.semantic_route_probe.grad is None
                )
                else semantic_router.semantic_route_probe.grad.detach().clone()
            )
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        if binary_rehearsal_active:
            self.semantic_binary_rehearsal_remaining = max(
                0, self.semantic_binary_rehearsal_remaining - 1
            )

        if semantic_router is not None:
            if semantic_router.semantic_router_mode in {
                "observer_consistency",
                "temporal_stability",
            }:
                semantic_router.update_semantic_router(t_env)
            elif semantic_router.semantic_router_mode in {
                "gradient_importance",
                "gradient_consistency",
            } and (
                not semantic_router.semantic_router_needs_independent_audit()
                and semantic_gradient is not None
            ):
                semantic_router.update_semantic_router(t_env, semantic_gradient)
            elif (
                semantic_router.semantic_router_needs_parameter_graph()
                and parameter_sensitivity_score is not None
            ):
                semantic_router.update_semantic_router(
                    t_env, parameter_sensitivity_score.detach()
                )
            elif (
                semantic_router.semantic_router_mode == "counterfactual"
                and semantic_route_gradient is not None
            ):
                # First-order approximation of L_bias - L_token. Positive
                # values mean routing the slot through TOKEN should lower TD loss.
                semantic_router.update_semantic_router(
                    t_env, -semantic_route_gradient
                )
            if semantic_router.semantic_probe_scale is not None:
                semantic_router.semantic_probe_scale.grad = None
            if semantic_router.semantic_route_probe is not None:
                semantic_router.semantic_route_probe.grad = None
            self._sync_semantic_router_to_target(semantic_router)

        self._audit_semantic_counterfactual(
            batch, actions, targets, td_mask, semantic_router, t_env
        )
        self._audit_semantic_gradient_importance(
            batch, actions, targets, td_mask, semantic_router, t_env
        )
        self._audit_semantic_binary(
            batch, actions, targets, td_mask, semantic_router, t_env
        )
        if binary_rehearsal_active:
            self._set_binary_rehearsal_route(False)

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
                route_version = int(semantic_router.semantic_route_version.item())
                if route_version != self.last_semantic_route_version_logged:
                    self.logger.console_logger.info(
                        "Semantic slot route | t_env={} | {}".format(
                            t_env, semantic_router.semantic_route_summary()
                        )
                    )
                    self.last_semantic_route_version_logged = route_version
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
