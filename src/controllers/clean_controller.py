import math

import torch as th

from .basic_controller import BasicMAC


class CleanMAC(BasicMAC):
    TRAJECTORY_PARAMETER_MODEL_SUFFIX = (
        "dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
    )

    @staticmethod
    def _fixed_parameter_projection(parameter_parts, projection_dim):
        """Project generated parameter blocks with a frozen signed hash.

        The arithmetic hash is deterministic across rollout and learner
        processes, has no trainable state, and avoids materializing a dense
        [projection_dim, parameter_count] matrix.
        """
        if not parameter_parts:
            return None
        projection_dim = int(projection_dim)
        if projection_dim <= 0:
            raise ValueError("trajectory parameter projection_dim must be positive")
        projected = None
        parameter_count = 0
        leading_size = None
        for block_index, parameter in enumerate(parameter_parts):
            if parameter is None or parameter.dim() < 1:
                raise RuntimeError("Invalid generated parameter block for projection")
            flat = parameter.reshape(parameter.shape[0], -1)
            if leading_size is None:
                leading_size = flat.shape[0]
                projected = flat.new_zeros(leading_size, projection_dim)
            elif flat.shape[0] != leading_size:
                raise RuntimeError("Generated parameter blocks have inconsistent batches")

            indices = th.arange(flat.shape[1], device=flat.device, dtype=th.long)
            seed = 104729 * (block_index + 1)
            buckets = (indices * 1103515245 + seed) % projection_dim
            sign_hash = (indices + seed) * 2654435761
            sign_hash = sign_hash ^ (sign_hash >> 16)
            sign_bits = sign_hash % 2
            signs = sign_bits.to(flat.dtype).mul_(2.0).sub_(1.0)
            projected.scatter_add_(
                1,
                buckets.unsqueeze(0).expand(flat.shape[0], -1),
                flat * signs.unsqueeze(0),
            )
            parameter_count += flat.shape[1]

        # Keep the projected scale comparable when GRF and SMAC generate
        # different numbers of head parameters.
        projected = projected / math.sqrt(
            max(float(parameter_count) / float(projection_dim), 1.0)
        )
        return projected

    def set_dynamic_branch_gate_t_env(self, t_env):
        if hasattr(self.agent, "set_dynamic_branch_gate_t_env"):
            self.agent.set_dynamic_branch_gate_t_env(t_env)

    def set_dynamic_branch_gate_force_open(self, enabled):
        if hasattr(self.agent, "set_dynamic_branch_gate_force_open"):
            self.agent.set_dynamic_branch_gate_force_open(enabled)

    def set_td_parameter_sampling_enabled(self, enabled):
        if hasattr(self.agent, "set_td_parameter_sampling_enabled"):
            self.agent.set_td_parameter_sampling_enabled(enabled)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        self.set_dynamic_branch_gate_t_env(t_env)
        return super().select_actions(
            ep_batch, t_ep, t_env, bs=bs, test_mode=test_mode
        )

    def _exclude_agent_id_from_trunk(self):
        model_type = getattr(self.args, "clean_model_type", "baseline").replace("-", "_")
        if model_type.startswith("rpg_simple_bias_"):
            return True
        return model_type in {
            "hypermarl_id",
            "hypermarl_fullnet",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            "rpg_dual_branch_relation_hypercond",
            "rpg_dual_branch_td_benefit_drop_hypercond",
            "rpg_dual_branch_parameter_invariant_drop_hypercond",
            "rpg_dual_branch_cstg_gate_hypercond",
            "rpg_dual_branch_bayesg_gate_hypercond",
            "rpg_dual_branch_hard_gate_hypercond",
            "rpg_dual_branch_hard_gate_param_stability_hypercond",
            "rpg_dual_branch_hard_gate_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_param_stability_hypercond",
            "rpg_dual_branch_binary_concrete_grad_consistency_hypercond",
            "rpg_dual_branch_hard_gate_adaptive_param_stability_hypercond",
            "rpg_dual_branch_hard_gate_adaptive_grad_consistency_hypercond",
            "rpg_dual_branch_attention_only_hard_gate_param_stability_hypercond",
            "rpg_dual_branch_attention_only_hard_gate_grad_consistency_hypercond",
            "rpg_dual_branch_split_head_hard_gate_param_stability_hypercond",
            "rpg_dual_branch_split_head_hard_gate_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond",
            "rpg_public_transformer_hypercond",
            "rpg_public_future_delta_token_transformer_hypercond",
            "rpg_public_future_delta_bias_transformer_hypercond",
            "rpg_public_private_token_transformer_hypercond",
            "rpg_public_private_bias_transformer_hypercond",
            "rpg_public_private_bias_transformer_pair_interaction_hypercond",
            "rpg_public_private_bias_transformer_topk_hypercond",
            "rpg_public_private_bias_transformer_threshold_hypercond",
            "rpg_public_past_delta_token_transformer_hypercond",
            "rpg_public_past_delta_bias_transformer_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_enemy_slot_hypercond",
            "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond",
            "rpg_public_private_full_token_transformer_hypercond",
            "rpg_public_past_delta_bias_transformer_private_head_input_hypercond",
            "rpg_global_public_transformer_hypercond",
            "rpg_global_public_private_bias_transformer_hypercond",
            "rpg_global_public_private_bias_transformer_eval_global_hypercond",
            "rpg_global_public_private_bias_transformer_memory_eval_hypercond",
            "rpg_global_public_private_bias_past_delta_token_transformer_hypercond",
            "rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond",
            "rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond",
            "rpg_public_transformer_relation_token_head_hypercond",
            "rpg_public_private_bias_transformer_relation_token_head_hypercond",
            "rpg_public_private_bias_transformer_relation_pair_token_head_hypercond",
            "rpg_public_private_bias_transformer_relation_private_token_head_hypercond",
            "rpg_public_private_bias_transformer_relation_delta_token_head_hypercond",
            "rpg_public_private_bias_transformer_slot_token_head_hypercond",
            "rpg_global_public_transformer_relation_token_head_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_relation_token_head_hypercond",
            "rpg_public_private_full_token_transformer_relation_token_head_hypercond",
            "rpg_full_obs_transformer_hypercond",
            "rpg_full_obs_transformer_relation_token_head_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_relation_token_topk_hypercond",
            "rpg_public_transformer_single_head_hypercond",
            "rpg_public_future_delta_token_transformer_single_head_hypercond",
            "rpg_public_future_delta_bias_transformer_single_head_hypercond",
            "rpg_public_private_token_transformer_single_head_hypercond",
            "rpg_public_private_bias_transformer_single_head_hypercond",
            "rpg_public_past_delta_token_transformer_single_head_hypercond",
            "rpg_public_past_delta_bias_transformer_single_head_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_single_head_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            "rpg_post_topk_enemy_interaction_hypercond",
            "rpg_post_threshold_enemy_interaction_hypercond",
            "rpg_pre_topk_entity_relation_hypercond",
            "rpg_pre_threshold_entity_relation_hypercond",
            "rpg_no_enemy_token_interaction_hypercond",
            "rpg_private_enemy_token_interaction_hypercond",
            "rpg_delta_enemy_token_interaction_hypercond",
            "rpg_entity_token_decision_head_hypercond",
            "rpg_self_enemy_pair_token_decision_head_hypercond",
            "rpg_relation_token_decision_head_hypercond",
            "rpg_policy_relation_fusion_head_hypercond",
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_public_hyper_private_input_single_head",
            "rpg_private_hyper_public_input_single_head",
            "rpg_delta_relation_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id and not self._exclude_agent_id_from_trunk():
            input_shape += self.n_agents
        return input_shape

    def _build_inputs(self, batch, t):
        batch_size = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id and not self._exclude_agent_id_from_trunk():
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch_size, -1, -1))
        return th.cat([x.reshape(batch_size, self.n_agents, -1) for x in inputs], dim=-1)

    def _build_model_context(self, batch, t):
        batch_size = batch.batch_size
        prev_action = batch["actions_onehot"][:, t - 1] if t > 0 else batch["actions_onehot"][:, t].new_zeros(
            batch["actions_onehot"][:, t].shape
        )
        prev_obs = batch["obs"][:, t - 1] if t > 0 else batch["obs"][:, t].new_zeros(
            batch["obs"][:, t].shape
        )
        next_obs = batch["obs"][:, t + 1] if t < batch.max_seq_length - 1 else batch["obs"][:, t].new_zeros(
            batch["obs"][:, t].shape
        )
        next_obs_mask = batch["filled"][:, t + 1] if t < batch.max_seq_length - 1 else batch["filled"][:, t].new_zeros(
            batch["filled"][:, t].shape
        )
        prev_state = batch["state"][:, t - 1] if t > 0 else batch["state"][:, t].new_zeros(
            batch["state"][:, t].shape
        )
        action_targets = action_target_mask = None
        if t < batch.max_seq_length - 1:
            action_targets = batch["actions"][:, t].reshape(batch_size, self.n_agents)
            action_target_mask = batch["filled"][:, t].reshape(batch_size, 1).expand(-1, self.n_agents)
        return {
            "obs": batch["obs"][:, t].reshape(batch_size, self.n_agents, -1),
            "prev_obs": prev_obs.reshape(batch_size, self.n_agents, -1),
            "next_obs": next_obs.reshape(batch_size, self.n_agents, -1),
            "next_obs_mask": next_obs_mask.reshape(batch_size, 1).expand(-1, self.n_agents),
            "prev_action": prev_action.reshape(batch_size, self.n_agents, -1),
            "action_targets": action_targets,
            "action_target_mask": action_target_mask,
            "state": batch["state"][:, t],
            "prev_state": prev_state,
        }

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if test_mode:
            self.agent.eval()
        model_context = self._build_model_context(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, context=model_context, test_mode=test_mode
        )
        self.execution_scope = getattr(self.agent, "execution_scope", "ctde")
        self.latest_route_logits = getattr(self.agent, "latest_route_logits", None)
        self.latest_route_indices = getattr(self.agent, "latest_route_indices", None)
        self.latest_graph_adj = getattr(self.agent, "latest_graph_adj", None)
        self.latest_graph_nodes = getattr(self.agent, "latest_graph_nodes", None)
        self.latest_condition = getattr(self.agent, "latest_condition", None)
        self.latest_condition_graph = getattr(
            self.agent, "latest_condition_graph", None
        )
        self.latest_generated_parameter_graph = getattr(
            self.agent, "latest_generated_parameter_graph", None
        )
        self.latest_trajectory_parameter_projection = None
        model_type = getattr(self.agent, "model_type", "")
        if model_type.endswith(self.TRAJECTORY_PARAMETER_MODEL_SUFFIX):
            projection_dim = int(
                getattr(self.args, "clean_trajectory_parameter_projection_dim", 64)
            )
            flat_projection = self._fixed_parameter_projection(
                self.latest_generated_parameter_graph, projection_dim
            )
            if flat_projection is None:
                raise RuntimeError(
                    "Trajectory parameter likelihood requires generated parameters"
                )
            self.latest_trajectory_parameter_projection = flat_projection.view(
                ep_batch.batch_size, self.n_agents, projection_dim
            )
        self.latest_generated_parameter_log_prob = getattr(
            self.agent, "latest_generated_parameter_log_prob", None
        )
        self.latest_aux_loss = getattr(self.agent, "latest_aux_loss", None)
        self.latest_aux_stats = getattr(self.agent, "latest_aux_stats", {})
        self.latest_teacher_q = getattr(self.agent, "latest_teacher_q", None)
        self.latest_generated_interaction_head = getattr(self.agent, "latest_generated_interaction_head", None)
        self.latest_generated_interaction_head_graph = getattr(
            self.agent, "latest_generated_interaction_head_graph", None
        )
        self.latest_relation_ally_attn = getattr(self.agent, "latest_relation_ally_attn", None)
        self.latest_relation_enemy_attn = getattr(self.agent, "latest_relation_enemy_attn", None)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
