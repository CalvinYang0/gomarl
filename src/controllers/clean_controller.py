import torch as th

from .basic_controller import BasicMAC


class CleanMAC(BasicMAC):
    def _exclude_agent_id_from_trunk(self):
        return getattr(self.args, "clean_model_type", "baseline").replace("-", "_") in {
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
            "rpg_public_transformer_hypercond",
            "rpg_public_future_delta_token_transformer_hypercond",
            "rpg_public_future_delta_bias_transformer_hypercond",
            "rpg_public_private_token_transformer_hypercond",
            "rpg_public_private_bias_transformer_hypercond",
            "rpg_public_past_delta_token_transformer_hypercond",
            "rpg_public_past_delta_bias_transformer_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_hypercond",
            "rpg_public_private_bias_past_delta_token_transformer_enemy_slot_hypercond",
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
        self.latest_aux_loss = getattr(self.agent, "latest_aux_loss", None)
        self.latest_aux_stats = getattr(self.agent, "latest_aux_stats", {})
        self.latest_teacher_q = getattr(self.agent, "latest_teacher_q", None)
        self.latest_generated_interaction_head = getattr(self.agent, "latest_generated_interaction_head", None)
        self.latest_relation_ally_attn = getattr(self.agent, "latest_relation_ally_attn", None)
        self.latest_relation_enemy_attn = getattr(self.agent, "latest_relation_enemy_attn", None)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
