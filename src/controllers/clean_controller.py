import math

import torch as th

from .basic_controller import BasicMAC


class CleanMAC(BasicMAC):
    TRAJECTORY_PARAMETER_MODEL_SUFFIX = (
        "dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
    )

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self._test_gate_probability_sum = None
        self._test_gate_probability_count = 0
        self._test_gate_trajectory_rows = []
        self._test_gate_trajectory_active = False
        self._test_gate_trajectory_last_t_ep = None
        self._test_gate_trajectory_max_steps = max(
            1,
            int(getattr(args, "clean_test_gate_trajectory_max_steps", 256)),
        )
        self._test_parameter_pca_enabled = bool(
            getattr(args, "wandb_test_parameter_pca", True)
        )
        self._random_drop_auxiliary_input_keep_probability = None
        self._random_drop_auxiliary_input_mask = None

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

    def set_dynamic_branch_gate_override(self, gates):
        if hasattr(self.agent, "set_dynamic_branch_gate_override"):
            self.agent.set_dynamic_branch_gate_override(gates)

    def set_dynamic_branch_gate_random_aux_mask(self, mask):
        if hasattr(self.agent, "set_dynamic_branch_gate_random_aux_mask"):
            self.agent.set_dynamic_branch_gate_random_aux_mask(mask)

    def set_dynamic_branch_gate_random_aux_combine_mode(self, mode):
        if hasattr(
            self.agent, "set_dynamic_branch_gate_random_aux_combine_mode"
        ):
            self.agent.set_dynamic_branch_gate_random_aux_combine_mode(mode)

    def set_random_drop_auxiliary_input_keep_probability(self, probability):
        """Enable one cached Bernoulli mask over the raw observation.

        Calling this method again invalidates the cache, so the learner can
        choose episode-level or timestep-level resampling without coupling the
        single-branch controls to the dual-branch gate implementation.
        """
        if probability is None:
            self._random_drop_auxiliary_input_keep_probability = None
            self._random_drop_auxiliary_input_mask = None
            return
        if th.is_tensor(probability):
            probability = float(probability.detach().item())
        probability = float(probability)
        if not 0.0 < probability <= 1.0:
            raise ValueError("Random auxiliary input keep probability must be in (0, 1]")
        self._random_drop_auxiliary_input_keep_probability = probability
        self._random_drop_auxiliary_input_mask = None

    def _random_drop_auxiliary_observation(self, observation):
        probability = self._random_drop_auxiliary_input_keep_probability
        if probability is None:
            return observation
        mask = self._random_drop_auxiliary_input_mask
        if (
            mask is None
            or mask.shape != observation.shape
            or mask.device != observation.device
            or mask.dtype != observation.dtype
        ):
            mask = th.empty_like(observation).bernoulli_(probability)
            self._random_drop_auxiliary_input_mask = mask
        return observation * mask

    def set_td_parameter_sampling_enabled(self, enabled):
        if hasattr(self.agent, "set_td_parameter_sampling_enabled"):
            self.agent.set_td_parameter_sampling_enabled(enabled)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        self.set_dynamic_branch_gate_t_env(t_env)
        selected_actions = super().select_actions(
            ep_batch, t_ep, t_env, bs=bs, test_mode=test_mode
        )
        if test_mode:
            if bool(
                getattr(self.args, "clean_print_test_slot_probabilities", True)
            ):
                self._accumulate_test_gate_probabilities(
                    bs, ep_batch.batch_size
                )
            self._record_test_gate_probability_trajectory(
                bs, ep_batch.batch_size, t_ep
            )
        return selected_actions

    def reset_test_gate_probability_trajectory(self):
        """Start capturing one deterministic test trajectory."""
        if self._test_gate_trajectory_rows:
            # Preserve the first trajectory until the runner logs it. This is
            # important for EpisodeRunner, which executes several test
            # episodes before emitting one aggregate test statistic.
            return
        self._test_gate_trajectory_rows = []
        self._test_gate_trajectory_active = True
        self._test_gate_trajectory_last_t_ep = None

    def finalize_test_gate_probability_trajectory(self):
        self._test_gate_trajectory_active = False

    def _record_test_gate_probability_trajectory(
        self, batch_selection, batch_size, t_ep
    ):
        if not self._test_gate_trajectory_active:
            return
        timestep = int(t_ep)
        # A reset of t_ep marks the next episode. Keep only the first one.
        if (
            self._test_gate_trajectory_last_t_ep is not None
            and timestep <= self._test_gate_trajectory_last_t_ep
        ):
            self._test_gate_trajectory_active = False
            return
        probabilities = getattr(
            self, "latest_dynamic_branch_probabilities_graph", None
        )
        capturer = getattr(self.agent, "rpg_relation_capturer", None)
        if probabilities is None and getattr(capturer, "counter_transformer_profile", None):
            # No-gate baseline still has generated parameters. Do not gate its
            # PCA capture on the existence of a learned-mask network.
            probabilities = th.ones(
                2, int(batch_size), int(self.n_agents), capturer.expected_obs_dim
            )
        if probabilities is None or probabilities.dim() < 3:
            return
        probabilities = probabilities.detach()
        slot_count = probabilities.size(-1)
        if probabilities.dim() == 3:
            expected = int(batch_size) * int(self.n_agents)
            if probabilities.size(1) != expected:
                return
            probabilities = probabilities.reshape(
                2, int(batch_size), int(self.n_agents), slot_count
            )
        elif probabilities.dim() != 4:
            return

        if isinstance(batch_selection, slice):
            selected_indices = list(range(int(batch_size)))[batch_selection]
        elif th.is_tensor(batch_selection):
            selected_indices = batch_selection.detach().cpu().reshape(-1).tolist()
        else:
            selected_indices = list(batch_selection)
        selected_indices = [int(index) for index in selected_indices]
        if not selected_indices:
            return
        # ParallelRunner removes finished environments from ``bs``.  Once
        # environment zero disappears, stop rather than silently switching
        # the plot to another environment's trajectory.
        if self._test_gate_trajectory_rows and 0 not in selected_indices:
            self._test_gate_trajectory_active = False
            return
        # In a parallel test run, use environment zero when it is still
        # active; otherwise use the first active environment as a fallback.
        if probabilities.size(1) == len(selected_indices):
            env_index = (
                selected_indices.index(0) if 0 in selected_indices else 0
            )
        else:
            env_index = 0 if 0 in selected_indices else selected_indices[0]
        if env_index >= probabilities.size(1):
            return
        values = probabilities[:, env_index].mean(dim=1).cpu().tolist()
        if len(values) != 2 or any(len(branch) != slot_count for branch in values):
            return
        parameter_vector = self._test_generated_parameter_vector(
            selected_indices,
            env_index,
            batch_size,
            probabilities.size(1),
        )
        agent_probabilities = {
            "linear": probabilities[0, env_index].cpu().tolist(),
            "attention": probabilities[1, env_index].cpu().tolist(),
        }
        auxiliary_probability = getattr(capturer, "latest_kl80_auxiliary_probability", None)
        if auxiliary_probability is not None:
            auxiliary_name = (
                "auxiliary_fixed80_attention"
                if getattr(capturer, "counter_transformer_profile", {}).get("aux") == "fixed_concrete"
                else "auxiliary_{}_attention".format(capturer.kl_auxiliary_tag)
            )
            agent_probabilities[auxiliary_name] = auxiliary_probability[1, env_index].cpu().tolist()
        self._test_gate_trajectory_rows.append(
            (timestep, values[0], values[1], parameter_vector, agent_probabilities)
        )
        self._test_gate_trajectory_last_t_ep = timestep
        if len(self._test_gate_trajectory_rows) >= self._test_gate_trajectory_max_steps:
            self._test_gate_trajectory_active = False

    def pop_test_gate_probability_trajectory(self):
        if not self._test_gate_trajectory_rows:
            return None
        relation_capturer = getattr(self.agent, "rpg_relation_capturer", None)
        slot_names = list(getattr(relation_capturer, "semantic_names", ()))
        slot_count = len(self._test_gate_trajectory_rows[0][1])
        if len(slot_names) != slot_count:
            slot_names = ["slot_{}".format(index) for index in range(slot_count)]
        trajectory = {
            "timesteps": [row[0] for row in self._test_gate_trajectory_rows],
            "slot_names": slot_names,
            "threshold": float(
                getattr(
                    self.agent,
                    "hard_gate_threshold",
                    getattr(self.args, "clean_hard_gate_threshold", 0.5),
                )
            ),
            "branches": {
                "linear": [row[1] for row in self._test_gate_trajectory_rows],
                "attention": [
                    row[2] for row in self._test_gate_trajectory_rows
                ],
            },
        }
        single_attention = getattr(relation_capturer, "relation_encoder_style", None) == "attention_only"
        if single_attention:
            trajectory["branches"].pop("linear")
        if len(self._test_gate_trajectory_rows[0]) > 4:
            branch_names = self._test_gate_trajectory_rows[0][4].keys()
            trajectory["agent_probability_branches"] = {
                name: [row[4][name] for row in self._test_gate_trajectory_rows]
                for name in branch_names if not (single_attention and name == "linear")
            }
        profile = getattr(relation_capturer, "counter_transformer_profile", {})
        trajectory["gate_note"] = (
            "No gate: all slots kept" if profile.get("label") == "baseline"
            else "Test mask bypassed: all slots kept; plotted values are learned probabilities"
            if profile.get("test_open") else "Learned keep probabilities"
        )
        parameter_vectors = [row[3] for row in self._test_gate_trajectory_rows]
        if parameter_vectors and all(
            vector is not None for vector in parameter_vectors
        ):
            parameter_size = parameter_vectors[0].numel()
            if all(vector.numel() == parameter_size for vector in parameter_vectors):
                trajectory["generated_parameter_vectors"] = th.stack(
                    parameter_vectors, dim=0
                )
        self._test_gate_trajectory_rows = []
        self._test_gate_trajectory_active = False
        self._test_gate_trajectory_last_t_ep = None
        return trajectory

    def _test_generated_parameter_vector(
        self,
        selected_indices,
        selected_env_index,
        full_batch_size,
        active_batch_size,
    ):
        """Return one exact generated-head vector for the plotted environment.

        All generated parameter blocks and all agents are concatenated in a
        fixed order.  Keeping the exact vector here lets the logger fit a 2-D
        PCA to the already collected deterministic test trajectory without an
        additional environment rollout or model forward.
        """
        if not self._test_parameter_pca_enabled:
            return None
        parameter_parts = getattr(
            self, "latest_generated_parameter_graph", None
        )
        if not parameter_parts:
            return None

        vectors = []
        for parameter in parameter_parts:
            if parameter is None or parameter.dim() < 1:
                return None
            detached = parameter.detach()
            leading_size = detached.size(0)
            if leading_size == int(active_batch_size) * int(self.n_agents):
                environment_parameters = detached.reshape(
                    int(active_batch_size), int(self.n_agents), -1
                )[int(selected_env_index)]
            elif leading_size == int(full_batch_size) * int(self.n_agents):
                actual_environment = (
                    0
                    if 0 in selected_indices
                    else int(selected_indices[int(selected_env_index)])
                )
                environment_parameters = detached.reshape(
                    int(full_batch_size), int(self.n_agents), -1
                )[actual_environment]
            else:
                return None
            vectors.append(environment_parameters.reshape(-1).to("cpu", th.float32))
        return th.cat(vectors, dim=0) if vectors else None

    def _accumulate_test_gate_probabilities(
        self, batch_selection, batch_size
    ):
        probabilities = getattr(
            self, "latest_dynamic_branch_probabilities_graph", None
        )
        if probabilities is None or probabilities.dim() < 3:
            return
        probabilities = probabilities.detach()
        slot_count = probabilities.size(-1)
        if probabilities.dim() == 3:
            expected = int(batch_size) * int(self.n_agents)
            if probabilities.size(1) != expected:
                raise RuntimeError(
                    "Dynamic gate probability batch-agent dimension is {}; expected {}"
                    .format(probabilities.size(1), expected)
                )
            probabilities = probabilities.reshape(
                2, int(batch_size), int(self.n_agents), slot_count
            )
        elif probabilities.dim() != 4:
            raise RuntimeError(
                "Dynamic gate probabilities have unsupported shape {}".format(
                    tuple(probabilities.shape)
                )
            )
        selected = probabilities[:, batch_selection]
        flattened = selected.reshape(2, -1, slot_count)
        probability_sum = flattened.double().sum(dim=1).cpu()
        probability_count = int(flattened.size(1))
        if probability_count <= 0:
            return
        if self._test_gate_probability_sum is None:
            self._test_gate_probability_sum = probability_sum
        else:
            if self._test_gate_probability_sum.shape != probability_sum.shape:
                raise RuntimeError(
                    "Dynamic gate test probability shape changed from {} to {}".format(
                        tuple(self._test_gate_probability_sum.shape),
                        tuple(probability_sum.shape),
                    )
                )
            self._test_gate_probability_sum += probability_sum
        self._test_gate_probability_count += probability_count

    def pop_test_gate_probability_summary(self):
        if (
            self._test_gate_probability_sum is None
            or self._test_gate_probability_count <= 0
        ):
            return None
        probability_mean = (
            self._test_gate_probability_sum
            / float(self._test_gate_probability_count)
        )
        relation_capturer = getattr(
            self.agent, "rpg_relation_capturer", None
        )
        slot_names = list(
            getattr(relation_capturer, "semantic_names", ())
        )
        if len(slot_names) != probability_mean.size(-1):
            slot_names = [
                "slot_{}".format(index)
                for index in range(probability_mean.size(-1))
            ]
        summary = {
            "slot_names": slot_names,
            "linear": probability_mean[0].tolist(),
            "attention": probability_mean[1].tolist(),
            "sample_count": self._test_gate_probability_count,
        }
        self._test_gate_probability_sum = None
        self._test_gate_probability_count = 0
        return summary

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
            "rpg_public_transformer_random_drop_aux_hypercond",
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
        # Random-drop auxiliaries perturb only the observation-conditioned
        # relation/hypernetwork path. The recurrent policy trunk deliberately
        # keeps the clean observation, matching the original dual-branch
        # random-gate auxiliary.
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
        observation = self._random_drop_auxiliary_observation(batch["obs"][:, t])
        return {
            "obs": observation.reshape(batch_size, self.n_agents, -1),
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
        self.latest_policy_hidden_graph = getattr(
            self.agent, "latest_policy_hidden_graph", None
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
        self.latest_dynamic_branch_gates_graph = getattr(
            self.agent, "latest_dynamic_branch_gates_graph", None
        )
        self.latest_dynamic_branch_probabilities_graph = getattr(
            self.agent, "latest_dynamic_branch_probabilities_graph", None
        )
        self.latest_dynamic_branch_logits_graph = getattr(
            self.agent, "latest_dynamic_branch_logits_graph", None
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

    def generated_parameters_with_observation_perturbation(
        self,
        ep_batch,
        t,
        hidden_state_before,
        gate_override,
        relative_std,
    ):
        """Re-run one state with perturbed condition obs and identical gates.

        The policy-GRU input and its incoming hidden state are unchanged. This
        isolates the auxiliary signal to the observation-conditioned gate /
        condition / hypernetwork path instead of measuring recurrent-state
        noise. The method never advances ``self.hidden_states``.
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        model_context = self._build_model_context(ep_batch, t)
        observation = model_context["obs"]
        feature_rms = observation.detach().pow(2).mean(
            dim=(0, 1), keepdim=True
        ).sqrt().clamp(min=1e-3)
        perturbation = th.randn_like(observation) * feature_rms * float(
            relative_std
        )
        model_context["obs"] = observation + perturbation
        self.set_dynamic_branch_gate_override(gate_override)
        try:
            self.agent(
                agent_inputs,
                hidden_state_before,
                context=model_context,
                test_mode=False,
            )
            parameter_graph = getattr(
                self.agent, "latest_generated_parameter_graph", None
            )
        finally:
            self.set_dynamic_branch_gate_override(None)
        return parameter_graph
