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
from modules.agents.counter_transformer_suite import profile_for
from utils.rl_utils import build_td_lambda_targets


class CleanLearner:
    def __init__(self, mac, scheme, logger, args):
        del scheme
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        if hasattr(self.target_mac.agent, "set_dynamic_branch_gate_target_mode"):
            self.target_mac.agent.set_dynamic_branch_gate_target_mode(True)
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
        self.semantic_critical_tail_fraction = float(
            getattr(args, "clean_semantic_critical_tail_fraction", 0.1)
        )
        if not 0.0 < self.semantic_critical_tail_fraction <= 1.0:
            raise ValueError(
                "clean_semantic_critical_tail_fraction must be in (0, 1]"
            )
        self.semantic_critical_tail_weight = float(
            getattr(args, "clean_semantic_critical_tail_weight", 0.5)
        )
        if not 0.0 <= self.semantic_critical_tail_weight <= 1.0:
            raise ValueError(
                "clean_semantic_critical_tail_weight must be in [0, 1]"
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
        self.last_branch_drop_version_logged = -1
        self.condition_gradient_consistency_active = (
            getattr(self.mac.agent, "model_type", "")
            in {
                "grf_abs_dual_branch_hard_gate_grad_consistency_hypercond",
                "rpg_dual_branch_hard_gate_grad_consistency_hypercond",
                "grf_abs_dual_branch_binary_concrete_grad_consistency_hypercond",
                "rpg_dual_branch_binary_concrete_grad_consistency_hypercond",
                "grf_abs_dual_branch_hard_gate_adaptive_grad_consistency_hypercond",
                "rpg_dual_branch_hard_gate_adaptive_grad_consistency_hypercond",
                "rpg_dual_branch_attention_only_hard_gate_grad_consistency_hypercond",
                "rpg_dual_branch_split_head_hard_gate_grad_consistency_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond",
            }
        )
        self.condition_gradient_consistency_coef = float(
            getattr(args, "clean_condition_gradient_consistency_coef", 0.1)
        )
        self.condition_gradient_consistency_pairs = max(
            1,
            int(getattr(args, "clean_condition_gradient_consistency_pairs", 2)),
        )
        self.condition_gradient_consistency_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_condition_gradient_consistency_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        model_type = getattr(self.mac.agent, "model_type", "")
        self.counter_transformer_profile = profile_for(model_type)
        self.generated_parameter_stability_active = (
            model_type
            in {
                "grf_abs_dual_branch_hard_gate_param_stability_hypercond",
                "rpg_dual_branch_hard_gate_param_stability_hypercond",
                "grf_abs_dual_branch_binary_concrete_param_stability_hypercond",
                "rpg_dual_branch_binary_concrete_param_stability_hypercond",
                "grf_abs_dual_branch_hard_gate_adaptive_param_stability_hypercond",
                "rpg_dual_branch_hard_gate_adaptive_param_stability_hypercond",
                "rpg_dual_branch_attention_only_hard_gate_param_stability_hypercond",
                "rpg_dual_branch_split_head_hard_gate_param_stability_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_param_stability_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_param_stability_hypercond",
                "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_param_stability_hypercond",
                "rpg_dual_branch_binary_concrete_adaptive_attention_only_param_stability_hypercond",
            }
        )
        self.generated_parameter_stability_coef = float(
            getattr(args, "clean_generated_parameter_stability_coef", 0.1)
        )
        self.generated_parameter_stability_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_generated_parameter_stability_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        self.generated_parameter_likelihood_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_adaptive_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_parameter_likelihood_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_attention_only_parameter_likelihood_hypercond",
        }
        self.generated_parameter_likelihood_std = float(
            getattr(args, "clean_generated_parameter_likelihood_std", 1.0)
        )
        if self.generated_parameter_likelihood_std <= 0.0:
            raise ValueError("clean_generated_parameter_likelihood_std must be positive")
        self.generated_parameter_likelihood_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_generated_parameter_likelihood_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        self.td_weighted_parameter_likelihood_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_adaptive_td_weighted_param_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_td_weighted_param_likelihood_hypercond",
        }
        self.td_weighted_parameter_likelihood_coef = float(
            getattr(
                args,
                "clean_td_weighted_parameter_likelihood_coef",
                0.01,
            )
        )
        self.td_weighted_parameter_likelihood_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_td_weighted_parameter_likelihood_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        if self.td_weighted_parameter_likelihood_coef < 0.0:
            raise ValueError(
                "clean_td_weighted_parameter_likelihood_coef must be non-negative"
            )
        self.trajectory_parameter_likelihood_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond",
        }
        self.trajectory_parameter_likelihood_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_trajectory_parameter_likelihood_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        self.gate_regularization_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_bayesg_kl20_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_threshold70_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl70_hypercond",
            "grf_abs_dual_branch_binary_concrete_bimodal_budget80_hypercond",
            "grf_abs_dual_branch_hard_concrete_l0_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl90_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_threshold70_relation_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_keep_relation_hypercond",
        }
        self.gate_regularization_active |= bool(self.counter_transformer_profile.get("kl"))
        self.perturbed_parameter_importance_active = model_type == (
            "grf_abs_dual_branch_binary_concrete_"
            "perturb_param_importance_hypercond"
        )
        self.gradient_importance_active = model_type == (
            "grf_abs_dual_branch_binary_concrete_gradient_importance_hypercond"
        )
        self.perturbed_head_td_quality_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond",
            "rpg_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond",
            "grf_abs_dual_branch_binary_concrete_mask_parameter_relation_perturbed_head_hypercond",
        }
        self.temporal_param_stability_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_temporal_param_stability_hypercond",
            "rpg_dual_branch_binary_concrete_temporal_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_grouped_property_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_param_stability_freeze2m_hypercond",
            "grf_abs_dual_branch_binary_concrete_mask_parameter_relation_temporal_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_group_gate_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_group_distance_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_stop_param_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_stop_mask_hypercond",
            "grf_abs_dual_branch_binary_concrete_random_drop_aux_hypercond",
        }
        self.mask_parameter_relation_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_mask_parameter_relation_hypercond",
            "grf_abs_dual_branch_binary_concrete_mask_parameter_relation_temporal_stability_hypercond",
            "grf_abs_dual_branch_hard_gate_mask_parameter_relation_hypercond",
            "grf_abs_dual_branch_binary_concrete_mask_parameter_relation_perturbed_head_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_threshold70_relation_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_keep_relation_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_group_gate_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_group_distance_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_stop_param_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_relation_stop_mask_hypercond",
        }
        self.mask_parameter_relation_active |= bool(self.counter_transformer_profile.get("relation"))
        self.temporal_param_stability_active |= bool(self.counter_transformer_profile.get("temporal"))
        self.mask_parameter_relation_group_distance = model_type == (
            "grf_abs_dual_branch_binary_concrete_"
            "temporal_relation_group_distance_hypercond"
        )
        self.mask_parameter_relation_stop_side = (
            "mask"
            if model_type
            == "grf_abs_dual_branch_binary_concrete_"
            "temporal_relation_stop_mask_hypercond"
            else "parameter"
        )
        self.temporal_param_small_change_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_temporal_param_small_change_hypercond",
            "rpg_dual_branch_binary_concrete_temporal_param_small_change_hypercond",
        }
        self.temporal_param_auxiliary_active = (
            self.temporal_param_stability_active
            or self.temporal_param_small_change_active
        )
        self.random_drop_auxiliary_active = model_type in {
            "grf_abs_dual_branch_binary_concrete_random_drop_aux_hypercond",
            "rpg_dual_branch_binary_concrete_random_drop_aux_hypercond",
            "grf_abs_mlp_relation_random_drop_aux_hypercond",
            "grf_abs_single_transformer_branch_random_drop_aux_hypercond",
            "grf_abs_single_transformer_branch_binary_concrete_gate_random_drop_aux_hypercond",
            "grf_abs_single_linear_branch_binary_concrete_gate_random_drop_aux_hypercond",
            "rpg_public_transformer_random_drop_aux_hypercond",
        }
        self.random_drop_auxiliary_active |= bool(self.counter_transformer_profile.get("aux"))
        self.kl80_random_drop_auxiliary = self.counter_transformer_profile.get("aux") == "kl80"
        self.concrete_random_drop_auxiliary = self.counter_transformer_profile.get("aux") in {"kl80", "fixed_concrete"}
        self.random_drop_auxiliary_input_mask = model_type in {
            "grf_abs_mlp_relation_random_drop_aux_hypercond",
            "grf_abs_single_transformer_branch_random_drop_aux_hypercond",
            "rpg_public_transformer_random_drop_aux_hypercond",
        }
        self.random_drop_auxiliary_keep_probability = float(
            getattr(args, "clean_random_drop_auxiliary_keep_probability", 0.8)
        )
        self.random_drop_auxiliary_coef = float(
            getattr(args, "clean_random_drop_auxiliary_coef", 0.5)
        )
        self.random_drop_auxiliary_scope = str(
            getattr(args, "clean_random_drop_auxiliary_scope", "episode")
        ).lower()
        self.random_drop_auxiliary_combine_mode = str(
            getattr(
                args,
                "clean_random_drop_auxiliary_combine_mode",
                "replace",
            )
        ).lower()
        if not 0.0 < self.random_drop_auxiliary_keep_probability <= 1.0:
            raise ValueError(
                "clean_random_drop_auxiliary_keep_probability must be in (0, 1]"
            )
        if self.random_drop_auxiliary_coef < 0.0:
            raise ValueError("clean_random_drop_auxiliary_coef must be non-negative")
        if self.random_drop_auxiliary_scope not in {"episode", "timestep"}:
            raise ValueError(
                "clean_random_drop_auxiliary_scope must be episode or timestep"
            )
        if self.random_drop_auxiliary_combine_mode not in {
            "replace",
            "multiply",
        }:
            raise ValueError(
                "clean_random_drop_auxiliary_combine_mode must be replace or multiply"
            )
        self.importance_auxiliary_active = (
            self.perturbed_parameter_importance_active
            or self.gradient_importance_active
            or self.perturbed_head_td_quality_active
            or self.temporal_param_auxiliary_active
            or self.mask_parameter_relation_active
        )
        self.importance_gate_parameters = ()
        if self.importance_auxiliary_active:
            relation_capturer = getattr(
                self.mac.agent, "rpg_relation_capturer", None
            )
            dynamic_gate = getattr(
                relation_capturer, "dynamic_branch_gate", None
            )
            if dynamic_gate is None:
                raise RuntimeError(
                    "Importance auxiliary variants require a dynamic branch gate"
                )
            self.importance_gate_parameters = tuple(
                parameter
                for parameter in dynamic_gate.parameters()
                if parameter.requires_grad
            )
            if not self.importance_gate_parameters:
                raise RuntimeError(
                    "Importance auxiliary variants require trainable gate parameters"
                )
        gate_parameter_ids = {
            id(parameter) for parameter in self.importance_gate_parameters
        }
        self.importance_non_gate_parameters = tuple(
            parameter
            for parameter in self.params
            if id(parameter) not in gate_parameter_ids
        )
        self.mask_parameter_relation_group_ids = None
        if self.mask_parameter_relation_group_distance:
            relation_capturer = getattr(
                self.mac.agent, "rpg_relation_capturer", None
            )
            semantic_names = getattr(relation_capturer, "semantic_names", None)
            if semantic_names is None:
                raise RuntimeError(
                    "Group-distance relation requires GRF semantic slot names"
                )
            self.mask_parameter_relation_group_ids = (
                self._property_group_ids(semantic_names)
            )
        self.importance_auxiliary_warmup_steps = max(
            0,
            int(
                getattr(
                    args,
                    "clean_importance_auxiliary_warmup_steps",
                    getattr(args, "clean_dynamic_branch_gate_warmup_steps", 250000),
                )
            ),
        )
        self.parameter_perturbation_relative_std = float(
            getattr(args, "clean_parameter_perturbation_relative_std", 0.05)
        )
        self.parameter_perturbation_probe_timesteps = max(
            1,
            int(getattr(args, "clean_parameter_perturbation_probe_timesteps", 2)),
        )
        if self.parameter_perturbation_relative_std <= 0.0:
            raise ValueError(
                "clean_parameter_perturbation_relative_std must be positive"
            )
        self.perturbed_head_relative_std = float(
            getattr(args, "clean_perturbed_head_relative_std", 0.05)
        )
        self.perturbed_head_minimum_rms = float(
            getattr(args, "clean_perturbed_head_minimum_rms", 1e-3)
        )
        if self.perturbed_head_relative_std <= 0.0:
            raise ValueError("clean_perturbed_head_relative_std must be positive")
        if self.perturbed_head_minimum_rms <= 0.0:
            raise ValueError("clean_perturbed_head_minimum_rms must be positive")
        self.temporal_param_switch_margin = float(
            getattr(args, "clean_temporal_param_switch_margin", 0.1)
        )
        self.temporal_param_scale_eps = float(
            getattr(args, "clean_temporal_param_scale_eps", 1e-6)
        )
        self.mask_parameter_relation_scale = float(
            getattr(args, "clean_mask_parameter_relation_scale", 0.1)
        )
        self.mask_parameter_relation_pairing = str(
            getattr(args, "clean_mask_parameter_relation_pairing", "fixed")
        ).strip().lower()
        self.mask_parameter_relation_mask_source = str(
            getattr(
                args,
                "clean_mask_parameter_relation_mask_source",
                "probability",
            )
        ).strip().lower()
        if self.mask_parameter_relation_pairing not in {
            "fixed",
            "episode_random",
            "global_random",
        }:
            raise ValueError(
                "clean_mask_parameter_relation_pairing must be one of "
                "fixed, episode_random, or global_random"
            )
        if self.mask_parameter_relation_mask_source not in {
            "probability",
            "sampled_gate",
        }:
            raise ValueError(
                "clean_mask_parameter_relation_mask_source must be "
                "probability or sampled_gate"
            )
        self.mask_parameter_relation_coef = float(
            getattr(args, "clean_mask_parameter_relation_coef", 10.0)
        )
        self.mask_parameter_relation_temporal_coef = float(
            getattr(args, "clean_mask_parameter_relation_temporal_coef", 50.0)
        )
        self.mask_parameter_relation_perturbed_head_coef = float(
            getattr(args, "clean_mask_parameter_relation_perturbed_head_coef", 1.0)
        )
        self.mask_parameter_relation_gate_regularization_coef = float(
            getattr(
                args,
                "clean_mask_parameter_relation_gate_regularization_coef",
                1.0,
            )
        )
        if self.mask_parameter_relation_scale <= 0.0:
            raise ValueError("clean_mask_parameter_relation_scale must be positive")
        for coefficient_name in (
            "mask_parameter_relation_coef",
            "mask_parameter_relation_temporal_coef",
            "mask_parameter_relation_perturbed_head_coef",
            "mask_parameter_relation_gate_regularization_coef",
        ):
            if getattr(self, coefficient_name) < 0.0:
                raise ValueError("{} must be non-negative".format(coefficient_name))
        if self.temporal_param_switch_margin <= 0.0:
            raise ValueError("clean_temporal_param_switch_margin must be positive")
        if self.temporal_param_scale_eps <= 0.0:
            raise ValueError("clean_temporal_param_scale_eps must be positive")
        self.importance_alternating_training = bool(
            getattr(args, "clean_importance_alternating_training", False)
        )
        self.importance_non_gate_phase_steps = max(
            1,
            int(getattr(args, "clean_importance_non_gate_phase_steps", 80000)),
        )
        self.importance_gate_phase_steps = max(
            1,
            int(getattr(args, "clean_importance_gate_phase_steps", 20000)),
        )
        if self.importance_alternating_training and not self.importance_auxiliary_active:
            raise ValueError(
                "Alternating importance training requires an importance variant"
            )
        self.last_importance_training_phase = None
        self.adaptive_auxiliary_ratio_active = model_type in {
            "grf_abs_dual_branch_hard_gate_adaptive_grad_consistency_hypercond",
            "rpg_dual_branch_hard_gate_adaptive_grad_consistency_hypercond",
            "grf_abs_dual_branch_hard_gate_adaptive_param_stability_hypercond",
            "rpg_dual_branch_hard_gate_adaptive_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_param_stability_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_param_stability_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_attention_only_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_parameter_likelihood_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_attention_only_parameter_likelihood_hypercond",
            "grf_abs_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond",
            "rpg_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl20_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl80_threshold70_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl70_hypercond",
            "grf_abs_dual_branch_binary_concrete_bimodal_budget80_hypercond",
            "grf_abs_dual_branch_hard_concrete_l0_hypercond",
            "grf_abs_dual_branch_binary_concrete_perturb_param_importance_hypercond",
            "grf_abs_dual_branch_binary_concrete_gradient_importance_hypercond",
            "grf_abs_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_param_small_change_hypercond",
            "grf_abs_dual_branch_binary_concrete_grouped_property_param_stability_hypercond",
            "grf_abs_dual_branch_binary_concrete_temporal_param_stability_freeze2m_hypercond",
            "grf_abs_dual_branch_binary_concrete_bayesg_kl90_hypercond",
            "rpg_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond",
            "rpg_dual_branch_binary_concrete_temporal_param_stability_hypercond",
            "rpg_dual_branch_binary_concrete_temporal_param_small_change_hypercond",
        }
        self.adaptive_auxiliary_ratio_active |= bool(self.counter_transformer_profile.get("kl"))
        self.adaptive_auxiliary_target_ratio = float(
            getattr(args, "clean_adaptive_auxiliary_target_ratio", 0.1)
        )
        self.adaptive_auxiliary_ema_decay = float(
            getattr(args, "clean_adaptive_auxiliary_ema_decay", 0.99)
        )
        self.adaptive_auxiliary_eps = float(
            getattr(args, "clean_adaptive_auxiliary_eps", 1e-8)
        )
        self.adaptive_auxiliary_max_coef = float(
            getattr(args, "clean_adaptive_auxiliary_max_coef", 100.0)
        )
        if not 0.0 <= self.adaptive_auxiliary_target_ratio:
            raise ValueError(
                "clean_adaptive_auxiliary_target_ratio must be non-negative"
            )
        if not 0.0 <= self.adaptive_auxiliary_ema_decay < 1.0:
            raise ValueError("clean_adaptive_auxiliary_ema_decay must be in [0, 1)")
        if not 0.0 < self.adaptive_auxiliary_eps:
            raise ValueError("clean_adaptive_auxiliary_eps must be positive")
        if not 0.0 < self.adaptive_auxiliary_max_coef:
            raise ValueError("clean_adaptive_auxiliary_max_coef must be positive")
        self.adaptive_auxiliary_ema_td = None
        self.adaptive_auxiliary_ema_aux = None
        self.latest_adaptive_auxiliary_stats = {}

    def _amp_context(self):
        return th.cuda.amp.autocast(enabled=True) if self.use_amp else nullcontext()

    def _adaptive_auxiliary_coefficient(self, td_loss, auxiliary_loss):
        """Match a detached EMA-scaled auxiliary term to a target TD ratio."""
        with th.no_grad():
            td_value = float(td_loss.detach().float().item())
            auxiliary_value = float(auxiliary_loss.detach().float().item())
            if auxiliary_value <= self.adaptive_auxiliary_eps:
                self.latest_adaptive_auxiliary_stats = {
                    "adaptive_auxiliary_ema_td": (
                        0.0
                        if self.adaptive_auxiliary_ema_td is None
                        else self.adaptive_auxiliary_ema_td
                    ),
                    "adaptive_auxiliary_ema_loss": (
                        0.0
                        if self.adaptive_auxiliary_ema_aux is None
                        else self.adaptive_auxiliary_ema_aux
                    ),
                    "adaptive_auxiliary_effective_coef": 0.0,
                    "adaptive_auxiliary_weighted_loss": 0.0,
                    "adaptive_auxiliary_weighted_to_td_ratio": 0.0,
                    "adaptive_auxiliary_target_ratio": self.adaptive_auxiliary_target_ratio,
                }
                return 0.0
            if self.adaptive_auxiliary_ema_td is None:
                self.adaptive_auxiliary_ema_td = td_value
                self.adaptive_auxiliary_ema_aux = auxiliary_value
            else:
                decay = self.adaptive_auxiliary_ema_decay
                self.adaptive_auxiliary_ema_td = (
                    decay * self.adaptive_auxiliary_ema_td
                    + (1.0 - decay) * td_value
                )
                self.adaptive_auxiliary_ema_aux = (
                    decay * self.adaptive_auxiliary_ema_aux
                    + (1.0 - decay) * auxiliary_value
                )
            coefficient = (
                self.adaptive_auxiliary_target_ratio
                * self.adaptive_auxiliary_ema_td
                / max(self.adaptive_auxiliary_ema_aux, self.adaptive_auxiliary_eps)
            )
            coefficient = min(coefficient, self.adaptive_auxiliary_max_coef)
            weighted_value = coefficient * auxiliary_value
            actual_ratio = weighted_value / max(td_value, self.adaptive_auxiliary_eps)
            self.latest_adaptive_auxiliary_stats = {
                "adaptive_auxiliary_ema_td": self.adaptive_auxiliary_ema_td,
                "adaptive_auxiliary_ema_loss": self.adaptive_auxiliary_ema_aux,
                "adaptive_auxiliary_effective_coef": coefficient,
                "adaptive_auxiliary_weighted_loss": weighted_value,
                "adaptive_auxiliary_weighted_to_td_ratio": actual_ratio,
                "adaptive_auxiliary_target_ratio": self.adaptive_auxiliary_target_ratio,
            }
        return coefficient

    def _backward_main_and_gate_only_auxiliary(
        self, main_loss, gate_only_auxiliary_loss=None
    ):
        """Backpropagate an auxiliary objective only into the dynamic gate.

        The main loss keeps its ordinary gradient path through every trainable
        parameter, including the gate. The separate auxiliary graph is then
        differentiated only with respect to gate parameters, so parameter
        stability cannot flatten the condition encoder or hypernetwork merely
        to reduce its own loss.
        """
        auxiliary_active = (
            gate_only_auxiliary_loss is not None
            and gate_only_auxiliary_loss.requires_grad
        )
        if auxiliary_active and not self.importance_gate_parameters:
            raise RuntimeError(
                "Gate-only auxiliary backward has no trainable gate parameters"
            )

        scaled_main_loss = (
            self.amp_scaler.scale(main_loss) if self.use_amp else main_loss
        )
        scaled_main_loss.backward(retain_graph=auxiliary_active)
        if not auxiliary_active:
            return

        scaled_auxiliary_loss = (
            self.amp_scaler.scale(gate_only_auxiliary_loss)
            if self.use_amp
            else gate_only_auxiliary_loss
        )
        auxiliary_gradients = th.autograd.grad(
            scaled_auxiliary_loss,
            self.importance_gate_parameters,
            allow_unused=True,
        )
        if not any(gradient is not None for gradient in auxiliary_gradients):
            raise RuntimeError(
                "Importance auxiliary loss is disconnected from the dynamic gate"
            )
        for parameter, gradient in zip(
            self.importance_gate_parameters, auxiliary_gradients
        ):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.add_(gradient.detach())

    def _dynamic_gate_training_is_frozen(self, t_env):
        """Return whether the online dynamic gate has reached its freeze step."""
        relation_capturer = getattr(self.mac.agent, "rpg_relation_capturer", None)
        freeze_steps = int(
            getattr(
                relation_capturer,
                "dynamic_branch_gate_training_freeze_steps",
                0,
            )
        )
        return freeze_steps > 0 and int(t_env) >= freeze_steps

    def _backward_parameters_only(self, loss, parameters):
        """Differentiate one loss only with respect to the selected parameters."""
        parameters = tuple(parameters)
        if not parameters:
            raise RuntimeError("Parameter-only backward received no parameters")
        scaled_loss = self.amp_scaler.scale(loss) if self.use_amp else loss
        gradients = th.autograd.grad(
            scaled_loss,
            parameters,
            allow_unused=True,
        )
        if not any(gradient is not None for gradient in gradients):
            raise RuntimeError("Selected parameters are disconnected from the loss")
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            parameter.grad = gradient.detach().clone()

    def _importance_training_phase(self, t_env):
        """Return the current phase of optional environment-step alternation."""
        if not self.importance_alternating_training:
            return "joint"
        if t_env < self.importance_auxiliary_warmup_steps:
            return "non_gate_td"
        cycle_steps = (
            self.importance_non_gate_phase_steps
            + self.importance_gate_phase_steps
        )
        cycle_position = (
            int(t_env) - self.importance_auxiliary_warmup_steps
        ) % cycle_steps
        if cycle_position < self.importance_non_gate_phase_steps:
            return "non_gate_td"
        return "gate_td_aux"

    @staticmethod
    def _semantic_router(mac):
        capturer = getattr(getattr(mac, "agent", None), "rpg_relation_capturer", None)
        if capturer is None or not bool(getattr(capturer, "semantic_router_active", False)):
            return None
        return capturer

    @staticmethod
    def _branch_drop_capturer(mac):
        capturer = getattr(getattr(mac, "agent", None), "rpg_relation_capturer", None)
        if capturer is None or not bool(getattr(capturer, "branch_drop_active", False)):
            return None
        return capturer

    def _sync_branch_drop_to_target(self, capturer):
        target = self._branch_drop_capturer(self.target_mac)
        if capturer is not None and target is not None:
            target.copy_branch_drop_from(capturer)

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

    def _condition_gradient_consistency_loss(
        self, td_error, td_mask, condition_graphs
    ):
        """Compare TD gradients in condition space at sampled adjacent steps."""
        zero = td_error.new_zeros(())
        if len(condition_graphs) < 2 or td_error.shape[1] < 2:
            return zero, {
                "condition_grad_consistency_cosine": zero,
                "condition_grad_consistency_pairs": zero,
            }

        time_valid = td_mask.reshape(
            td_mask.shape[0], td_mask.shape[1], -1
        ).any(dim=-1)
        candidate_times = (
            (time_valid[:, 1:] & time_valid[:, :-1])
            .any(dim=0)
            .nonzero(as_tuple=False)
            .flatten()
            + 1
        )
        if candidate_times.numel() == 0:
            return zero, {
                "condition_grad_consistency_cosine": zero,
                "condition_grad_consistency_pairs": zero,
            }
        if candidate_times.numel() > self.condition_gradient_consistency_pairs:
            order = th.randperm(candidate_times.numel(), device=candidate_times.device)
            candidate_times = candidate_times[
                order[: self.condition_gradient_consistency_pairs]
            ]

        gradient_cache = {}

        def condition_gradient(time_index):
            if time_index not in gradient_cache:
                step_mask = td_mask[:, time_index]
                step_loss = (
                    td_error[:, time_index].pow(2) * step_mask
                ).sum() / step_mask.sum().clamp(min=1.0)
                gradient_cache[time_index] = th.autograd.grad(
                    step_loss,
                    condition_graphs[time_index],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            return gradient_cache[time_index]

        losses = []
        cosine_sums = []
        valid_counts = []
        used_pairs = 0
        for time_index in candidate_times.tolist():
            previous_gradient = condition_gradient(time_index - 1)
            current_gradient = condition_gradient(time_index)
            if previous_gradient is None or current_gradient is None:
                continue

            previous_gradient = previous_gradient.float()
            current_gradient = current_gradient.float()
            cosine = F.cosine_similarity(
                current_gradient, previous_gradient, dim=-1, eps=1e-8
            )
            pair_valid = (
                time_valid[:, time_index]
                & time_valid[:, time_index - 1]
            ).unsqueeze(-1).expand_as(cosine)
            nonzero_gradient = (
                current_gradient.norm(dim=-1) > 1e-8
            ) & (previous_gradient.norm(dim=-1) > 1e-8)
            valid = pair_valid & nonzero_gradient
            if not valid.any():
                continue
            valid_float = valid.to(cosine.dtype)
            valid_count = valid_float.sum()
            losses.append(((1.0 - cosine) * valid_float).sum())
            cosine_sums.append((cosine * valid_float).sum())
            valid_counts.append(valid_count)
            used_pairs += 1

        if not losses:
            return zero, {
                "condition_grad_consistency_cosine": zero,
                "condition_grad_consistency_pairs": zero,
            }
        total_valid = th.stack(valid_counts).sum().clamp(min=1.0)
        consistency_loss = th.stack(losses).sum() / total_valid
        mean_cosine = th.stack(cosine_sums).sum() / total_valid
        return consistency_loss, {
            "condition_grad_consistency_cosine": mean_cosine.detach(),
            "condition_grad_consistency_pairs": zero.new_tensor(float(used_pairs)),
        }

    @staticmethod
    def _generated_parameter_stability_pair(
        previous_parameters,
        current_parameters,
        pair_valid,
        batch_size,
        n_agents,
    ):
        """Exact mean L1 change for one valid adjacent parameter pair."""
        if len(previous_parameters) != len(current_parameters):
            raise RuntimeError("Generated parameter structures do not match.")
        absolute_sum = None
        parameter_count = 0
        for previous, current in zip(previous_parameters, current_parameters):
            if previous.shape != current.shape:
                raise RuntimeError("Adjacent generated parameter shapes do not match.")
            difference = (current - previous).abs().reshape(
                batch_size, n_agents, -1
            )
            part_sum = difference.sum(dim=-1)
            absolute_sum = (
                part_sum if absolute_sum is None else absolute_sum + part_sum
            )
            parameter_count += difference.shape[-1]
        if absolute_sum is None or parameter_count == 0:
            zero = pair_valid.new_zeros((), dtype=th.float32)
            return zero, zero
        per_agent_mean = absolute_sum / float(parameter_count)
        valid = pair_valid.reshape(batch_size, 1).expand(-1, n_agents)
        valid_float = valid.to(per_agent_mean.dtype)
        return (per_agent_mean * valid_float).sum(), valid_float.sum()

    @staticmethod
    def _normalized_generated_parameter_change(
        previous_parameters,
        current_parameters,
        pair_valid,
        batch_size,
        n_agents,
        scale_eps,
    ):
        """Return equal-block relative L1 change for every agent and pair.

        Each generated parameter block is normalized independently so the two
        weight matrices cannot dominate the two bias vectors merely because
        they contain more elements. The scale is detached to prevent the
        auxiliary objective from gaming its denominator.
        """
        if len(previous_parameters) != len(current_parameters):
            raise RuntimeError("Generated parameter structures do not match.")
        relative_changes = []
        for previous, current in zip(previous_parameters, current_parameters):
            if previous.shape != current.shape:
                raise RuntimeError("Adjacent generated parameter shapes do not match.")
            previous_flat = previous.reshape(batch_size, n_agents, -1)
            current_flat = current.reshape(batch_size, n_agents, -1)
            absolute_change = (current_flat - previous_flat).abs().mean(dim=-1)
            previous_rms = previous_flat.detach().pow(2).mean(dim=-1).sqrt()
            current_rms = current_flat.detach().pow(2).mean(dim=-1).sqrt()
            scale = (0.5 * (previous_rms + current_rms)).clamp(
                min=float(scale_eps)
            )
            relative_changes.append(absolute_change / scale)
        if not relative_changes:
            zero = pair_valid.new_zeros(
                (batch_size, n_agents), dtype=th.float32
            )
            return zero, zero
        per_agent_change = th.stack(relative_changes, dim=0).mean(dim=0)
        valid = pair_valid.reshape(batch_size, 1).expand(-1, n_agents)
        return per_agent_change, valid.to(per_agent_change.dtype)

    def _mask_parameter_relation_pair(
        self,
        previous_parameters,
        current_parameters,
        previous_probabilities,
        current_probabilities,
        pair_valid,
        batch_size,
        n_agents,
    ):
        """Match mask distance to generated-parameter distance.

        The default/stop-parameter direction treats parameter distance as a
        detached mode label and organizes q(mask|obs).  The stop-mask ablation
        reverses that relation gradient: mask distance is the detached label
        and generated-parameter distance receives the relation signal.  In
        either case the learner requests auxiliary gradients only for gate
        parameters, so TD remains the sole updater of the main network.
        """
        parameter_change, valid = self._normalized_generated_parameter_change(
            previous_parameters,
            current_parameters,
            pair_valid,
            batch_size,
            n_agents,
            self.temporal_param_scale_eps,
        )
        if previous_probabilities.shape != current_probabilities.shape:
            raise RuntimeError("Dynamic gate probability shapes do not match")
        if previous_probabilities.size(0) != 2:
            raise RuntimeError("Dynamic gate probabilities must start with two branches")
        previous_probabilities = previous_probabilities.reshape(
            2, batch_size, n_agents, -1
        )
        current_probabilities = current_probabilities.reshape(
            2, batch_size, n_agents, -1
        )
        if self.mask_parameter_relation_group_distance:
            previous_probabilities = self._group_mask_probabilities(
                previous_probabilities
            )
            current_probabilities = self._group_mask_probabilities(
                current_probabilities
            )
        # [branch, batch, agent, raw-slot] -> [batch, agent]
        if getattr(self, "counter_transformer_profile", {}).get("relation"):
            # The unused Linear gate must neither dilute the metric nor learn
            # to satisfy the relation loss without affecting the policy.
            previous_probabilities = previous_probabilities[1:2]
            current_probabilities = current_probabilities[1:2]
        mask_distance = (
            current_probabilities - previous_probabilities
        ).abs().mean(dim=(0, -1))
        if self.mask_parameter_relation_stop_side == "mask":
            mask_distance = mask_distance.detach()
            parameter_target = parameter_change / (
                parameter_change.detach() + self.mask_parameter_relation_scale
            )
        else:
            parameter_target = (
                parameter_change.detach()
                / (parameter_change.detach() + self.mask_parameter_relation_scale)
            )
        pair_loss = (mask_distance - parameter_target).abs()
        return (pair_loss * valid).sum(), valid.sum(), mask_distance, parameter_target

    @staticmethod
    def _random_relation_pair_indices(valid_steps, pairing):
        """Return random valid-state pairs for the requested sampling scope.

        A random permutation followed by a one-position cyclic shift makes
        every selected state participate exactly once on each side of a pair.
        This gives full valid-state coverage at O(BT) cost without constructing
        every possible O((BT)^2) pair.
        """
        if valid_steps.dim() != 2:
            raise RuntimeError("Relation validity must have shape [batch, time]")
        if pairing not in {"episode_random", "global_random"}:
            raise ValueError("Random relation pairing mode is invalid")

        device = valid_steps.device
        previous_batch = []
        previous_time = []
        current_batch = []
        current_time = []

        if pairing == "episode_random":
            for batch_index in range(valid_steps.size(0)):
                times = th.nonzero(
                    valid_steps[batch_index], as_tuple=False
                ).flatten()
                if times.numel() < 2:
                    continue
                shuffled = times[th.randperm(times.numel(), device=device)]
                shifted = th.roll(shuffled, shifts=-1, dims=0)
                batch_indices = th.full_like(shuffled, batch_index)
                previous_batch.append(batch_indices)
                previous_time.append(shuffled)
                current_batch.append(batch_indices)
                current_time.append(shifted)
        else:
            coordinates = th.nonzero(valid_steps, as_tuple=False)
            if coordinates.size(0) >= 2:
                shuffled = coordinates[
                    th.randperm(coordinates.size(0), device=device)
                ]
                shifted = th.roll(shuffled, shifts=-1, dims=0)
                previous_batch.append(shuffled[:, 0])
                previous_time.append(shuffled[:, 1])
                current_batch.append(shifted[:, 0])
                current_time.append(shifted[:, 1])

        if not previous_batch:
            empty = th.empty(0, dtype=th.long, device=device)
            return empty, empty, empty, empty
        return (
            th.cat(previous_batch),
            th.cat(previous_time),
            th.cat(current_batch),
            th.cat(current_time),
        )

    @staticmethod
    def _gather_relation_states(
        parameters_by_time,
        probabilities_by_time,
        batch_indices,
        time_indices,
        batch_size,
        n_agents,
    ):
        """Gather arbitrary trajectory states into the pair helper layout."""
        sample_count = batch_indices.numel()
        sort_order = th.argsort(time_indices)
        inverse_order = th.argsort(sort_order)
        sorted_batch = batch_indices[sort_order]
        sorted_time = time_indices[sort_order]
        unique_times = th.unique_consecutive(sorted_time).tolist()
        selected_parameters = []
        for component_index in range(len(parameters_by_time[0])):
            time_chunks = []
            for time_index in unique_times:
                time_mask = sorted_time == time_index
                component = parameters_by_time[time_index][
                    component_index
                ].reshape(
                    batch_size,
                    n_agents,
                    *parameters_by_time[time_index][component_index].shape[1:],
                )
                time_chunks.append(
                    component.index_select(0, sorted_batch[time_mask])
                )
            selected = th.cat(time_chunks, dim=0).index_select(0, inverse_order)
            selected_parameters.append(
                selected.reshape(
                    sample_count * n_agents,
                    *selected.shape[2:],
                )
            )

        probability_chunks = []
        for time_index in unique_times:
            time_mask = sorted_time == time_index
            probability = probabilities_by_time[time_index].reshape(
                2, batch_size, n_agents, -1
            )
            probability_chunks.append(
                probability.index_select(1, sorted_batch[time_mask])
            )
        selected_probabilities = (
            th.cat(probability_chunks, dim=1)
            .index_select(1, inverse_order)
            .reshape(2, sample_count * n_agents, -1)
        )
        return tuple(selected_parameters), selected_probabilities

    @staticmethod
    def _property_group_ids(slot_names):
        """Group repeated ally/opponent attributes without merging x/y."""
        group_by_key = {}
        group_ids = []
        for slot_name in slot_names:
            parts = slot_name.split("_", 2)
            if (
                len(parts) == 3
                and parts[0] in {"ally", "opponent"}
                and parts[1].isdigit()
            ):
                key = (parts[0], parts[2])
            else:
                key = ("singleton", slot_name)
            if key not in group_by_key:
                group_by_key[key] = len(group_by_key)
            group_ids.append(group_by_key[key])
        return tuple(group_ids)

    def _group_mask_probabilities(self, probabilities):
        """Mean-pool a raw-slot mask into a permutation-invariant group mask."""
        group_ids = th.as_tensor(
            self.mask_parameter_relation_group_ids,
            device=probabilities.device,
            dtype=th.long,
        )
        if probabilities.size(-1) != group_ids.numel():
            raise RuntimeError("Mask slots do not match relation group ids")
        group_count = int(group_ids.max().item()) + 1
        grouped = probabilities.new_zeros(*probabilities.shape[:-1], group_count)
        scatter_index = group_ids.view(
            *((1,) * (probabilities.dim() - 1)), group_ids.numel()
        ).expand_as(probabilities)
        grouped.scatter_add_(-1, scatter_index, probabilities)
        counts = probabilities.new_zeros(group_count)
        counts.scatter_add_(
            0,
            group_ids,
            probabilities.new_ones(group_ids.numel()),
        )
        return grouped / counts.clamp(min=1.0)

    def _gradient_importance_gate_loss(self, td_loss, gate_graphs):
        """Penalize keeping slots whose sampled gates have low TD sensitivity.

        The importance weights are detached, so this is first-order training:
        task gradients decide which slots are protected while the auxiliary
        gradient only pushes low-importance keep probabilities downward.
        """
        zero = td_loss.new_zeros(())
        if not gate_graphs:
            return zero, {
                "dynamic_gate_gradient_importance_mean": zero,
                "dynamic_gate_gradient_importance_max": zero,
            }
        sampled_gates = [item[1] for item in gate_graphs]
        gradients = th.autograd.grad(
            td_loss,
            sampled_gates,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        penalties = []
        importance_values = []
        for (probability, _), gradient in zip(gate_graphs, gradients):
            if gradient is None:
                continue
            importance = gradient.detach().abs()
            normalized = importance / importance.mean(
                dim=-1, keepdim=True
            ).clamp(min=self.adaptive_auxiliary_eps)
            low_importance_weight = th.exp(-normalized).detach()
            penalties.append((probability * low_importance_weight).mean())
            importance_values.append(importance)
        if not penalties:
            return zero, {
                "dynamic_gate_gradient_importance_mean": zero,
                "dynamic_gate_gradient_importance_max": zero,
            }
        flattened = th.cat([value.reshape(-1) for value in importance_values])
        return th.stack(penalties).mean(), {
            "dynamic_gate_gradient_importance_mean": flattened.mean(),
            "dynamic_gate_gradient_importance_max": flattened.max(),
        }

    def _generated_parameter_conditional_nll(
        self,
        masked_parameters,
        full_target_parameters,
        state_valid,
        batch_size,
        n_agents,
    ):
        """Fixed-scale Gaussian NLL of full target parameters under masked obs.

        The detached target network supplies the Gaussian mean target. Each
        parameter block is normalized by its target RMS so one large head does
        not dominate merely because of scale. The variance is fixed to prevent
        the likelihood model from improving by inflating a learned variance.
        """
        if len(masked_parameters) != len(full_target_parameters):
            raise RuntimeError("Generated parameter structures do not match.")
        squared_sum = None
        parameter_count = 0
        for masked, full_target in zip(masked_parameters, full_target_parameters):
            if masked.shape != full_target.shape:
                raise RuntimeError("Masked and full target parameter shapes do not match.")
            masked_flat = masked.reshape(batch_size, n_agents, -1)
            target_flat = full_target.detach().reshape(batch_size, n_agents, -1)
            target_rms = target_flat.pow(2).mean(dim=-1, keepdim=True).sqrt()
            target_rms = target_rms.clamp(min=self.adaptive_auxiliary_eps)
            standardized = (masked_flat - target_flat) / (
                self.generated_parameter_likelihood_std * target_rms
            )
            part_sum = 0.5 * standardized.pow(2).sum(dim=-1)
            squared_sum = part_sum if squared_sum is None else squared_sum + part_sum
            parameter_count += standardized.shape[-1]
        if squared_sum is None or parameter_count == 0:
            zero = state_valid.new_zeros((), dtype=th.float32)
            return zero, zero
        per_agent_nll = squared_sum / float(parameter_count)
        valid = state_valid.reshape(batch_size, 1).expand(-1, n_agents)
        valid_float = valid.to(per_agent_nll.dtype)
        return (per_agent_nll * valid_float).sum(), valid_float.sum()

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

    def _audit_branch_drop(
        self, batch, actions, targets, td_mask, capturer, t_env
    ):
        if capturer is None or not capturer.branch_drop_needs_audit(t_env):
            return False
        if (
            t_env - self.last_semantic_router_audit_t
            < self.semantic_router_audit_interval
        ):
            return False

        audit_batch = self._binary_audit_batch(batch)
        audit_size = audit_batch.batch_size
        audit_actions = actions[:audit_size]
        audit_targets = targets[:audit_size]
        audit_td_mask = td_mask[:audit_size]
        current_keep = capturer.branch_group_keep_state().detach() >= 0.5
        group_scores = audit_td_mask.new_zeros(
            2, capturer.semantic_field_count
        )
        stats = {}
        try:
            capturer.set_branch_drop_audit()
            if capturer.branch_drop_mode == "td_benefit":
                with th.no_grad():
                    baseline = self._counterfactual_td_loss(
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

            for branch_index, branch_name in enumerate(("linear", "attention")):
                for group_index in range(capturer.semantic_field_count):
                    baseline_is_keep = bool(
                        current_keep[branch_index, group_index].item()
                    )
                    capturer.set_branch_drop_audit(
                        branch_index,
                        group_index,
                        keep=not baseline_is_keep,
                    )
                    if capturer.branch_drop_mode == "td_benefit":
                        with th.no_grad():
                            alternative = self._counterfactual_td_loss(
                                audit_batch,
                                audit_actions,
                                audit_targets,
                                audit_td_mask,
                            )
                        if baseline_is_keep:
                            keep_loss, drop_loss = baseline, alternative
                        else:
                            keep_loss, drop_loss = alternative, baseline
                        # Positive means KEEP lowers TD loss; negative means
                        # the branch-slot is harmful and DROP is beneficial.
                        score = (drop_loss - keep_loss) / keep_loss.abs().clamp(
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
                    group_scores[branch_index, group_index] = score.detach()
                    stats[
                        "branch_drop_{}_group_{}_score".format(
                            branch_name, group_index
                        )
                    ] = score.detach()
                    capturer.set_branch_drop_audit()
        finally:
            capturer.set_branch_drop_audit()

        changed = capturer.update_branch_drop(t_env, group_scores)
        self._sync_branch_drop_to_target(capturer)
        stats.update(
            {
                "branch_drop_audit_score_mean": group_scores.mean(),
                "branch_drop_audit_score_min": group_scores.min(),
                "branch_drop_audit_score_max": group_scores.max(),
                "branch_drop_audit_changed": group_scores.new_tensor(
                    float(changed)
                ),
            }
        )
        self.latest_semantic_counterfactual_stats = stats
        self.last_semantic_router_audit_t = t_env
        return True

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if hasattr(self.mac, "set_dynamic_branch_gate_t_env"):
            self.mac.set_dynamic_branch_gate_t_env(t_env)
        if hasattr(self.target_mac, "set_dynamic_branch_gate_t_env"):
            self.target_mac.set_dynamic_branch_gate_t_env(t_env)
        importance_training_phase = self._importance_training_phase(t_env)
        if (
            self.importance_alternating_training
            and importance_training_phase
            != self.last_importance_training_phase
        ):
            self.logger.console_logger.info(
                "Importance alternating phase | t_env={} | {}".format(
                    t_env, importance_training_phase
                )
            )
            self.last_importance_training_phase = importance_training_phase
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        semantic_router = self._semantic_router(self.mac)
        branch_drop_capturer = self._branch_drop_capturer(self.mac)
        binary_rehearsal_active = self._start_semantic_binary_rehearsal(
            semantic_router, t_env
        )
        if binary_rehearsal_active:
            self._set_binary_rehearsal_route(True)
        if (
            semantic_router is not None
            and semantic_router.semantic_router_needs_critical_importance()
        ):
            semantic_router.begin_semantic_critical_capture()
        generated_parameter_graphs = []
        condition_graphs = (
            []
            if self.condition_gradient_consistency_active
            and t_env >= self.condition_gradient_consistency_warmup_steps
            else None
        )
        generated_parameter_stability_enabled = (
            self.generated_parameter_stability_active
            and t_env >= self.generated_parameter_stability_warmup_steps
        )
        previous_generated_parameters = None
        generated_parameter_stability_sum = None
        generated_parameter_stability_count = mask.new_zeros(())
        generated_parameter_likelihood_enabled = (
            self.generated_parameter_likelihood_active
            and t_env >= self.generated_parameter_likelihood_warmup_steps
        )
        generated_parameter_likelihood_sum = None
        generated_parameter_likelihood_count = mask.new_zeros(())
        td_weighted_parameter_likelihood_enabled = (
            self.td_weighted_parameter_likelihood_active
            and t_env >= self.td_weighted_parameter_likelihood_warmup_steps
        )
        td_parameter_log_probs = (
            [] if td_weighted_parameter_likelihood_enabled else None
        )
        trajectory_parameter_likelihood_enabled = (
            self.trajectory_parameter_likelihood_active
            and t_env >= self.trajectory_parameter_likelihood_warmup_steps
        )
        trajectory_parameter_projection_sum = None
        trajectory_parameter_projection_count = mask.new_zeros(())
        importance_auxiliary_enabled = (
            t_env >= self.importance_auxiliary_warmup_steps
            and (
                not self.importance_alternating_training
                or importance_training_phase == "gate_td_aux"
            )
        )
        temporal_param_auxiliary_enabled = (
            self.temporal_param_auxiliary_active
            and importance_auxiliary_enabled
        )
        previous_temporal_parameters = None
        temporal_param_loss_sum = None
        temporal_param_change_sum = mask.new_zeros(())
        temporal_param_small_count = mask.new_zeros(())
        temporal_param_valid_count = mask.new_zeros(())
        relation_parameters = (
            []
            if self.mask_parameter_relation_active
            and importance_auxiliary_enabled
            else None
        )
        relation_probabilities = (
            []
            if self.mask_parameter_relation_active
            and importance_auxiliary_enabled
            else None
        )
        relation_valid_steps = (
            []
            if self.mask_parameter_relation_active
            and importance_auxiliary_enabled
            else None
        )
        perturbed_parameter_sum = None
        perturbed_parameter_count = mask.new_zeros(())
        perturb_probe_times = (
            self._uniform_probe_times(
                max(1, mask.shape[1]),
                self.parameter_perturbation_probe_timesteps,
            )
            if self.perturbed_parameter_importance_active
            and importance_auxiliary_enabled
            else set()
        )
        dynamic_gate_graphs = (
            []
            if self.gradient_importance_active and importance_auxiliary_enabled
            else None
        )
        perturbed_head_parameter_graphs = (
            []
            if self.perturbed_head_td_quality_active
            and importance_auxiliary_enabled
            else None
        )
        perturbed_head_hidden_graphs = (
            [] if perturbed_head_parameter_graphs is not None else None
        )
        perturbed_head_interaction_graphs = (
            [] if perturbed_head_parameter_graphs is not None else None
        )
        perturbed_head_enemy_masks = (
            [] if perturbed_head_parameter_graphs is not None else None
        )
        if hasattr(self.mac, "set_td_parameter_sampling_enabled"):
            self.mac.set_td_parameter_sampling_enabled(
                td_weighted_parameter_likelihood_enabled
            )
        precomputed_target_mac_out = []
        precomputed_target_relation_conditions = (
            [] if self.target_relation_mixer_gate is not None else None
        )
        if generated_parameter_likelihood_enabled:
            self.target_mac.init_hidden(batch.batch_size)
            self.target_mac.set_dynamic_branch_gate_force_open(True)
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
                hidden_state_before = self.mac.hidden_states
                self.mac.agent.capture_semantic_parameter_graph = (
                    t in parameter_probe_times
                )
                if semantic_router is not None:
                    semantic_router.capture_semantic_observation_score = (
                        t in observation_probe_times
                    )
                mac_out.append(self.mac.forward(batch, t=t))
                if (
                    perturbed_head_parameter_graphs is not None
                    and t < mask.shape[1]
                ):
                    generated_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    policy_hidden = getattr(
                        self.mac, "latest_policy_hidden_graph", None
                    )
                    if generated_parameters is None or policy_hidden is None:
                        raise RuntimeError(
                            "Perturbed-head TD quality requires differentiable "
                            "generated parameters and policy hidden states"
                        )
                    perturbed_head_parameter_graphs.append(generated_parameters)
                    perturbed_head_hidden_graphs.append(policy_hidden)
                    perturbed_head_interaction_graphs.append(
                        getattr(
                            self.mac.agent,
                            "latest_policy_interaction_input_graph",
                            None,
                        )
                    )
                    perturbed_head_enemy_masks.append(
                        getattr(
                            self.mac.agent,
                            "latest_policy_enemy_mask_graph",
                            None,
                        )
                    )
                if dynamic_gate_graphs is not None and t < mask.shape[1]:
                    gate_probability_graph = getattr(
                        self.mac,
                        "latest_dynamic_branch_probabilities_graph",
                        None,
                    )
                    if gate_probability_graph is None:
                        raise RuntimeError(
                            "Gradient importance requires differentiable gate probabilities"
                        )
                    sampled_gate_graph = getattr(
                        self.mac, "latest_dynamic_branch_gates_graph", None
                    )
                    if sampled_gate_graph is None:
                        raise RuntimeError(
                            "Gradient importance requires differentiable sampled gates"
                        )
                    dynamic_gate_graphs.append(
                        (gate_probability_graph, sampled_gate_graph)
                    )
                if trajectory_parameter_likelihood_enabled and t < mask.shape[1]:
                    current_projection = getattr(
                        self.mac, "latest_trajectory_parameter_projection", None
                    )
                    if current_projection is None:
                        raise RuntimeError(
                            "Trajectory parameter likelihood requires a current "
                            "generated-parameter projection."
                        )
                    behavior_projection = batch[
                        "trajectory_parameter_projection"
                    ][:, t].detach().to(dtype=current_projection.dtype)
                    per_agent_nll = 0.5 * (
                        current_projection - behavior_projection
                    ).pow(2).mean(dim=-1)
                    state_valid = (
                        mask[:, t].reshape(batch.batch_size, 1) > 0
                    ).expand(-1, self.args.n_agents)
                    valid_float = state_valid.to(per_agent_nll.dtype)
                    likelihood_sum = (per_agent_nll * valid_float).sum()
                    trajectory_parameter_projection_sum = (
                        likelihood_sum
                        if trajectory_parameter_projection_sum is None
                        else trajectory_parameter_projection_sum + likelihood_sum
                    )
                    trajectory_parameter_projection_count = (
                        trajectory_parameter_projection_count + valid_float.sum()
                    )
                if td_parameter_log_probs is not None:
                    parameter_log_prob = getattr(
                        self.mac, "latest_generated_parameter_log_prob", None
                    )
                    if parameter_log_prob is None:
                        raise RuntimeError(
                            "TD-weighted parameter likelihood requires a "
                            "stochastic generated-parameter score."
                        )
                    td_parameter_log_probs.append(parameter_log_prob)
                if generated_parameter_likelihood_enabled:
                    masked_generated_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    if masked_generated_parameters is None:
                        raise RuntimeError(
                            "Parameter likelihood requires differentiable masked "
                            "hypernetwork outputs."
                        )
                    with th.no_grad():
                        precomputed_target_mac_out.append(
                            self.target_mac.forward(batch, t=t)
                        )
                        full_target_parameters = getattr(
                            self.target_mac, "latest_generated_parameter_graph", None
                        )
                        if full_target_parameters is None:
                            raise RuntimeError(
                                "Parameter likelihood requires full-observation "
                                "target hypernetwork outputs."
                            )
                        if precomputed_target_relation_conditions is not None:
                            target_condition = getattr(
                                self.target_mac, "latest_condition", None
                            )
                            if target_condition is None:
                                raise RuntimeError(
                                    "Relation mixer gate requires a target condition."
                                )
                            precomputed_target_relation_conditions.append(
                                target_condition
                            )
                    if t < mask.shape[1]:
                        state_valid = mask[:, t].reshape(batch.batch_size) > 0
                        likelihood_sum, likelihood_count = (
                            self._generated_parameter_conditional_nll(
                                masked_generated_parameters,
                                full_target_parameters,
                                state_valid,
                                batch.batch_size,
                                self.args.n_agents,
                            )
                        )
                        generated_parameter_likelihood_sum = (
                            likelihood_sum
                            if generated_parameter_likelihood_sum is None
                            else generated_parameter_likelihood_sum + likelihood_sum
                        )
                        generated_parameter_likelihood_count = (
                            generated_parameter_likelihood_count + likelihood_count
                        )
                if generated_parameter_stability_enabled and t < mask.shape[1]:
                    current_generated_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    if current_generated_parameters is None:
                        raise RuntimeError(
                            "Generated-parameter stability requires the exact "
                            "differentiable hypernetwork outputs."
                        )
                    if previous_generated_parameters is not None:
                        pair_valid = (
                            mask[:, t].reshape(batch.batch_size) > 0
                        ) & (mask[:, t - 1].reshape(batch.batch_size) > 0)
                        pair_sum, pair_count = (
                            self._generated_parameter_stability_pair(
                                previous_generated_parameters,
                                current_generated_parameters,
                                pair_valid,
                                batch.batch_size,
                                self.args.n_agents,
                            )
                        )
                        generated_parameter_stability_sum = (
                            pair_sum
                            if generated_parameter_stability_sum is None
                            else generated_parameter_stability_sum + pair_sum
                        )
                        generated_parameter_stability_count = (
                            generated_parameter_stability_count + pair_count
                        )
                    previous_generated_parameters = current_generated_parameters
                if temporal_param_auxiliary_enabled and t < mask.shape[1]:
                    current_temporal_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    if current_temporal_parameters is None:
                        raise RuntimeError(
                            "Temporal parameter auxiliary requires exact "
                            "differentiable hypernetwork outputs"
                        )
                    if previous_temporal_parameters is not None:
                        pair_valid = (
                            mask[:, t].reshape(batch.batch_size) > 0
                        ) & (mask[:, t - 1].reshape(batch.batch_size) > 0)
                        change, valid_float = (
                            self._normalized_generated_parameter_change(
                                previous_temporal_parameters,
                                current_temporal_parameters,
                                pair_valid,
                                batch.batch_size,
                                self.args.n_agents,
                                self.temporal_param_scale_eps,
                            )
                        )
                        if self.temporal_param_stability_active:
                            pair_penalty = change
                        else:
                            small_change = (
                                change
                                < 0.5 * self.temporal_param_switch_margin
                            ).detach()
                            pair_penalty = th.where(
                                small_change,
                                change.pow(2),
                                th.zeros_like(change),
                            )
                            temporal_param_small_count = (
                                temporal_param_small_count
                                + (small_change.to(change.dtype) * valid_float).sum()
                            )
                        pair_loss_sum = (pair_penalty * valid_float).sum()
                        temporal_param_loss_sum = (
                            pair_loss_sum
                            if temporal_param_loss_sum is None
                            else temporal_param_loss_sum + pair_loss_sum
                        )
                        temporal_param_change_sum = (
                            temporal_param_change_sum
                            + (change.detach() * valid_float).sum()
                        )
                        temporal_param_valid_count = (
                            temporal_param_valid_count + valid_float.sum()
                        )
                    previous_temporal_parameters = current_temporal_parameters
                if relation_parameters is not None and t < mask.shape[1]:
                    current_relation_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    relation_mask_attribute = (
                        "latest_dynamic_branch_gates_graph"
                        if self.mask_parameter_relation_mask_source
                        == "sampled_gate"
                        else "latest_dynamic_branch_probabilities_graph"
                    )
                    current_relation_probabilities = getattr(
                        self.mac, relation_mask_attribute, None
                    )
                    if (
                        current_relation_parameters is None
                        or current_relation_probabilities is None
                    ):
                        raise RuntimeError(
                            "Mask-parameter relation requires exact generated "
                            "parameters and the requested dynamic gate values"
                        )
                    relation_parameters.append(current_relation_parameters)
                    relation_probabilities.append(current_relation_probabilities)
                    relation_valid_steps.append(
                        mask[:, t].reshape(batch.batch_size) > 0
                    )
                if t in perturb_probe_times:
                    base_parameters = getattr(
                        self.mac, "latest_generated_parameter_graph", None
                    )
                    base_gates = getattr(
                        self.mac, "latest_dynamic_branch_gates_graph", None
                    )
                    if base_parameters is None or base_gates is None:
                        raise RuntimeError(
                            "Perturbed parameter importance requires generated "
                            "parameters and current dynamic gates"
                        )
                    perturbed_parameters = (
                        self.mac.generated_parameters_with_observation_perturbation(
                            batch,
                            t,
                            hidden_state_before,
                            base_gates,
                            self.parameter_perturbation_relative_std,
                        )
                    )
                    if perturbed_parameters is None:
                        raise RuntimeError(
                            "Perturbed forward did not expose generated parameters"
                        )
                    state_valid = mask[:, t].reshape(batch.batch_size) > 0
                    perturb_sum, perturb_count = (
                        self._generated_parameter_stability_pair(
                            base_parameters,
                            perturbed_parameters,
                            state_valid,
                            batch.batch_size,
                            self.args.n_agents,
                        )
                    )
                    perturbed_parameter_sum = (
                        perturb_sum
                        if perturbed_parameter_sum is None
                        else perturbed_parameter_sum + perturb_sum
                    )
                    perturbed_parameter_count = (
                        perturbed_parameter_count + perturb_count
                    )
                if condition_graphs is not None:
                    condition_graph = getattr(
                        self.mac, "latest_condition_graph", None
                    )
                    if condition_graph is None:
                        raise RuntimeError(
                            "Condition-gradient consistency requires a "
                            "differentiable condition graph."
                        )
                    condition_graphs.append(condition_graph)
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
            mask_parameter_relation_sum = None
            mask_parameter_relation_count = mask.new_zeros(())
            mask_parameter_pair_count = 0
            if relation_parameters is not None and len(relation_parameters) > 1:
                if self.mask_parameter_relation_pairing == "fixed":
                    pair_indices = set()
                    for current_index in range(1, len(relation_parameters)):
                        pair_indices.add((current_index - 1, current_index))
                        # A non-local pair keeps revisited/different behavior modes
                        # visible without the O(T^2) cost of every trajectory pair.
                        anchor_index = current_index // 2
                        if anchor_index < current_index - 1:
                            pair_indices.add((anchor_index, current_index))
                    for previous_index, current_index in sorted(pair_indices):
                        pair_valid = (
                            relation_valid_steps[previous_index]
                            & relation_valid_steps[current_index]
                        )
                        (
                            pair_sum,
                            pair_count,
                            _,
                            _,
                        ) = self._mask_parameter_relation_pair(
                            relation_parameters[previous_index],
                            relation_parameters[current_index],
                            relation_probabilities[previous_index],
                            relation_probabilities[current_index],
                            pair_valid,
                            batch.batch_size,
                            self.args.n_agents,
                        )
                        mask_parameter_relation_sum = (
                            pair_sum
                            if mask_parameter_relation_sum is None
                            else mask_parameter_relation_sum + pair_sum
                        )
                        mask_parameter_relation_count = (
                            mask_parameter_relation_count + pair_count
                        )
                        mask_parameter_pair_count += 1
                else:
                    valid_steps = th.stack(relation_valid_steps, dim=1)
                    (
                        previous_batch,
                        previous_time,
                        current_batch,
                        current_time,
                    ) = self._random_relation_pair_indices(
                        valid_steps,
                        self.mask_parameter_relation_pairing,
                    )
                    random_pair_count = previous_batch.numel()
                    if random_pair_count > 0:
                        previous_parameters, previous_probabilities = (
                            self._gather_relation_states(
                                relation_parameters,
                                relation_probabilities,
                                previous_batch,
                                previous_time,
                                batch.batch_size,
                                self.args.n_agents,
                            )
                        )
                        current_parameters, current_probabilities = (
                            self._gather_relation_states(
                                relation_parameters,
                                relation_probabilities,
                                current_batch,
                                current_time,
                                batch.batch_size,
                                self.args.n_agents,
                            )
                        )
                        pair_valid = th.ones(
                            random_pair_count,
                            dtype=th.bool,
                            device=mask.device,
                        )
                        (
                            mask_parameter_relation_sum,
                            mask_parameter_relation_count,
                            _,
                            _,
                        ) = self._mask_parameter_relation_pair(
                            previous_parameters,
                            current_parameters,
                            previous_probabilities,
                            current_probabilities,
                            pair_valid,
                            random_pair_count,
                            self.args.n_agents,
                        )
                        mask_parameter_pair_count = random_pair_count

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
                target_mac_out = precomputed_target_mac_out
                target_relation_conditions = precomputed_target_relation_conditions
                if not generated_parameter_likelihood_enabled:
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
                else:
                    self.target_mac.set_dynamic_branch_gate_force_open(False)
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
            random_drop_auxiliary_loss = td_loss.new_zeros(())
            kl80_random_auxiliary_loss = td_loss.new_zeros(())
            kl80_random_auxiliary_coef = 0.0
            random_drop_auxiliary_enabled = (
                self.random_drop_auxiliary_active
                and t_env >= self.importance_auxiliary_warmup_steps
                and self.random_drop_auxiliary_coef > 0.0
            )
            if random_drop_auxiliary_enabled:
                random_mac_out = []
                random_relation_conditions = (
                    [] if self.relation_mixer_gate is not None else None
                )
                self.mac.init_hidden(batch.batch_size)
                auxiliary_kl_terms = []
                random_capturer = getattr(self.mac.agent, "rpg_relation_capturer", None)
                try:
                    if self.concrete_random_drop_auxiliary:
                        random_capturer.kl80_auxiliary_enabled = True
                    elif not self.random_drop_auxiliary_input_mask:
                        self.mac.set_dynamic_branch_gate_random_aux_combine_mode(
                            self.random_drop_auxiliary_combine_mode
                        )
                    if not self.concrete_random_drop_auxiliary and self.random_drop_auxiliary_scope == "episode":
                        keep_probability = td_loss.new_tensor(
                            self.random_drop_auxiliary_keep_probability
                        )
                        if self.random_drop_auxiliary_input_mask:
                            self.mac.set_random_drop_auxiliary_input_keep_probability(
                                keep_probability
                            )
                        else:
                            self.mac.set_dynamic_branch_gate_random_aux_mask(
                                keep_probability
                            )
                    for t in range(batch.max_seq_length):
                        if not self.concrete_random_drop_auxiliary and self.random_drop_auxiliary_scope == "timestep":
                            keep_probability = td_loss.new_tensor(
                                self.random_drop_auxiliary_keep_probability
                            )
                            if self.random_drop_auxiliary_input_mask:
                                self.mac.set_random_drop_auxiliary_input_keep_probability(
                                    keep_probability
                                )
                            else:
                                self.mac.set_dynamic_branch_gate_random_aux_mask(
                                    keep_probability
                                )
                        random_mac_out.append(self.mac.forward(batch, t=t))
                        if self.kl80_random_drop_auxiliary:
                            auxiliary_kl_terms.append(random_capturer.latest_kl80_auxiliary_loss)
                        if random_relation_conditions is not None:
                            condition = getattr(self.mac, "latest_condition", None)
                            if condition is None:
                                raise RuntimeError(
                                    "Random-drop auxiliary with relation mixer "
                                    "requires relation conditions"
                                )
                            random_relation_conditions.append(condition)
                finally:
                    if self.concrete_random_drop_auxiliary:
                        random_capturer.kl80_auxiliary_enabled = False
                    if self.random_drop_auxiliary_input_mask:
                        self.mac.set_random_drop_auxiliary_input_keep_probability(None)
                    else:
                        self.mac.set_dynamic_branch_gate_random_aux_mask(None)
                random_mac_out = th.stack(random_mac_out, dim=1)
                random_chosen_qvals = th.gather(
                    random_mac_out[:, :-1], dim=3, index=actions
                ).squeeze(3)
                if self.mixer is not None:
                    if random_relation_conditions is not None:
                        random_relation_conditions = th.stack(
                            random_relation_conditions, dim=1
                        )
                        random_chosen_qvals = self._apply_relation_gate(
                            random_chosen_qvals,
                            random_relation_conditions[:, :-1],
                            target=False,
                        )
                    random_chosen_qvals = self.mixer(
                        random_chosen_qvals, batch["state"][:, :-1]
                    )
                random_td_error = random_chosen_qvals - targets.detach()
                random_masked_td_error = random_td_error * td_mask
                random_drop_auxiliary_loss = (
                    random_masked_td_error.pow(2).sum()
                    / td_mask.sum().clamp(min=1.0)
                )
                if auxiliary_kl_terms:
                    kl80_random_auxiliary_loss = th.stack(auxiliary_kl_terms).mean()
                    kl80_random_auxiliary_coef = self._adaptive_auxiliary_coefficient(
                        td_loss, kl80_random_auxiliary_loss
                    )
            perturbed_head_td_quality_loss = td_loss.new_zeros(())
            if perturbed_head_parameter_graphs is not None:
                time_steps = len(perturbed_head_parameter_graphs)
                if time_steps != mask.shape[1]:
                    raise RuntimeError(
                        "Perturbed-head TD quality did not capture every train timestep"
                    )
                # Reuse each timestep's existing hidden state and generated
                # tensors directly. Avoid stacking the much larger parameter
                # blocks, which would create an unnecessary full trajectory
                # copy before the perturbed head is even evaluated.
                perturbed_mac_out = th.stack(
                    [
                        self.mac.agent.perturbed_q_from_generated_parameters(
                            hidden,
                            parameters,
                            self.perturbed_head_relative_std,
                            self.perturbed_head_minimum_rms,
                            interaction_input=interaction_input,
                            enemy_mask=enemy_mask,
                        )
                        for hidden, parameters, interaction_input, enemy_mask in zip(
                            perturbed_head_hidden_graphs,
                            perturbed_head_parameter_graphs,
                            perturbed_head_interaction_graphs,
                            perturbed_head_enemy_masks,
                        )
                    ],
                    dim=1,
                )
                perturbed_chosen_qvals = th.gather(
                    perturbed_mac_out, dim=3, index=actions
                ).squeeze(3)
                if self.mixer is not None:
                    if self.relation_mixer_gate is not None:
                        perturbed_chosen_qvals = self._apply_relation_gate(
                            perturbed_chosen_qvals,
                            relation_conditions[:, :-1],
                            target=False,
                        )
                    perturbed_chosen_qvals = self.mixer(
                        perturbed_chosen_qvals, batch["state"][:, :-1]
                    )
                perturbed_td_error = (
                    perturbed_chosen_qvals - targets.detach()
                )
                perturbed_head_td_quality_loss = (
                    perturbed_td_error.pow(2) * td_mask
                ).sum() / td_mask.sum().clamp(min=1.0)
            td_weighted_parameter_likelihood_loss = td_loss.new_zeros(())
            td_parameter_quality_advantage_std = td_loss.new_zeros(())
            td_parameter_mean_log_prob = td_loss.new_zeros(())
            if td_parameter_log_probs is not None:
                parameter_log_prob = th.stack(td_parameter_log_probs, dim=1)
                parameter_log_prob = parameter_log_prob[:, :-1].mean(dim=-1)
                per_state_td = td_error.detach().pow(2).mean(dim=-1)
                state_mask = td_mask.detach().mean(dim=-1)
                valid_count = state_mask.sum().clamp(min=1.0)
                td_baseline = (per_state_td * state_mask).sum() / valid_count
                centered_td = per_state_td - td_baseline
                td_std = (
                    (centered_td.pow(2) * state_mask).sum() / valid_count
                ).sqrt().clamp(min=self.adaptive_auxiliary_eps)
                quality_advantage = -centered_td / td_std
                td_weighted_parameter_likelihood_loss = -(
                    quality_advantage.detach() * parameter_log_prob * state_mask
                ).sum() / valid_count
                td_parameter_quality_advantage_std = (
                    quality_advantage.detach().pow(2) * state_mask
                ).sum().div(valid_count).sqrt()
                td_parameter_mean_log_prob = (
                    parameter_log_prob.detach() * state_mask
                ).sum() / valid_count
            aux_loss = th.stack(aux_losses).mean() if aux_losses else td_loss.new_zeros(())
            gradient_importance_loss = td_loss.new_zeros(())
            gradient_importance_stats = {}
            if dynamic_gate_graphs is not None:
                (
                    gradient_importance_loss,
                    gradient_importance_stats,
                ) = self._gradient_importance_gate_loss(
                    td_loss, dynamic_gate_graphs
                )
            teacher_td_loss = td_loss.new_zeros(())
            if teacher_mac_out is not None:
                teacher_chosen_qvals = th.gather(teacher_mac_out[:, :-1], dim=3, index=actions).squeeze(3)
                if self.mixer is not None:
                    teacher_chosen_qvals = self.mixer(teacher_chosen_qvals, batch["state"][:, :-1])
                teacher_td_error = teacher_chosen_qvals - targets.detach()
                teacher_masked_td_error = teacher_td_error * td_mask
                teacher_td_loss = (teacher_masked_td_error.pow(2).sum()) / td_mask.sum().clamp(min=1.0)
            condition_gradient_consistency_loss = td_loss.new_zeros(())
            condition_gradient_consistency_stats = {}
            if condition_graphs is not None:
                (
                    condition_gradient_consistency_loss,
                    condition_gradient_consistency_stats,
                ) = self._condition_gradient_consistency_loss(
                    td_error, td_mask, condition_graphs[:-1]
                )
            generated_parameter_stability_loss = td_loss.new_zeros(())
            if generated_parameter_stability_sum is not None:
                generated_parameter_stability_loss = (
                    generated_parameter_stability_sum
                    / generated_parameter_stability_count.clamp(min=1.0)
                )
            generated_parameter_likelihood_loss = td_loss.new_zeros(())
            if generated_parameter_likelihood_sum is not None:
                generated_parameter_likelihood_loss = (
                    generated_parameter_likelihood_sum
                    / generated_parameter_likelihood_count.clamp(min=1.0)
                )
            trajectory_parameter_likelihood_loss = td_loss.new_zeros(())
            if trajectory_parameter_projection_sum is not None:
                trajectory_parameter_likelihood_loss = (
                    trajectory_parameter_projection_sum
                    / trajectory_parameter_projection_count.clamp(min=1.0)
                )
            perturbed_parameter_importance_loss = td_loss.new_zeros(())
            if perturbed_parameter_sum is not None:
                perturbed_parameter_importance_loss = (
                    perturbed_parameter_sum
                    / perturbed_parameter_count.clamp(min=1.0)
                )
            temporal_param_auxiliary_loss = td_loss.new_zeros(())
            if temporal_param_loss_sum is not None:
                temporal_param_auxiliary_loss = (
                    temporal_param_loss_sum
                    / temporal_param_valid_count.clamp(min=1.0)
                )
            mask_parameter_relation_loss = td_loss.new_zeros(())
            if mask_parameter_relation_sum is not None:
                mask_parameter_relation_loss = (
                    mask_parameter_relation_sum
                    / mask_parameter_relation_count.clamp(min=1.0)
                )
            condition_gradient_consistency_coef = (
                self.condition_gradient_consistency_coef
            )
            generated_parameter_stability_coef = (
                self.generated_parameter_stability_coef
            )
            trajectory_parameter_likelihood_coef = (
                self.generated_parameter_stability_coef
            )
            gate_auxiliary_coef = 1.0
            perturbed_parameter_importance_coef = 0.0
            gradient_importance_coef = 0.0
            perturbed_head_td_quality_coef = 0.0
            temporal_param_auxiliary_coef = 0.0
            mask_parameter_combined_auxiliary_coef = 0.0
            random_drop_auxiliary_coef = (
                self.random_drop_auxiliary_coef
                if random_drop_auxiliary_enabled
                else 0.0
            )
            adaptive_auxiliary_enabled = False
            if self.mask_parameter_relation_active:
                # Relation-family experiments use independently interpretable
                # fixed weights. A shared EMA coefficient allowed the larger
                # raw relation loss to swallow the temporal contribution.
                mask_parameter_combined_auxiliary_coef = 1.0
                if self.gate_regularization_active and aux_losses:
                    # KL/budget regularization is included in the gate-only
                    # fixed bundle below, so do not add it twice.
                    gate_auxiliary_coef = 0.0
            if self.adaptive_auxiliary_ratio_active:
                if self.gate_regularization_active and aux_losses:
                    gate_auxiliary_coef = self._adaptive_auxiliary_coefficient(
                        td_loss, aux_loss
                    )
                    adaptive_auxiliary_enabled = True
                elif self.perturbed_parameter_importance_active:
                    perturbed_parameter_importance_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, perturbed_parameter_importance_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif self.gradient_importance_active:
                    gradient_importance_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, gradient_importance_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif self.perturbed_head_td_quality_active:
                    perturbed_head_td_quality_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, perturbed_head_td_quality_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif self.temporal_param_auxiliary_active:
                    temporal_param_auxiliary_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, temporal_param_auxiliary_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif condition_graphs is not None:
                    condition_gradient_consistency_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, condition_gradient_consistency_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif generated_parameter_stability_sum is not None:
                    generated_parameter_stability_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, generated_parameter_stability_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif generated_parameter_likelihood_sum is not None:
                    generated_parameter_stability_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, generated_parameter_likelihood_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
                elif trajectory_parameter_projection_sum is not None:
                    trajectory_parameter_likelihood_coef = (
                        self._adaptive_auxiliary_coefficient(
                            td_loss, trajectory_parameter_likelihood_loss
                        )
                    )
                    adaptive_auxiliary_enabled = True
            if not adaptive_auxiliary_enabled:
                self.latest_adaptive_auxiliary_stats = {}

            dynamic_gate_training_frozen = (
                self._dynamic_gate_training_is_frozen(t_env)
            )
            if dynamic_gate_training_frozen:
                # The capturer detaches both gates and probabilities once the
                # freeze step is reached.  From then on the rest of the model
                # continues TD training with evaluation-equivalent hard gates,
                # while every gate-only objective must be inactive as well.
                gate_auxiliary_coef = 0.0
                perturbed_parameter_importance_coef = 0.0
                gradient_importance_coef = 0.0
                perturbed_head_td_quality_coef = 0.0
                temporal_param_auxiliary_coef = 0.0
                mask_parameter_combined_auxiliary_coef = 0.0
                random_drop_auxiliary_coef = 0.0
                self.latest_adaptive_auxiliary_stats = {}

            gate_only_importance_loss = None
            if (
                self.mask_parameter_relation_active
                and mask_parameter_relation_sum is not None
                and mask_parameter_combined_auxiliary_coef > 0.0
            ):
                combined_gate_auxiliary = (
                    self.mask_parameter_relation_coef
                    * mask_parameter_relation_loss
                )
                if self.temporal_param_auxiliary_active:
                    combined_gate_auxiliary = (
                        combined_gate_auxiliary
                        + self.mask_parameter_relation_temporal_coef
                        * temporal_param_auxiliary_loss
                    )
                if self.perturbed_head_td_quality_active:
                    combined_gate_auxiliary = (
                        combined_gate_auxiliary
                        + self.mask_parameter_relation_perturbed_head_coef
                        * perturbed_head_td_quality_loss
                    )
                if self.gate_regularization_active and aux_losses:
                    combined_gate_auxiliary = (
                        combined_gate_auxiliary
                        + self.mask_parameter_relation_gate_regularization_coef
                        * aux_loss
                    )
                gate_only_importance_loss = (
                    mask_parameter_combined_auxiliary_coef
                    * combined_gate_auxiliary
                )
            elif (
                self.perturbed_parameter_importance_active
                and perturbed_parameter_sum is not None
                and perturbed_parameter_importance_coef > 0.0
            ):
                gate_only_importance_loss = (
                    perturbed_parameter_importance_coef
                    * perturbed_parameter_importance_loss
                )
            elif (
                self.gradient_importance_active
                and dynamic_gate_graphs is not None
                and gradient_importance_coef > 0.0
            ):
                gate_only_importance_loss = (
                    gradient_importance_coef * gradient_importance_loss
                )
            elif (
                self.perturbed_head_td_quality_active
                and perturbed_head_parameter_graphs is not None
                and perturbed_head_td_quality_coef > 0.0
            ):
                gate_only_importance_loss = (
                    perturbed_head_td_quality_coef
                    * perturbed_head_td_quality_loss
                )
            elif (
                self.temporal_param_auxiliary_active
                and temporal_param_loss_sum is not None
                and temporal_param_auxiliary_coef > 0.0
            ):
                gate_only_importance_loss = (
                    temporal_param_auxiliary_coef
                    * temporal_param_auxiliary_loss
                )
            loss = (
                td_loss
                + random_drop_auxiliary_coef * random_drop_auxiliary_loss
                + kl80_random_auxiliary_coef * kl80_random_auxiliary_loss
                + gate_auxiliary_coef * aux_loss
                + float(
                    getattr(self.args, "clean_relation_teacher_td_coef", 0.0)
                )
                * teacher_td_loss
                + condition_gradient_consistency_coef
                * condition_gradient_consistency_loss
                + generated_parameter_stability_coef
                * generated_parameter_stability_loss
                + generated_parameter_stability_coef
                * generated_parameter_likelihood_loss
                + trajectory_parameter_likelihood_coef
                * trajectory_parameter_likelihood_loss
                + self.td_weighted_parameter_likelihood_coef
                * td_weighted_parameter_likelihood_loss
            )

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

        semantic_critical_score = None
        semantic_gradient_scale = 1.0
        self.optimiser.zero_grad()
        if self.importance_alternating_training:
            # Optimizers with momentum can still move a parameter whose
            # gradient is a zero tensor.  None is required so the inactive
            # parameter group is skipped completely in this phase.
            for parameter in self.params:
                parameter.grad = None
        if semantic_router is not None and semantic_router.semantic_probe_scale is not None:
            semantic_router.semantic_probe_scale.grad = None
        if semantic_router is not None and semantic_router.semantic_route_probe is not None:
            semantic_router.semantic_route_probe.grad = None
        if self.use_amp:
            semantic_gradient_scale = float(self.amp_scaler.get_scale())
            if importance_training_phase == "non_gate_td":
                self._backward_parameters_only(
                    td_loss, self.importance_non_gate_parameters
                )
            elif importance_training_phase == "gate_td_aux":
                gate_phase_loss = td_loss
                if gate_only_importance_loss is not None:
                    gate_phase_loss = (
                        gate_phase_loss + gate_only_importance_loss
                    )
                self._backward_parameters_only(
                    gate_phase_loss, self.importance_gate_parameters
                )
            else:
                self._backward_main_and_gate_only_auxiliary(
                    loss, gate_only_importance_loss
                )
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
            if importance_training_phase == "non_gate_td":
                self._backward_parameters_only(
                    td_loss, self.importance_non_gate_parameters
                )
            elif importance_training_phase == "gate_td_aux":
                gate_phase_loss = td_loss
                if gate_only_importance_loss is not None:
                    gate_phase_loss = (
                        gate_phase_loss + gate_only_importance_loss
                    )
                self._backward_parameters_only(
                    gate_phase_loss, self.importance_gate_parameters
                )
            else:
                self._backward_main_and_gate_only_auxiliary(
                    loss, gate_only_importance_loss
                )
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

        if (
            semantic_router is not None
            and semantic_router.semantic_router_needs_critical_importance()
        ):
            semantic_critical_score = (
                semantic_router.consume_semantic_critical_importance(
                    td_mask,
                    tail_fraction=self.semantic_critical_tail_fraction,
                    tail_weight=self.semantic_critical_tail_weight,
                    gradient_scale=semantic_gradient_scale,
                )
            )

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
                semantic_router.semantic_router_mode
                == "gradient_importance_critical"
                and semantic_critical_score is not None
            ):
                semantic_router.update_semantic_router(
                    t_env, semantic_critical_score
                )
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

        if (
            semantic_router is None
            or semantic_router.semantic_router_mode
            != "gradient_importance_critical"
        ):
            self._audit_semantic_counterfactual(
                batch, actions, targets, td_mask, semantic_router, t_env
            )
            self._audit_semantic_gradient_importance(
                batch, actions, targets, td_mask, semantic_router, t_env
            )
            self._audit_semantic_binary(
                batch, actions, targets, td_mask, semantic_router, t_env
            )
        self._audit_branch_drop(
            batch,
            actions,
            targets,
            td_mask,
            branch_drop_capturer,
            t_env,
        )
        if binary_rehearsal_active:
            self._set_binary_rehearsal_route(False)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            if self.importance_alternating_training:
                self.logger.log_stat(
                    "importance_alternating_gate_phase",
                    float(importance_training_phase == "gate_td_aux"),
                    t_env,
                )
            if aux_losses:
                self.logger.log_stat("loss_aux", aux_loss.item(), t_env)
            if self.gate_regularization_active:
                weighted_gate_auxiliary = gate_auxiliary_coef * aux_loss.item()
                self.logger.log_stat(
                    "weighted_loss_dynamic_gate_regularizer",
                    weighted_gate_auxiliary,
                    t_env,
                )
                self.logger.log_stat(
                    "dynamic_gate_regularizer_to_td_ratio",
                    weighted_gate_auxiliary
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
            if self.perturbed_parameter_importance_active:
                weighted_perturbed_parameter = (
                    perturbed_parameter_importance_coef
                    * perturbed_parameter_importance_loss.item()
                )
                self.logger.log_stat(
                    "loss_perturbed_generated_parameter_l1",
                    perturbed_parameter_importance_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_perturbed_generated_parameter_l1",
                    weighted_perturbed_parameter,
                    t_env,
                )
                self.logger.log_stat(
                    "perturbed_generated_parameter_to_td_ratio",
                    weighted_perturbed_parameter
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
            if self.gradient_importance_active:
                weighted_gradient_importance = (
                    gradient_importance_coef * gradient_importance_loss.item()
                )
                self.logger.log_stat(
                    "loss_dynamic_gate_gradient_importance",
                    gradient_importance_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_dynamic_gate_gradient_importance",
                    weighted_gradient_importance,
                    t_env,
                )
                self.logger.log_stat(
                    "dynamic_gate_gradient_importance_to_td_ratio",
                    weighted_gradient_importance
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                for stat_name, stat_value in gradient_importance_stats.items():
                    self.logger.log_stat(stat_name, stat_value.item(), t_env)
            if self.perturbed_head_td_quality_active:
                weighted_perturbed_head_td = (
                    perturbed_head_td_quality_coef
                    * perturbed_head_td_quality_loss.item()
                )
                self.logger.log_stat(
                    "loss_perturbed_head_td_quality",
                    perturbed_head_td_quality_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_perturbed_head_td_quality",
                    weighted_perturbed_head_td,
                    t_env,
                )
                self.logger.log_stat(
                    "perturbed_head_td_quality_to_td_ratio",
                    weighted_perturbed_head_td
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
            if self.random_drop_auxiliary_active:
                if self.kl80_random_drop_auxiliary:
                    self.logger.log_stat("loss_kl80_random_auxiliary", kl80_random_auxiliary_loss.item(), t_env)
                    self.logger.log_stat("kl80_random_auxiliary_coef", kl80_random_auxiliary_coef, t_env)
                    self.logger.log_stat("weighted_loss_kl80_random_auxiliary", kl80_random_auxiliary_coef * kl80_random_auxiliary_loss.item(), t_env)
                weighted_random_drop = (
                    random_drop_auxiliary_coef
                    * random_drop_auxiliary_loss.item()
                )
                self.logger.log_stat(
                    "loss_random_drop_td_auxiliary",
                    random_drop_auxiliary_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_random_drop_td_auxiliary",
                    weighted_random_drop,
                    t_env,
                )
                self.logger.log_stat(
                    "random_drop_td_auxiliary_to_td_ratio",
                    weighted_random_drop
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "random_drop_auxiliary_keep_probability",
                    float(self.mac.agent.rpg_relation_capturer.latest_kl80_auxiliary_probability[1].mean().item())
                    if self.kl80_random_drop_auxiliary else self.random_drop_auxiliary_keep_probability,
                    t_env,
                )
                self.logger.log_stat(
                    "random_drop_auxiliary_episode_scope",
                    float(self.random_drop_auxiliary_scope == "episode"),
                    t_env,
                )
            if self.temporal_param_auxiliary_active:
                valid_temporal_pairs = temporal_param_valid_count.clamp(min=1.0)
                weighted_temporal_loss = (
                    temporal_param_auxiliary_coef
                    * temporal_param_auxiliary_loss.item()
                )
                if self.counter_transformer_profile.get("relation"):
                    weighted_temporal_loss = (
                        mask_parameter_combined_auxiliary_coef
                        * self.mask_parameter_relation_temporal_coef
                        * temporal_param_auxiliary_loss.item()
                    )
                self.logger.log_stat(
                    "loss_temporal_parameter",
                    temporal_param_auxiliary_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "temporal_parameter_change_mean",
                    (
                        temporal_param_change_sum / valid_temporal_pairs
                    ).item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_temporal_parameter",
                    weighted_temporal_loss,
                    t_env,
                )
                self.logger.log_stat(
                    "temporal_parameter_to_td_ratio",
                    weighted_temporal_loss
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                if self.temporal_param_small_change_active:
                    self.logger.log_stat(
                        "temporal_parameter_small_change_fraction",
                        (
                            temporal_param_small_count / valid_temporal_pairs
                        ).item(),
                        t_env,
                    )
            if self.mask_parameter_relation_active:
                if self.counter_transformer_profile:
                    weighted_relation = (
                        mask_parameter_combined_auxiliary_coef
                        * self.mask_parameter_relation_coef
                        * mask_parameter_relation_loss.item()
                    )
                    self.logger.log_stat("weighted_loss_mask_parameter_relation", weighted_relation, t_env)
                    self.logger.log_stat(
                        "mask_parameter_relation_only_to_td_ratio",
                        weighted_relation / max(td_loss.item(), self.adaptive_auxiliary_eps), t_env,
                    )
                weighted_relation_bundle = (
                    mask_parameter_combined_auxiliary_coef
                    * (
                        self.mask_parameter_relation_coef
                        * mask_parameter_relation_loss.item()
                        + (
                            self.mask_parameter_relation_temporal_coef
                            * temporal_param_auxiliary_loss.item()
                            if self.temporal_param_auxiliary_active
                            else 0.0
                        )
                        + (
                            self.mask_parameter_relation_perturbed_head_coef
                            * perturbed_head_td_quality_loss.item()
                            if self.perturbed_head_td_quality_active
                            else 0.0
                        )
                        + (
                            self.mask_parameter_relation_gate_regularization_coef
                            * aux_loss.item()
                            if self.gate_regularization_active and aux_losses
                            else 0.0
                        )
                    )
                )
                self.logger.log_stat(
                    "loss_mask_parameter_relation",
                    mask_parameter_relation_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_mask_parameter_relation_bundle",
                    weighted_relation_bundle,
                    t_env,
                )
                self.logger.log_stat(
                    "mask_parameter_relation_to_td_ratio",
                    weighted_relation_bundle
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "mask_parameter_relation_pair_count",
                    float(mask_parameter_pair_count),
                    t_env,
                )
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
            if branch_drop_capturer is not None:
                for stat_name, stat_value in branch_drop_capturer.branch_drop_stats().items():
                    self.logger.log_stat(stat_name, float(stat_value.item()), t_env)
                branch_version = int(
                    branch_drop_capturer.branch_drop_version.item()
                )
                if branch_version != self.last_branch_drop_version_logged:
                    self.logger.console_logger.info(
                        "Dual branch DROP | t_env={} | {}".format(
                            t_env, branch_drop_capturer.branch_drop_summary()
                        )
                    )
                    self.last_branch_drop_version_logged = branch_version
                if semantic_router is None:
                    for stat_name, stat_value in self.latest_semantic_counterfactual_stats.items():
                        self.logger.log_stat(
                            stat_name, float(stat_value.item()), t_env
                        )
            if teacher_mac_out is not None:
                self.logger.log_stat("loss_teacher_td", teacher_td_loss.item(), t_env)
            if self.condition_gradient_consistency_active:
                weighted_condition_gradient_loss = (
                    condition_gradient_consistency_coef
                    * condition_gradient_consistency_loss.item()
                )
                self.logger.log_stat(
                    "loss_condition_grad_consistency",
                    condition_gradient_consistency_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_condition_grad_consistency",
                    weighted_condition_gradient_loss,
                    t_env,
                )
                self.logger.log_stat(
                    "condition_grad_consistency_to_td_ratio",
                    weighted_condition_gradient_loss
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "condition_grad_consistency_active",
                    float(condition_graphs is not None),
                    t_env,
                )
                for stat_name, stat_value in condition_gradient_consistency_stats.items():
                    self.logger.log_stat(stat_name, stat_value.item(), t_env)
            if self.generated_parameter_stability_active:
                weighted_parameter_stability_loss = (
                    generated_parameter_stability_coef
                    * generated_parameter_stability_loss.item()
                )
                self.logger.log_stat(
                    "loss_generated_parameter_stability",
                    generated_parameter_stability_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_generated_parameter_stability",
                    weighted_parameter_stability_loss,
                    t_env,
                )
                self.logger.log_stat(
                    "generated_parameter_stability_to_td_ratio",
                    weighted_parameter_stability_loss
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "generated_parameter_stability_active",
                    float(generated_parameter_stability_enabled),
                    t_env,
                )
            if self.generated_parameter_likelihood_active:
                weighted_parameter_likelihood_loss = (
                    generated_parameter_stability_coef
                    * generated_parameter_likelihood_loss.item()
                )
                self.logger.log_stat(
                    "loss_generated_parameter_conditional_nll",
                    generated_parameter_likelihood_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_generated_parameter_conditional_nll",
                    weighted_parameter_likelihood_loss,
                    t_env,
                )
                self.logger.log_stat(
                    "generated_parameter_conditional_nll_to_td_ratio",
                    weighted_parameter_likelihood_loss
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "generated_parameter_conditional_nll_active",
                    float(generated_parameter_likelihood_enabled),
                    t_env,
                )
            if self.td_weighted_parameter_likelihood_active:
                self.logger.log_stat(
                    "loss_td_weighted_parameter_likelihood",
                    td_weighted_parameter_likelihood_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_td_parameter_likelihood",
                    self.td_weighted_parameter_likelihood_coef
                    * td_weighted_parameter_likelihood_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "td_parameter_quality_advantage_std",
                    td_parameter_quality_advantage_std.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "td_parameter_mean_log_prob",
                    td_parameter_mean_log_prob.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "td_weighted_parameter_likelihood_active",
                    float(td_weighted_parameter_likelihood_enabled),
                    t_env,
                )
            if self.trajectory_parameter_likelihood_active:
                weighted_trajectory_parameter_loss = (
                    trajectory_parameter_likelihood_coef
                    * trajectory_parameter_likelihood_loss.item()
                )
                self.logger.log_stat(
                    "loss_trajectory_parameter_likelihood",
                    trajectory_parameter_likelihood_loss.item(),
                    t_env,
                )
                self.logger.log_stat(
                    "weighted_loss_trajectory_parameter_likelihood",
                    weighted_trajectory_parameter_loss,
                    t_env,
                )
                self.logger.log_stat(
                    "trajectory_parameter_likelihood_to_td_ratio",
                    weighted_trajectory_parameter_loss
                    / max(td_loss.item(), self.adaptive_auxiliary_eps),
                    t_env,
                )
                self.logger.log_stat(
                    "trajectory_parameter_likelihood_active",
                    float(trajectory_parameter_likelihood_enabled),
                    t_env,
                )
            for stat_name, stat_value in self.latest_adaptive_auxiliary_stats.items():
                self.logger.log_stat(stat_name, float(stat_value), t_env)
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
        if self.adaptive_auxiliary_ratio_active or self.kl80_random_drop_auxiliary:
            th.save(
                {
                    "ema_td": self.adaptive_auxiliary_ema_td,
                    "ema_aux": self.adaptive_auxiliary_ema_aux,
                },
                "{}/adaptive_auxiliary_ema.th".format(path),
            )
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
        adaptive_ema_path = "{}/adaptive_auxiliary_ema.th".format(path)
        if (self.adaptive_auxiliary_ratio_active or self.kl80_random_drop_auxiliary) and os.path.exists(adaptive_ema_path):
            adaptive_state = th.load(
                adaptive_ema_path, map_location=lambda storage, loc: storage
            )
            self.adaptive_auxiliary_ema_td = adaptive_state.get("ema_td")
            self.adaptive_auxiliary_ema_aux = adaptive_state.get("ema_aux")
