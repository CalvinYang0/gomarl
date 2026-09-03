"""Explicit feature matrix for the nine Counter Transformer experiments.

Kept dependency-free so submission preflights and learner/agent use one source
of truth rather than separate, easily divergent model whitelists.
"""

PROFILES = {
    "baseline": {},
    "relation": {"gate": True, "relation": True},
    "relation_temporal": {"gate": True, "relation": True, "temporal": True},
    "relation_random50": {"gate": True, "relation": True, "aux": "bernoulli"},
    "relation_temporal_random50": {
        "gate": True, "relation": True, "temporal": True, "aux": "bernoulli",
    },
    "kl80": {"gate": True, "kl": True},
    "relation_kl80aux": {"gate": True, "relation": True, "aux": "kl80"},
    "relation_temporal_kl80aux": {
        "gate": True, "relation": True, "temporal": True, "aux": "kl80",
    },
    "kl80_test_open": {"gate": True, "kl": True, "test_open": True},
}

# Keep the original nine-model submission unchanged. These controls are
# opt-in and submitted separately, without cancelling the original runs.
ABLATION_PROFILES = {
    "obs_gate_kl80aux": {"gate": True, "aux": "kl80"},
}
ALL_PROFILES = dict(PROFILES, **ABLATION_PROFILES)

MODEL_PROFILES = {
    "grf_abs_single_transformer_suite_{}_hypercond".format(label): dict(flags, label=label)
    for label, flags in ALL_PROFILES.items()
}


def profile_for(model_type):
    return MODEL_PROFILES.get(model_type, {})


def experiment_overrides(label):
    """Shared by actual submission and runtime tests; no model-name guessing."""
    flags = ALL_PROFILES[label]
    return {
        "clean_model_type": "grf_abs_single_transformer_suite_{}_hypercond".format(label),
        "clean_mask_parameter_relation_coef": float(bool(flags.get("relation"))),
        "clean_mask_parameter_relation_pairing": "fixed",
        "clean_mask_parameter_relation_mask_source": "probability",
        "clean_mask_parameter_relation_temporal_coef": float(bool(flags.get("temporal"))),
        "clean_mask_parameter_relation_perturbed_head_coef": 0.0,
        "clean_mask_parameter_relation_gate_regularization_coef": 0.0,
        "clean_random_drop_auxiliary_coef": float(bool(flags.get("aux"))),
        "clean_random_drop_auxiliary_keep_probability": 0.5,
        "clean_random_drop_auxiliary_scope": "timestep",
        "clean_random_drop_auxiliary_combine_mode": "multiply",
        "clean_hard_gate_initial_keep_probability": 0.95,
        "clean_binary_concrete_temperature": 0.5,
        "clean_dynamic_branch_gate_warmup_steps": 250000,
        "clean_importance_auxiliary_warmup_steps": 250000,
        "clean_importance_alternating_training": False,
        "clean_relation_teacher_td_coef": 0.0,
        "clean_relation_distill_coef": 0.0,
        "clean_smooth_head_loss_coef": 0.0,
        "clean_action_pred_loss_coef": 0.0,
        "clean_public_delta_loss_coef": 0.0,
        "clean_condition_gradient_consistency_coef": 0.0,
        "clean_generated_parameter_stability_coef": 0.0,
        "clean_td_weighted_parameter_likelihood_coef": 0.0,
        "clean_test_gate_trajectory_max_steps": 1000,
        "wandb_test_parameter_pca": True,
    }
