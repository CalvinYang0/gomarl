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
    # Pure Transformer-only individual network.  The only intervention is an
    # auxiliary Binary-Concrete KL80 drop over agent utilities before QMIX.
    "mixer_kl80aux": {"mixer_aux": "kl80"},
    "mixer_kl80aux_coef01": {
        "mixer_aux": "kl80", "mixer_aux_coef": 0.1,
    },
    "mixer_kl80aux_coef001": {
        "mixer_aux": "kl80", "mixer_aux_coef": 0.01,
    },
    "obs_gate_kl80aux_mixer_coef001": {
        "gate": True,
        "aux": "kl80",
        "mixer_aux": "kl80",
        "mixer_aux_coef": 0.01,
    },
    "relation_kl90aux": {
        "gate": True, "relation": True, "aux": "kl80", "aux_prior": 0.9,
    },
    "relation_kl80aux_mixer": {
        "gate": True, "relation": True, "aux": "kl80", "mixer_aux": "kl80",
    },
    "relation_kl80aux_adjrand": {
        "gate": True,
        "relation": True,
        "aux": "kl80",
        "relation_pairing": "adjacent_random",
    },
    "relation_kl80aux_centered_product": {
        "gate": True,
        "relation": True,
        "aux": "kl80",
        "relation_objective": "centered_product",
    },
    "relation_random80": {"gate": True, "relation": True, "aux": "fixed_concrete"},
    # "kl80" is the legacy auxiliary implementation kind; the prior is separate.
    "relation_kl50aux": {"gate": True, "relation": True, "aux": "kl80", "aux_prior": 0.5},
    "relation_kl30aux": {"gate": True, "relation": True, "aux": "kl80", "aux_prior": 0.3},
    "linear_relation_kl80aux": {"gate": True, "relation": True, "aux": "kl80", "branch": "linear"},
    # Auxiliary KL80 corruption is applied before the obs-conditioned main
    # gate, so the latter learns from the already-masked observation.
    "relation_kl80aux_klfirst": {
        "gate": True,
        "relation": True,
        "aux": "kl80",
        "aux_order": "kl_first",
    },
    # Keep training unchanged, but use the learned keep probability as a
    # deterministic soft mask during test execution instead of p>0.5.
    "relation_kl80aux_softtest": {
        "gate": True,
        "relation": True,
        "aux": "kl80",
        "test_soft_gate": True,
    },
    # Matched hypernetwork baselines.  They share the recurrent policy encoder,
    # generated two-layer Q head and QMIX learner; only the hypernetwork
    # condition source changes.
    "hyper_hypermarl_id": {"hyper_condition": "agent_id"},
    "hyper_cash_obs_type": {"hyper_condition": "obs_agent_type"},
    "hyper_rpg_relation": {"hyper_condition": "rpg_relation"},
}
ALL_PROFILES = dict(PROFILES, **ABLATION_PROFILES)

# Variants whose observation-gate semantics have an explicit SMAC adapter and
# simulator-free regression coverage.  Keep this list narrow: accepting an
# arbitrary GRF ablation here can silently apply the wrong entity layout.
SMAC_PROFILES = (
    "baseline",
    "relation_kl80aux",
    "obs_gate_kl80aux",
    "relation_kl80aux_mixer",
    "obs_gate_kl80aux_mixer_coef001",
)


def model_type_for(label, domain="grf"):
    if domain not in {"grf", "smac"}:
        raise ValueError("Unknown suite domain: " + domain)
    if domain == "smac":
        if label not in SMAC_PROFILES:
            raise ValueError("SMAC suite currently supports " + ", ".join(SMAC_PROFILES))
        return "smac_single_transformer_suite_{}_hypercond".format(label)
    branch = ALL_PROFILES[label].get("branch", "transformer")
    variant = label[len("linear_"):] if branch == "linear" else label
    return "grf_abs_single_{}_suite_{}_hypercond".format(branch, variant)


MODEL_PROFILES = {
    model_type_for(label): dict(flags, label=label)
    for label, flags in ALL_PROFILES.items()
}
MODEL_PROFILES.update({
    model_type_for(label, "smac"): dict(ALL_PROFILES[label], label=label, domain="smac")
    for label in SMAC_PROFILES
})


def profile_for(model_type):
    return MODEL_PROFILES.get(model_type, {})


def kl_auxiliary_tag(prior):
    if not 0.0 < prior < 1.0:
        raise ValueError("KL auxiliary keep prior must lie strictly between 0 and 1")
    return "kl{:g}".format(prior * 100)


def experiment_overrides(label, domain="grf"):
    """Shared by actual submission and runtime tests; no model-name guessing."""
    flags = ALL_PROFILES[label]
    overrides = {
        "clean_model_type": model_type_for(label, domain),
        "clean_mask_parameter_relation_coef": float(bool(flags.get("relation"))),
        "clean_mask_parameter_relation_pairing": flags.get(
            "relation_pairing", "fixed"
        ),
        "clean_mask_parameter_relation_objective": flags.get(
            "relation_objective", "l1"
        ),
        "clean_mask_parameter_relation_mask_source": "probability",
        "clean_mask_parameter_relation_temporal_coef": float(bool(flags.get("temporal"))),
        "clean_mask_parameter_relation_perturbed_head_coef": 0.0,
        "clean_mask_parameter_relation_gate_regularization_coef": 0.0,
        "clean_random_drop_auxiliary_coef": float(bool(flags.get("aux"))),
        "clean_mixer_kl80_auxiliary_coef": flags.get(
            "mixer_aux_coef", float(bool(flags.get("mixer_aux")))
        ),
        "clean_mixer_kl80_prior": 0.8,
        "clean_mixer_kl80_temperature": 0.5,
        "clean_kl_auxiliary_prior": flags.get("aux_prior", 0.8),
        "clean_random_drop_auxiliary_keep_probability": (
            0.8 if flags.get("aux") == "fixed_concrete" else 0.5
        ),
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
    if flags.get("hyper_condition") == "agent_id":
        overrides["clean_apply_hypermarl_init"] = True
    return overrides
