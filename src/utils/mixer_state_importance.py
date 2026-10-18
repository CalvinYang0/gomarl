"""Finite state perturbations: fixed agent utilities and fixed TD targets.

These scores measure current mixer dependence, not action flips or win-rate
contribution. Mean replacement can be off-manifold for correlated state fields.
"""
from hashlib import sha256
import json
from pathlib import Path

import torch as th


def state_slot_names(args, state_dim):
    env = args.env_args
    marine_maps = {"5m_vs_6m": (5, 6), "8m_vs_9m": (8, 9)}
    if env.get("map_name") in marine_maps and not env.get("obs_instead_of_state", False):
        allies, enemies = marine_maps[env["map_name"]]
        names = ["ally_{}_{}".format(i, field) for i in range(allies)
                 for field in ("health", "cooldown", "x", "y")]
        names += ["enemy_{}_{}".format(i, field) for i in range(enemies) for field in ("health", "x", "y")]
        if env.get("state_last_action", True):
            names += ["ally_{}_last_action_{}".format(i, a) for i in range(allies) for a in range(args.n_actions)]
        if env.get("state_timestep_number", False):
            names += ["episode_timestep"]
        if len(names) != state_dim:
            raise ValueError("5m6m state layout mismatch: {} names vs {} fields".format(len(names), state_dim))
        return names
    return ["state_{}".format(j) for j in range(state_dim)]


@th.no_grad()
def score_state_slots(mixer, agent_qs, states, targets, valid, budget=256):
    indices = valid.reshape(-1).bool().nonzero().flatten()
    if not indices.numel():
        return None
    selected = indices[th.linspace(0, len(indices) - 1, min(budget, len(indices)), device=indices.device).long()]
    q = agent_qs.detach().reshape(-1, agent_qs.shape[-1])[selected].unsqueeze(1)
    s = states.detach().reshape(-1, mixer.state_dim)[selected].unsqueeze(1)
    y = targets.detach().reshape(-1, targets.shape[-1])[selected].unsqueeze(1)
    reference = states.detach().reshape(-1, mixer.state_dim)[indices].mean(0)
    masks = mixer.masks(s, "eval")
    qchange, tderror = s.new_zeros(mixer.state_dim), s.new_zeros(mixer.state_dim)
    for group in range(len(mixer.mixers)):
        mask = masks[group]
        base = mixer.conditioned_value(q, s, group, mask)
        target = y[..., group:group + 1]
        for j in range(mixer.state_dim):
            if not mask[..., j].bool().any():
                continue
            altered = s.clone()
            altered[..., j] = reference[j]
            changed = mixer.conditioned_value(q, altered, group, mask)
            qchange[j] = (changed - base).abs().mean()
            # Clip AFTER averaging: positive and negative effects may cancel.
            tderror[j] = ((target - changed).square() - (target - base).square()).mean().clamp(min=0)
    if not th.isfinite(qchange).all() or not th.isfinite(tderror).all():
        raise RuntimeError("Non-finite mixer importance scores")
    return qchange, tderror, len(selected)


@th.no_grad()
def score_state_slot_calibration(mixer, agent_qs, states, returns, valid, budget=256,
                                 perturbation="mask_zero", reference_valid=None):
    """Masking benefit = (MSE(Q,G)-MSE(Q_masked,G))/max(E[G^2],1e-6).

    G is an observed complete-episode discounted return, not true optimal Q.
    Agent Q/actions/checkpoint and G remain fixed; probe the owning branch.
    Legacy mask_zero removes both slot value and membership bit. batch_mean
    replaces only the field value, keeping membership fixed like Q-change;
    reference_valid selects the original Q-change reference population.
    Positive means replacement/removal improves calibration: lower quality.
    """
    if perturbation not in {"mask_zero", "batch_mean"}:
        raise ValueError("Unknown calibration perturbation")
    indices = valid.reshape(-1).bool().nonzero().flatten()
    if not indices.numel():
        return None
    if budget < 1:
        raise ValueError("Calibration scoring requires positive probe budget")
    selected = indices[th.linspace(0, len(indices) - 1, min(budget, len(indices)),
                                  device=indices.device).long()]
    q = agent_qs.detach().reshape(-1, agent_qs.shape[-1])[selected].unsqueeze(1)
    s = states.detach().reshape(-1, mixer.state_dim)[selected].unsqueeze(1)
    g = returns.detach().reshape(-1, 1)[selected].unsqueeze(1)
    reference = None
    if perturbation == "batch_mean":
        reference_valid = valid if reference_valid is None else reference_valid
        if reference_valid.shape != valid.shape or (valid.bool() & ~reference_valid.bool()).any():
            raise ValueError("Calibration probes must be a subset of valid reference transitions")
        reference_indices = reference_valid.reshape(-1).bool().nonzero().flatten()
        reference = states.detach().reshape(-1, mixer.state_dim)[reference_indices].mean(0)
    scale = g.square().mean().clamp(min=1e-6)
    result = s.new_zeros(mixer.state_dim)
    original_error, masked_error = result.clone(), result.clone()
    masks = mixer.masks(s, "eval")
    for group, mask in enumerate(masks):
        base = mixer.conditioned_value(q, s, group, mask)
        base_error = (base - g).square().mean()
        for j in range(mixer.state_dim):
            if not mask[..., j].bool().any():
                continue
            if perturbation == "mask_zero":
                removed = mask.clone()
                removed[..., j] = 0
                changed = mixer.conditioned_value(q, s, group, removed)
            else:
                altered = s.clone()
                altered[..., j] = reference[j]
                changed = mixer.conditioned_value(q, altered, group, mask)
            original_error[j] = base_error
            masked_error[j] = (changed - g).square().mean()
            result[j] = (original_error[j] - masked_error[j]) / scale
    if not th.isfinite(result).all():
        raise RuntimeError("Non-finite calibration improvement scores")
    return result, len(selected), original_error, masked_error


@th.no_grad()
def score_state_slot_variance(mixer, agent_qs, states, valid, budget=256):
    """Var(Q_tot)-Var(Q_tot without slot), population variance (unbiased=False).

    Clean score-only ablation of score_state_slots: use the identical valid
    samples, fixed executed-action agent Q, owning branch, and batch-mean slot
    replacement. Membership bits stay fixed. No MC reference, normalization,
    absolute value, clipping, or sigmoid. Negative differences are retained.
    """
    indices = valid.reshape(-1).bool().nonzero().flatten()
    if budget < 2:
        raise ValueError("Variance scoring requires a probe budget of at least two")
    if indices.numel() < 2:
        return None
    selected = indices[th.linspace(0, len(indices) - 1, min(budget, len(indices)), device=indices.device).long()]
    q = agent_qs.detach().reshape(-1, agent_qs.shape[-1])[selected].unsqueeze(1)
    s = states.detach().reshape(-1, mixer.state_dim)[selected].unsqueeze(1)
    reference = states.detach().reshape(-1, mixer.state_dim)[indices].mean(0)
    difference = s.new_zeros(mixer.state_dim)
    original, masked = difference.clone(), difference.clone()
    for group, mask in enumerate(mixer.masks(s, "eval")):
        base = mixer.conditioned_value(q, s, group, mask)
        base_variance = base.reshape(-1).var(unbiased=False)
        for j in range(mixer.state_dim):
            if not mask[..., j].bool().any():
                continue
            altered = s.clone()
            altered[..., j] = reference[j]
            changed = mixer.conditioned_value(q, altered, group, mask)
            original[j] = base_variance
            masked[j] = changed.reshape(-1).var(unbiased=False)
            difference[j] = original[j] - masked[j]
    if not th.isfinite(difference).all():
        raise RuntimeError("Non-finite Q_tot variance differences")
    return difference, len(selected), original, masked


@th.no_grad()
def complete_episode_returns(rewards, terminated, valid, gamma):
    """No bootstrap: incomplete/time-limit-only episodes cannot define G."""
    returns = th.zeros_like(rewards)
    future = th.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        future = (rewards[:, t] + gamma * (1 - terminated[:, t]) * future) * valid[:, t]
        returns[:, t] = future
    complete = (terminated * valid).sum(dim=1, keepdim=True) > 0
    return returns, valid * complete.to(valid)


def log_state_snapshot(mixer, args, logger, t_env, updates, migration=None, images=True,
                       evaluated_partition=None):
    names = state_slot_names(args, mixer.state_dim)
    if mixer.grouped:
        rows = th.stack((mixer.important_slots, ~mixer.important_slots)).float().cpu()
        labels = ["important membership", "unimportant membership"]
    elif mixer.variant == "ones":
        rows = th.ones(2, mixer.state_dim)
        labels = ["constant ones condition", "constant ones (no gate)"]
    else:
        p = mixer.gate_logits.sigmoid().detach().cpu()
        rows = th.stack((p, (p > mixer.threshold).float()))
        labels = ["learned keep probability", "evaluation mask p > {}".format(mixer.threshold)]
    logger.log_stat("mixer_input/important_count", rows[0].sum().item() if mixer.grouped else rows[1].sum().item(), t_env)
    logger.log_stat("mixer_input/train_updates", updates, t_env)
    if not images:
        return None
    run_name = getattr(args, "wandb_run_name", None) or args.name
    identity = sha256(run_name.encode()).hexdigest()[:8]
    directory = Path(args.local_results_path) / "mixer_input_control" / args.unique_token / (
        "seed_{}_{}".format(args.seed, identity))
    directory.mkdir(parents=True, exist_ok=True)
    stem = "step_{:09d}_update_{:07d}".format(int(t_env), updates)
    result = dict(t_env=int(t_env), train_updates=updates, variant=mixer.variant,
                  unimportant_agent_gradient_detached=bool(getattr(args, "mixer_input_detach_unimportant", False)),
                  state_slot_names=names, row_labels=labels, masks_or_probability=rows.tolist(),
                  qchange=mixer.qchange_scores.cpu().tolist(), tderror=mixer.tderror_scores.cpu().tolist(),
                  ranking_ema=mixer.score_ema.cpu().tolist(), migration=migration,
                  append_membership_features=mixer.append_mask,
                  mixer_condition_dim=mixer.mixers[0].state_dim,
                  note="Fixed agent Q and fixed own-branch TD target; not action flips or win-rate attribution")
    result.update(gate_l1_coef=float(getattr(args, "mixer_gate_l1_coef", 0)),
                  gate_variance_coef=float(getattr(args, "mixer_gate_variance_coef", 0)))
    if mixer.gate_aux_score != "none":
        result.update(gate_aux_score=mixer.gate_aux_score, main_td_gate_detached=True,
                      gate_aux_probe_count=int(mixer.gate_aux_probe_count.item()),
                      gate_aux_probe_update=int(mixer.gate_aux_probe_update.item()),
                      gate_aux_updates=int(mixer.gate_aux_updates.item()),
                      gate_aux_score_available=bool(mixer.gate_aux_probe_count.item()),
                      gate_aux_raw_scores=mixer.gate_aux_raw_scores.cpu().tolist(),
                      gate_aux_targets=mixer.gate_aux_targets.cpu().tolist(),
                      gate_aux_probability_at_probe=mixer.gate_aux_probability_at_probe.cpu().tolist(),
                      gate_aux_original_mc_mse=mixer.gate_aux_original_mc_mse.cpu().tolist(),
                      gate_aux_perturbed_mc_mse=mixer.gate_aux_perturbed_mc_mse.cpu().tolist(),
                      gate_aux_interval=int(getattr(args, "mixer_gate_aux_interval", 100)),
                      gate_aux_coef=float(getattr(args, "mixer_gate_aux_coef", 1.0)),
                      gate_aux_definition="BCE(logits, detached targets); Q-change/max(Q-change) or "
                          "sigmoid(-normalized MC-MSE replacement benefit); fixed soft p, Q and mixer; "
                          "batch-mean field replacement; high quality => high keep target",
                      gate_aux_timing_note="Targets/probe probability precede gate update; top rows show current probability/hard evaluation mask",
                      gate_aux_probe_scope=("complete observed episodes" if mixer.gate_aux_score == "calibration"
                                            else "valid TD transitions"))
    if hasattr(mixer, "gate_variance_difference"):
        result.update(gate_variance_difference=mixer.gate_variance_difference.cpu().tolist(),
                      gate_variance_penalty_coefficients=mixer.gate_variance_difference.clamp(min=0).cpu().tolist(),
                      gate_original_variance=mixer.gate_original_variance.cpu().tolist(),
                      gate_masked_variance=mixer.gate_masked_variance.cpu().tolist(),
                      gate_variance_probe_count=int(mixer.gate_variance_probe_count.item()),
                      gate_variance_probe_update=int(mixer.gate_variance_probe_update.item()),
                      gate_variance_probability_at_probe=mixer.gate_variance_probability_at_probe.cpu().tolist(),
                      gate_variance_interval=int(getattr(args, "mixer_gate_variance_interval", 100)),
                      gate_variance_definition="Var(Q_tot(p*state))-Var(Q_tot(p*state with field j zero)); population variance; detached positive part weights p_j")
    if mixer.variant == "group_calibration":
        perturbation = mixer.calibration_perturbation
        result.update(normalized_removal_gain=mixer.removal_gain_scores.cpu().tolist(),
                      calibration_probe_count=int(mixer.calibration_probe_count.item()),
                      calibration_score_available=bool(mixer.calibration_probe_count.item()),
                      original_mc_mse=mixer.original_mc_mse.cpu().tolist(),
                      masked_mc_mse=mixer.masked_mc_mse.cpu().tolist(),
                      ranking_definition="slot quality = sigmoid(-normalized_removal_gain); high gain means remove",
                      calibration_perturbation=perturbation,
                      calibration_probe_scope="complete observed episodes only; padding excluded",
                      calibration_reference_scope=("all valid TD transitions; same reference as Q-change"
                                                   if perturbation == "batch_mean" else "not applicable"),
                      metric_definition="(MSE(Q,G)-MSE(Q_perturbed,G))/max(mean(G^2),1e-6); "
                          "G = complete observed return; perturbation=" + perturbation)
    if mixer.variant == "group_variance":
        result.update(qtot_variance_difference=mixer.variance_difference_scores.cpu().tolist(),
                      original_qtot_variance=mixer.original_qtot_variance.cpu().tolist(),
                      masked_qtot_variance=mixer.masked_qtot_variance.cpu().tolist(),
                      variance_probe_count=int(mixer.variance_probe_count.item()),
                      ranking_definition="EMA of signed Var(Q_tot)-Var(Q_tot_slot_replaced); high means more variance removed",
                      metric_definition="population variance, unbiased=False; fixed Q and samples; batch-mean replacement as Q-change baseline")
    if evaluated_partition is not None:
        result["score_important_membership"] = evaluated_partition.cpu().tolist()
        result["score_partition_note"] = "Scores evaluated before migration; membership rows show partition after migration"
    from utils.grouping_diagnostics import enrich_snapshot, log_group_details
    enrich_snapshot(result, args, logger)
    if mixer.grouped:
        from utils.mixer_group_visualization import prepare_group_score_snapshot
        result = prepare_group_score_snapshot(result)
    path = directory / (stem + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    image_path = directory / (stem + ".png")
    if mixer.grouped:
        from utils.mixer_group_visualization import render_group_score_snapshot
        render_group_score_snapshot(result, image_path)
        if logger.use_wandb:
            image = logger.wandb_module.Image(str(image_path))
            logger._update_wandb_buffer("mixer_input/partition_probability_heatmap", image, t_env)
            logger._update_wandb_buffer("mixer_input/grouping_scores_by_group", image, t_env)
            log_group_details(result, logger, "mixer_input", image, path)
            logger.wandb.summary.update({"mixer_input/grouping_score_t_env": t_env,
                "mixer_input/grouping_score_partition_note": result["score_partition_note"]})
            logger.wandb_module.save(str(path), base_path=str(directory))
        return path
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(names) * .20), 7), constrained_layout=True)
    try:
        pixel = axes[0].imshow(rows.numpy(), aspect="auto", vmin=0, vmax=1, cmap="Reds", interpolation="nearest")
        axes[0].set_yticks([0, 1], labels=labels)
        fig.colorbar(pixel, ax=axes[0])
        if mixer.gate_aux_score != "none":
            if result["gate_aux_score_available"]:
                axes[1].bar(range(len(names)), result["gate_aux_raw_scores"])
                axes[2].bar(range(len(names)), result["gate_aux_targets"], label="detached target", alpha=.65)
                axes[2].plot(range(len(names)), rows[0].tolist(), color="tab:orange", linewidth=1,
                             label="current keep probability")
                axes[2].legend(loc="upper right", fontsize=7)
            else:
                for axis in axes[1:]:
                    axis.text(.5, .5, "No valid supervision at last probe; gate update skipped",
                              ha="center", va="center", transform=axis.transAxes)
            axes[1].axhline(0, color="black", linewidth=.5)
            axes[1].set_ylabel("mean abs Q_tot change" if mixer.gate_aux_score == "qchange"
                               else "MC-MSE replacement benefit\npositive = lower quality")
            axes[2].set_ylabel("gate supervision probability")
            axes[2].set_ylim(0, 1)
        elif mixer.variant == "group_variance":
            axes[1].bar(range(len(names)), result["ranking_ema"])
            axes[1].set_ylabel("EMA variance-difference ranking")
            axes[2].bar(range(len(names)), result["qtot_variance_difference"])
            axes[2].set_ylabel("Var(Q_tot) - Var(Q_tot_replaced)")
            axes[1].axhline(0, color="black", linewidth=.5)
            axes[2].axhline(0, color="black", linewidth=.5)
        elif hasattr(mixer, "gate_variance_difference"):
            axes[1].bar(range(len(names)), result["gate_variance_difference"])
            axes[1].set_ylabel("signed variance-removal gain")
            axes[1].axhline(0, color="black", linewidth=.5)
            axes[2].bar(range(len(names)), result["gate_variance_penalty_coefficients"])
            axes[2].set_ylabel("detached gate penalty coefficient")
        else:
            axes[1].bar(range(len(names)), result["qchange"])
            axes[1].set_ylabel("mean abs Q_tot change")
            axes[2].bar(range(len(names)), result.get("normalized_removal_gain", result["tderror"]))
            axes[2].set_ylabel("normalized masking benefit" if mixer.variant == "group_calibration"
                               else "positive mean TD error increase")
        for axis in axes:
            axis.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=6)
        title = "{} / {} / seed={} / t_env={}".format(args.env_args.get("map_name"), mixer.variant, args.seed, t_env)
        if mixer.gate_aux_score != "none":
            title += " / gate-only " + mixer.gate_aux_score + " BCE; main TD gate detached"
            title += "\nAux scores measured at main update={}; current snapshot update={}".format(
                result["gate_aux_probe_update"], updates)
        if run_name.startswith("SYNTHETIC"):
            title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
        fig.suptitle(title)
        image_path = directory / (stem + ".png")
        fig.savefig(image_path, dpi=130)
        if logger.use_wandb:
            image = logger.wandb_module.Image(str(image_path))
            logger._update_wandb_buffer("mixer_input/partition_probability_heatmap",
                                      image, t_env)
            if mixer.gate_aux_score != "none":
                logger._update_wandb_buffer("mixer_input/gate_aux_heatmap", image, t_env)
            log_group_details(result, logger, "mixer_input", image, path)
            logger.wandb_module.save(str(path), base_path=str(directory))
    finally:
        plt.close(fig)
    return path
