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
    if env.get("map_name") == "5m_vs_6m" and not env.get("obs_instead_of_state", False):
        names = ["ally_{}_{}".format(i, field) for i in range(5)
                 for field in ("health", "cooldown", "x", "y")]
        names += ["enemy_{}_{}".format(i, field) for i in range(6) for field in ("health", "x", "y")]
        if env.get("state_last_action", True):
            names += ["ally_{}_last_action_{}".format(i, a) for i in range(5) for a in range(args.n_actions)]
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


def log_state_snapshot(mixer, args, logger, t_env, updates, migration=None, images=True):
    names = state_slot_names(args, mixer.state_dim)
    if mixer.grouped:
        rows = th.stack((mixer.important_slots, ~mixer.important_slots)).float().cpu()
        labels = ["important membership", "unimportant membership"]
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
                  state_slot_names=names, row_labels=labels, masks_or_probability=rows.tolist(),
                  qchange=mixer.qchange_scores.cpu().tolist(), tderror=mixer.tderror_scores.cpu().tolist(),
                  ranking_ema=mixer.score_ema.cpu().tolist(), migration=migration,
                  note="Fixed agent Q and fixed own-branch TD target; not action flips or win-rate attribution")
    path = directory / (stem + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(names) * .20), 7), constrained_layout=True)
    try:
        pixel = axes[0].imshow(rows.numpy(), aspect="auto", vmin=0, vmax=1, cmap="Reds", interpolation="nearest")
        axes[0].set_yticks([0, 1], labels=labels)
        fig.colorbar(pixel, ax=axes[0])
        axes[1].bar(range(len(names)), result["qchange"])
        axes[1].set_ylabel("mean abs Q_tot change")
        axes[2].bar(range(len(names)), result["tderror"])
        axes[2].set_ylabel("positive mean TD error increase")
        for axis in axes:
            axis.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=6)
        title = "{} / {} / seed={} / t_env={}".format(args.env_args.get("map_name"), mixer.variant, args.seed, t_env)
        if run_name.startswith("SYNTHETIC"):
            title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
        fig.suptitle(title)
        image_path = directory / (stem + ".png")
        fig.savefig(image_path, dpi=130)
        if logger.use_wandb:
            logger._update_wandb_buffer("mixer_input/partition_probability_heatmap",
                                      logger.wandb_module.Image(str(image_path)), t_env)
            logger.wandb_module.save(str(path), base_path=str(directory))
    finally:
        plt.close(fig)
    return path
