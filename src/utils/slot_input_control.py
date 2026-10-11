"""Training-replay slot attribution and auditable global partition snapshots.

Finite perturbations of the hyper-input only. Hidden state/action availability
stay fixed. This is a decision-dependence proxy, not a win-rate attribution.
"""
from hashlib import sha256
import json
from pathlib import Path

import torch as th

from utils.hyper_obs_importance import replay_samples


def action_margin(q, available, chosen=None):
    ranked = q.masked_fill(~available.bool(), -th.inf)
    if chosen is None:
        chosen = ranked.argmax(-1)
    selected = q.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    competitors = ranked.scatter(-1, chosen.unsqueeze(-1), -th.inf).max(-1).values
    return selected - competitors, chosen


@th.no_grad()
def score_slots(mac, batch, budget=256, episodes=10):
    agent = mac.agent
    samples = replay_samples(mac, batch, min(episodes, batch.batch_size))
    if samples is None:
        return None
    selected = []
    cap = max(1, budget // mac.n_agents)
    for i in range(mac.n_agents):
        candidates = (samples["agent"] == i).nonzero().flatten()
        if candidates.numel():
            positions = th.linspace(0, candidates.numel() - 1, min(cap, candidates.numel())).long()
            selected.extend(candidates[positions].tolist())
    if not selected:
        return None
    device = next(agent.parameters()).device
    obs, hidden, avail = (samples[k][selected].to(device) for k in ("obs", "hidden", "avail"))
    if not th.isfinite(obs).all() or not th.isfinite(hidden).all():
        raise RuntimeError("Non-finite training probes")
    full_obs = samples["obs"].to(device)
    reference = full_obs.mean(0)
    names = agent.rpg_relation_capturer.semantic_names
    # Do not fill unobserved entities with a population health/geometry mean.
    visible_columns = {}
    for j, name in enumerate(names):
        if name.startswith(("enemy_", "ally_")) and name.endswith(("_visible", "_visible_or_attackable")):
            visible_columns["_".join(name.split("_")[:2])] = j
    active_rows = []
    for j, name in enumerate(names):
        prefix = "_".join(name.split("_")[:2])
        visible = visible_columns.get(prefix)
        rows = obs[:, visible] > 0 if visible is not None and visible != j else th.ones(len(obs), dtype=th.bool, device=device)
        active_rows.append(rows)
        if visible is not None and visible != j:
            population = full_obs[:, visible] > 0
            if population.any():
                reference[j] = full_obs[population, j].mean()
    masks = agent.group_masks(obs) if agent.is_grouped else (
        th.sigmoid(agent.rpg_relation_capturer.dynamic_branch_gate.static_logits) > .1).to(obs).unsqueeze(0)
    scores, flips = obs.new_zeros(agent.obs_dim), obs.new_zeros(agent.obs_dim)
    for group in range(len(masks)):
        q, _, _ = agent.head_q(obs * masks[group], hidden, group)
        margin, chosen = action_margin(q, avail)
        # One common pre-perturbation scale per owning head/probe population.
        # Centering cancels additive utility offsets; a common positive
        # rescaling of this head also rescales this denominator.
        count = avail.sum(-1).clamp(min=1)
        mean = q.masked_fill(~avail, 0).sum(-1) / count
        centered = (q - mean.unsqueeze(-1))[avail]
        scale = centered.square().mean().sqrt()
        if not th.isfinite(q).all() or not th.isfinite(scale):
            raise RuntimeError("Non-finite attribution Q/scale")
        if scale < 1e-6:
            continue
        columns = masks[group].nonzero().flatten().tolist()
        for j in columns:
            altered = obs.clone()
            altered[active_rows[j], j] = reference[j]
            changed, _, _ = agent.head_q(altered * masks[group], hidden, group)
            new_margin, _ = action_margin(changed, avail, chosen)
            scores[j] = (margin - new_margin).abs().mean() / scale
            flips[j] = (changed.masked_fill(~avail, -th.inf).argmax(-1) != chosen).float().mean()
    if not th.isfinite(scores).all():
        raise RuntimeError("Non-finite slot importance")
    return scores, flips, len(selected)


def log_slot_snapshot(agent, args, logger, t_env, train_steps, migration=None, images=True):
    """Persist both probabilities and exact evaluation masks, or both groups."""
    names = list(agent.rpg_relation_capturer.semantic_names)
    if agent.is_grouped:
        rows = agent.group_masks(agent.slot_score_ema).detach().cpu()
        labels = ["important membership", "unimportant membership"]
    else:
        probability = th.sigmoid(agent.rpg_relation_capturer.dynamic_branch_gate.static_logits).detach().cpu()
        rows = th.stack((probability, (probability > .1).float()))
        labels = ["learned keep probability", "test mask: p > 0.1"]
    for j, name in enumerate(names):
        logger.log_stat("slot_input/importance/" + name, agent.slot_score_ema[j].item(), t_env)
    logger.log_stat("slot_input/important_count", int(agent.important_slots.sum()) if agent.is_grouped else int(rows[1].sum()), t_env)
    logger.log_stat("slot_input/train_updates", train_steps, t_env)
    run_name = getattr(args, "wandb_run_name", None) or args.name
    identity = sha256(run_name.encode()).hexdigest()[:8]
    directory = Path(args.local_results_path) / "slot_input_control" / args.unique_token / (
        "seed_{}_{}".format(args.seed, identity))
    directory.mkdir(parents=True, exist_ok=True)
    stem = "step_{:09d}_update_{:07d}".format(int(t_env), int(train_steps))
    result = dict(t_env=int(t_env), train_updates=train_steps, variant=agent.slot_variant,
                  slot_names=names, row_labels=labels, masks_or_probability=rows.tolist(),
                  score=agent.slot_score_ema.detach().cpu().tolist(),
                  flip_rate=agent.slot_flip_rate.detach().cpu().tolist(), migration=migration,
                  note="Hyper-input dependence only; hidden/action mask fixed; not win-rate importance")
    path = directory / (stem + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    if not images:
        return path
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(names) * .23), 7), constrained_layout=True)
    try:
        pixel = axes[0].imshow(rows.numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Reds")
        axes[0].set_yticks([0, 1], labels=labels)
        fig.colorbar(pixel, ax=axes[0], label="probability / binary membership")
        axes[1].bar(range(len(names)), result["score"])
        axes[1].set_ylabel("EMA normalized margin change")
        axes[2].bar(range(len(names)), result["flip_rate"])
        axes[2].set_ylabel("greedy action flip fraction")
        for axis in axes:
            axis.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=6)
        title = "{} seed={} t_env={} train updates={} {}".format(args.env_args["map_name"], args.seed, t_env, train_steps, agent.slot_variant)
        if run_name.startswith("SYNTHETIC"):
            title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
        fig.suptitle(title)
        image_path = directory / (stem + ".png")
        fig.savefig(image_path, dpi=130)
        if logger.use_wandb:
            logger._update_wandb_buffer("slot_input/partition_probability_heatmap",
                                      logger.wandb_module.Image(str(image_path)), t_env)
            logger.wandb_module.save(str(path), base_path=str(directory))
        return path
    finally:
        plt.close(fig)
