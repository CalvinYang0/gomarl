"""Detached attribution targets and gate-only probability supervision."""
import torch as th
import torch.nn.functional as F


@th.no_grad()
def gate_score_targets(mixer, agent_qs, states, valid, budget=256,
                      criterion="qchange", returns=None, reference_valid=None):
    """Fixed soft p, Q, model and samples; batch-mean replacement, not mask edits.

    Q-change targets: score / max(score), skipping an all-zero/no-signal batch.
    MC targets: sigmoid(-normalized MSE replacement benefit), same quality
    direction as calibration grouping. MC valid selects complete returns only;
    reference_valid still selects all valid TD transitions for the field mean.
    No graph from these measurements reaches gate, mixer, actor or return.
    """
    if mixer.variant != "static_gate" or criterion not in {"qchange", "calibration"}:
        raise ValueError("Gate supervision requires static_gate and a known score")
    if budget < 1:
        raise ValueError("Gate supervision requires a positive probe budget")
    indices = valid.reshape(-1).bool().nonzero().flatten()
    if not indices.numel():
        return None
    reference_valid = valid if reference_valid is None else reference_valid
    if reference_valid.shape != valid.shape or (valid.bool() & ~reference_valid.bool()).any():
        raise ValueError("Gate probes must be a subset of valid reference samples")
    reference_indices = reference_valid.reshape(-1).bool().nonzero().flatten()
    selected = indices[th.linspace(0, len(indices) - 1, min(budget, len(indices)), device=indices.device).long()]
    q = agent_qs.detach().reshape(-1, agent_qs.shape[-1])[selected].unsqueeze(1)
    s = states.detach().reshape(-1, mixer.state_dim)[selected].unsqueeze(1)
    reference = states.detach().reshape(-1, mixer.state_dim)[reference_indices].mean(0)
    probability = mixer.gate_logits.sigmoid().detach()
    mask = probability.expand_as(s)
    base = mixer.conditioned_value(q, s, 0, mask)
    raw = s.new_zeros(mixer.state_dim)
    original_mse, perturbed_mse = raw.clone(), raw.clone()
    g = scale = None
    if criterion == "calibration":
        if returns is None:
            raise ValueError("MC gate supervision requires complete observed returns")
        g = returns.detach().reshape(-1, 1)[selected].unsqueeze(1)
        scale = g.square().mean().clamp(min=1e-6)
        original_mse.fill_((base - g).square().mean())
    for j in range(mixer.state_dim):
        altered = s.clone()
        altered[..., j] = reference[j]
        changed = mixer.conditioned_value(q, altered, 0, mask)
        if criterion == "qchange":
            raw[j] = (base - changed).abs().mean()
        else:
            perturbed_mse[j] = (changed - g).square().mean()
            raw[j] = (original_mse[j] - perturbed_mse[j]) / scale
    if not th.isfinite(raw).all():
        raise RuntimeError("Non-finite gate supervision scores")
    if criterion == "qchange":
        peak = raw.max()
        if peak <= 1e-8:
            return None  # No evidence, not an instruction to close every gate.
        targets = (raw / peak).clamp(0, 1)
    else:
        targets = th.sigmoid(-raw.clamp(-20, 20))
    return dict(raw_scores=raw, targets=targets, count=len(selected),
                probability_at_probe=probability,
                original_mc_mse=original_mse, perturbed_mc_mse=perturbed_mse)


def gate_supervision_loss(gate_logits, targets):
    if gate_logits.shape != targets.shape or not th.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("Invalid gate probability targets")
    return F.binary_cross_entropy_with_logits(gate_logits, targets.detach())
