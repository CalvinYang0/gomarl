"""Gate-only regularizers: never backpropagate variance attribution into Q."""
import torch as th


@th.no_grad()
def gate_variance_coefficients(mixer, agent_qs, states, valid, budget=256):
    """Signed population Var(Q_tot(p*s))-Var(Q_tot(p*s with field j zero)).

    Fixed executed-action Q, fixed probes and deterministic soft gate p, not a
    noisy training gate or the discontinuous evaluation mask. The caller uses
    positive removal gains as detached gate penalties. No MC-return claim.
    """
    if mixer.variant != "static_gate" or budget < 2:
        raise ValueError("Variance gate probes require static_gate and budget >= 2")
    indices = valid.reshape(-1).bool().nonzero().flatten()
    if len(indices) < 2:
        return None
    selected = indices[th.linspace(0, len(indices) - 1, min(budget, len(indices)), device=indices.device).long()]
    q = agent_qs.detach().reshape(-1, agent_qs.shape[-1])[selected].unsqueeze(1)
    s = states.detach().reshape(-1, mixer.state_dim)[selected].unsqueeze(1)
    mask = mixer.gate_logits.sigmoid().detach().expand_as(s)
    original = mixer.conditioned_value(q, s, 0, mask).reshape(-1).var(unbiased=False)
    masked = s.new_zeros(mixer.state_dim)
    for j in range(mixer.state_dim):
        off = mask.clone()
        off[..., j] = 0
        masked[j] = mixer.conditioned_value(q, s, 0, off).reshape(-1).var(unbiased=False)
    difference = original - masked
    if not th.isfinite(difference).all():
        raise RuntimeError("Non-finite gate variance attribution")
    return difference, original.expand_as(masked), masked, len(selected)


def gate_penalties(mixer, l1_coef, variance_coef):
    """Only gate logits receive these gradients. Higher c_j => stronger closure."""
    zero = next(mixer.parameters()).new_zeros(())
    if l1_coef == 0 and variance_coef == 0:
        return zero, zero
    if mixer.variant != "static_gate":
        raise ValueError("Gate penalties require static_gate")
    p = mixer.gate_logits.sigmoid()
    l1 = p.mean() if l1_coef else zero
    variance = (p * mixer.gate_variance_difference.detach().clamp(min=0)).mean() if variance_coef else zero
    return l1, variance
