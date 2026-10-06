#!/usr/bin/env python3
"""Simulator-free checks of masks, MC returns, derivatives and state isolation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics, episode_value_samples
from utils.logging import Logger


class Batch(dict):
    batch_size = 2
    max_seq_length = 4


def fixture():
    # First episode naturally terminates after two transitions. Second is
    # time-limit truncated after three. Last filled observation has no reward.
    return Batch(
        obs=th.zeros(2, 4, 2, 1), state=th.zeros(2, 4, 1),
        reward=th.tensor([[[1.], [2.], [999.], [0.]], [[1.], [2.], [4.], [0.]]]),
        terminated=th.tensor([[[0], [1], [0], [0]], [[0], [0], [0], [0]]]),
        filled=th.tensor([[[1], [1], [1], [0]], [[1], [1], [1], [1]]]),
        actions=th.zeros(2, 4, 2, 1, dtype=th.long),
        avail_actions=th.ones(2, 4, 2, 2, dtype=th.long),
    )


class SumMixer(th.nn.Module):
    def forward(self, q, states):
        return q.sum(-1, keepdim=True)


def main():
    th.set_num_threads(1)
    batch = fixture()
    summary = ValueDiagnosticSummary()
    episode_value_samples(th.ones(2, 3, 2, 2), SumMixer(), batch, .5, summary)
    def mean(name):
        n, s, _, _ = summary.moments[name]
        return s / n
    assert summary.moments['q_tot'][0] == 5
    assert mean('q_tot') == 2 and mean('mixer_dqtot_dqi') == 1
    assert mean('episode_discounted_return') == 3
    assert mean('q_mc_bias') == 0 and mean('q_mc_abs_error') == 0
    assert mean('naturally_terminal_episode') == .5
    vdn_summary = ValueDiagnosticSummary()
    episode_value_samples(th.ones(2, 3, 2, 2), VDNMixer(), batch, .5, vdn_summary)
    assert vdn_summary.moments == summary.moments
    assert 'mixer_w1' not in vdn_summary.moments
    # Inserting a dead agent must only change its utility count, not Q_tot.
    batch['avail_actions'][0, 0, 1, 1] = 0
    masked = ValueDiagnosticSummary()
    episode_value_samples(th.ones(2, 3, 2, 2), SumMixer(), batch, .5, masked)
    assert masked.moments['agent_q'][0] == 9
    assert masked.moments['agent_1_q'][0] == 4
    args = SimpleNamespace(n_agents=2, state_shape=1, mixing_embed_dim=2, hypernet_layers=1)
    mixer = QMixer(args)
    for parameter in mixer.parameters():
        parameter.data.zero_()
    mixer.hyper_w1.bias.data.fill_(2)
    mixer.hyper_w_final.bias.data.fill_(3)
    computed = ValueDiagnosticSummary()
    episode_value_samples(th.ones(2, 3, 2, 2), mixer, batch, .5, computed)
    assert computed.moments['mixer_w1'][1] / computed.moments['mixer_w1'][0] == 2
    assert computed.moments['mixer_dqtot_dqi'][1] / computed.moments['mixer_dqtot_dqi'][0] == 12
    assert all(p.grad is None for p in mixer.parameters())
    # Exercise the real minimal W&B filter and buffered upload path without
    # network access or a W&B run.
    import logging
    logger = Logger(logging.getLogger('value-diagnostics-test'))
    logger.use_wandb = True
    logger.wandb_current_t = 100
    logger.wandb_current_data = {}
    summary.log(logger, 100)
    assert 'test_value/q_tot_mean' in logger.wandb_current_data
    assert 'test_value/q_mc_bias_mean' in logger.wandb_current_data
    assert all(Logger._wandb_metric_allowed(k) for k in logger.wandb_current_data)

    class MAC:
        agent = th.nn.Sequential(th.nn.Linear(1, 1), th.nn.Dropout(.5))
        hidden_states = th.tensor([17.])
        def init_hidden(self, size):
            self.hidden_states = th.zeros(size)
        def forward(self, batch, t, test_mode):
            self.agent.eval()
            self.hidden_states += 1
            # Deliberately consume RNG to check diagnostics restore it.
            return th.rand(2, 2, 2)

    mac = MAC()
    previous = mac.hidden_states
    rng = th.get_rng_state().clone()
    collect_value_diagnostics(mac, mixer, batch, .5, ValueDiagnosticSummary())
    assert mac.hidden_states is previous and previous.item() == 17
    assert mac.agent.training and mac.agent[1].training
    assert th.equal(rng, th.get_rng_state())
    assert all(p.grad is None for p in mixer.parameters())
    print('PASS: masks/dead agents/truncations/MC/weights/derivatives/RNG/hidden/grad isolation/W&B filter')


if __name__ == '__main__':
    main()
