#!/usr/bin/env python3
"""Real 8m_vs_9m shapes and production fixed-head learner, without SC2 launch."""
import torch as th

from smoke_test_counter_transformer_nine import make_case
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics


def main():
    th.set_num_threads(1)
    initial_agent = None
    for method, mixer_type in (("vdn", VDNMixer), ("qmix", QMixer)):
        th.manual_seed(29)
        mac, learner, batch, logger = make_case(
            "baseline", "8m_vs_9m", production_shapes=True,
            config_overrides={"clean_model_type": "qmix_minimal", "mixer": method},
        )
        assert mac.args.n_agents == 8 and mac.args.n_actions == 15
        assert mac.args.obs_shape == 85 and mac.args.state_shape == 179
        assert mac.agent.hidden_dim == 64 and mac.agent.fc1.in_features == 108
        assert isinstance(mac.agent.rnn, th.nn.GRUCell)
        assert mac.agent.model_type == "qmix_minimal"
        assert mac.agent.fixed_head is not None and mac.agent.hyper_out_w is None
        assert isinstance(learner.mixer, mixer_type) and isinstance(learner.target_mixer, mixer_type)
        assert learner.main_td_coef == 1.0 and learner.nomask_td_auxiliary_coef == 0.0
        assert mac.args.td_lambda == 0.6 and mac.args.test_greedy
        assert mac.args.lr == 0.001 and mac.args.optimizer == "adam"
        for flag in ("mask_parameter_relation_active", "random_drop_auxiliary_active",
                     "temporal_param_auxiliary_active", "gate_regularization_active",
                     "advantage_margin_auxiliary_active", "mixer_kl80_auxiliary_active"):
            assert not getattr(learner, flag), flag
        state = mac.agent.state_dict()
        if initial_agent is None:
            initial_agent = {k: v.clone() for k, v in state.items()}
        else:
            assert state.keys() == initial_agent.keys()
            assert all(th.equal(v, initial_agent[k]) for k, v in state.items())
        for t in (10, 300000):
            learner.train(batch, t_env=t, episode_num=1)
        assert all(th.isfinite(p).all() for p in mac.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mac.agent.fixed_head.parameters())
        # The generic learner fixture omits the post-terminal state. Value
        # calibration needs that snapshot to count the terminal transition.
        batch["filled"][1, 3] = 1
        summary = ValueDiagnosticSummary()
        collect_value_diagnostics(mac, learner.mixer, batch, mac.args.gamma, summary)
        assert summary.moments["q_tot"][0] > 0 and summary.moments["q_mc_bias"][0] > 0
        if method == "vdn":
            n, total, _, _ = summary.moments["mixer_dqtot_dqi"]
            assert total / n == 1 and "mixer_w1" not in summary.moments
        else:
            assert summary.moments["mixer_w1"][0] > 0
        print("PASS: 8m_vs_9m {} production fixed GRU/head, TD-only update and value diagnostics".format(method))


if __name__ == "__main__":
    main()
