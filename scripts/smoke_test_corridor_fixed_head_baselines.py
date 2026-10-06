#!/usr/bin/env python3
"""Real Corridor dimensions: fixed-head learner and scalar value diagnostics.

Needs cluster SMAC/PySC2 dependencies, but does not launch StarCraft.
"""
import torch as th

from smoke_test_counter_transformer_nine import make_case
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics


def main():
    th.set_num_threads(1)
    for method, mixer_type in (("vdn", VDNMixer), ("qmix", QMixer)):
        th.manual_seed(29)
        mac, learner, batch, logger = make_case(
            "baseline", "corridor",
            config_overrides={"clean_model_type": "qmix_minimal", "mixer": method},
        )
        assert mac.agent.model_type == "qmix_minimal"
        assert isinstance(learner.mixer, mixer_type)
        assert isinstance(learner.target_mixer, mixer_type)
        for t in (10, 300000):
            learner.train(batch, t_env=t, episode_num=1)
        assert all(th.isfinite(p).all() for p in mac.parameters())
        summary = ValueDiagnosticSummary()
        collect_value_diagnostics(mac, learner.mixer, batch, mac.args.gamma, summary)
        assert summary.moments["q_tot"][0] > 0
        assert summary.moments["q_mc_bias"][0] > 0
        if method == "vdn":
            n, total, _, _ = summary.moments["mixer_dqtot_dqi"]
            assert total / n == 1
            assert "mixer_w1" not in summary.moments
        else:
            assert summary.moments["mixer_w1"][0] > 0
        print("corridor {}: learner + value diagnostics OK".format(method))


if __name__ == "__main__":
    main()
