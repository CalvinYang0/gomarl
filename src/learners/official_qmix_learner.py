"""Registry boundary for the pinned, import-adapted upstream QMIX learner.

No training method is overridden. Unsupported upstream branches (Qatten and
matrix-game printing) are deliberately excluded at this boundary.
"""
from third_party.pymarl2_qmix.learner import NQLearner


UPSTREAM_COMMIT = "8ccac7c5aa134422a2e3009be735d23cdf8ce2f8"


class OfficialQMIXLearner(NQLearner):
    def __init__(self, mac, scheme, logger, args):
        expected = {
            "env": "sc2", "agent": "official_pymarl2_rnn",
            "mac": "official_pymarl2_mac", "mixer": "qmix",
            "official_qmix_upstream_commit": UPSTREAM_COMMIT,
            "optimizer": "adam", "lr": 0.001, "td_lambda": 0.6,
            "gamma": 0.99, "rnn_hidden_dim": 64,
            "mixing_embed_dim": 32, "hypernet_embed": 64,
            "batch_size_run": 8, "batch_size": 128, "buffer_size": 5000,
            "epsilon_start": 1.0, "epsilon_finish": 0.05,
            "epsilon_anneal_time": 100000, "target_update_interval": 200,
            "grad_norm_clip": 10, "learner_updates_per_collect": 1,
            "test_noise": 0.0,
            "obs_last_action": True, "obs_agent_id": True,
            "q_lambda": False, "use_per": False,
            "use_layer_norm": False, "use_orthogonal": False,
        }
        for key, value in expected.items():
            if getattr(args, key, None) != value:
                raise ValueError("Official QMIX contract: {} must equal {!r}".format(key, value))
        if getattr(args, "qmix_pos_func", "abs") != "abs":
            raise ValueError("Official QMIX suite requires the upstream default abs mixer")
        if getattr(args, "weight_decay", 0) != 0 or getattr(args, "use_amp", False):
            raise ValueError("Official QMIX suite has no weight decay or AMP")
        super().__init__(mac, scheme, logger, args)
