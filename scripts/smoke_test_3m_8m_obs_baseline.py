#!/usr/bin/env python3
"""Real-map dimensions, default main-input semantics and learner updates.

No SC2 launch: uses synthetic padded episodes with installed SMAC dimensions.
"""
import torch as th

from smoke_test_counter_transformer_nine import make_case


def check(scene):
    th.manual_seed(31)
    fixture_mac, fixture_learner, batch, logger = make_case("linear_baseline", scene)
    args = fixture_mac.args
    # The shared fixture strips these extras. Restore production defaults
    # explicitly so this preflight also exercises agent ID and previous action.
    args.obs_agent_id = True
    args.obs_last_action = True
    mac = type(fixture_mac)(batch.scheme, {"agents": args.n_agents}, args)
    learner = type(fixture_learner)(mac, batch.scheme, logger, args)
    assert not mac._exclude_agent_id_from_trunk()
    assert mac.agent.rpg_relation_capturer.relation_encoder_style == "linear_only"
    assert mac.agent.rpg_relation_capturer.dynamic_branch_gate is None
    assert mac.agent.counter_hyper_condition_source is None
    # Identical raw obs produce identical hypernetwork conditions, while
    # one-hot IDs distinguish the recurrent main-network inputs.
    obs = batch["obs"][:, 0]
    obs[:] = obs[:, :1].expand_as(obs).clone()
    inputs = mac._build_inputs(batch, 0)
    assert not th.equal(inputs[:, 0], inputs[:, 1])
    assert th.equal(inputs[:, :, -args.n_agents:],
                    th.eye(args.n_agents).unsqueeze(0).expand(batch.batch_size, -1, -1))
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        q = mac.forward(batch, t=0, test_mode=True)
        condition = mac.agent.latest_condition
    assert th.allclose(condition[:, 0], condition[:, 1], atol=1e-6, rtol=1e-5)
    assert q.shape == (batch.batch_size, args.n_agents, args.n_actions)
    for t_env in (10, 300000):
        learner.train(batch, t_env=t_env, episode_num=1)
    assert all(th.isfinite(p).all() for p in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(scene + ": main-input IDs, obs-only hypercondition and learner updates OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    for scene in ("3m", "8m"):
        check(scene)
