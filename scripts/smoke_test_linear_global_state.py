#!/usr/bin/env python3
"""Check shared global-state hyperconditioning on 5m_vs_6m, no SC2 launch."""
import torch as th

from smoke_test_counter_transformer_nine import make_case


def check():
    th.manual_seed(29)
    mac, learner, batch, logger = make_case(
        "linear_global_state_baseline", "5m_vs_6m"
    )
    agent = mac.agent
    assert agent.rpg_relation_capturer.relation_encoder_style == "linear_only"
    assert agent.counter_hyper_condition_source == "global_state_linear"
    assert agent.counter_state_condition_encoder is not None
    assert agent.counter_transformer_policy_projection is None
    assert agent.rpg_relation_capturer.dynamic_branch_gate is None

    context = mac._build_model_context(batch, 0)
    hidden = th.zeros(
        batch.batch_size, mac.args.n_agents, agent.hidden_dim
    )
    condition = agent._counter_hyper_condition(hidden, context)
    assert condition.shape == (
        batch.batch_size, mac.args.n_agents, agent.cond_dim
    )
    assert th.equal(condition, condition[:, :1].expand_as(condition))

    with th.no_grad():
        mac.init_hidden(batch.batch_size)
        mac.forward(batch, t=0, test_mode=True)
        test_condition = agent.latest_condition.clone()
        assert th.equal(
            test_condition,
            test_condition[:, :1].expand_as(test_condition),
        )
        generated = agent.latest_generated_parameter_graph
        assert generated is not None
        for parameter in generated:
            per_agent = parameter.reshape(
                batch.batch_size, mac.args.n_agents, -1
            )
            assert th.allclose(
                per_agent,
                per_agent[:, :1].expand_as(per_agent),
                atol=1e-6,
                rtol=1e-6,
            )

        # The evaluation hypercondition must respond to the global state.
        batch["state"][:, 0] += th.randn_like(batch["state"][:, 0])
        mac.init_hidden(batch.batch_size)
        mac.forward(batch, t=0, test_mode=True)
        changed_condition = agent.latest_condition.clone()
        assert not th.allclose(test_condition, changed_condition)
        assert th.equal(
            changed_condition,
            changed_condition[:, :1].expand_as(changed_condition),
        )

    # Training gradients must reach the state-to-condition encoder; the normal
    # recurrent policy still consumes each agent's local observation.
    agent.train()
    agent.zero_grad(set_to_none=True)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0).square().mean().backward()
    assert any(
        parameter.grad is not None
        and th.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in agent.counter_state_condition_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.rnn.parameters()
    )

    learner.train(batch, t_env=10, episode_num=1)
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(
        "5m_vs_6m: global-state conditions shared across agents in eval; "
        "state encoder + local-observation GRU train OK",
        flush=True,
    )


if __name__ == "__main__":
    th.set_num_threads(1)
    check()
