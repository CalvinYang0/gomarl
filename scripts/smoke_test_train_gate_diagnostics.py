#!/usr/bin/env python3
"""Validate replay masking, real applied masks, W&B output and RNG neutrality."""
from types import SimpleNamespace
import torch as th
from smoke_test_counter_transformer_nine import make_case
from utils.train_gate_diagnostics import TrainGateDiagnostics


def check_collector():
    valid = th.tensor([[[1.], [1.]], [[1.], [0.]]])
    diagnostics = TrainGateDiagnostics(valid)
    diagnostics.add("constant", th.full((2, 1, 2), .8, requires_grad=True), 0)
    diagnostics.add("constant", th.tensor([[[.8, .8]], [[99., 99.]]]), 1)
    diagnostics.add("constant", th.full((2, 1, 2), 99.), 2)
    values = {}
    diagnostics.log(SimpleNamespace(log_stat=lambda key, value, t: values.update({key: value})), 1, [])
    assert values["train_gate/constant/valid_slot_count"] == 6
    assert abs(values["train_gate/constant/mean"] - .8) < 1e-6
    assert values["train_gate/constant/std"] < 1e-7
    assert all(not tensor.requires_grad for pair in diagnostics.stats.values() for tensor in pair)


def run(label, enabled):
    th.manual_seed(7)
    mac, learner, batch, logger = make_case(label)
    learner.args.clean_train_gate_diagnostics = enabled
    frames = {}
    rendered = []
    if enabled:
        logger.use_wandb = True
        logger.wandb_current_t = 300000
        logger.wandb_current_data = {}
        logger.wandb = SimpleNamespace(log=lambda *args, **kwargs: None)

        def image(fig):
            fig.canvas.draw()
            rendered.append(True)
            return "rendered"

        logger.wandb_module = SimpleNamespace(Image=image)
        plot = logger.log_train_gate_heatmaps

        def capture(trajectories, names, t, episode):
            frames.update(trajectories)
            plot(trajectories, names, t, episode)

        logger.log_train_gate_heatmaps = capture
    th.manual_seed(123)
    learner.train(batch, t_env=300000, episode_num=1)
    rng = th.get_rng_state().clone()
    state = {key: tensor.detach().clone() for key, tensor in mac.agent.state_dict().items()}
    if enabled:
        main_name = "main_" + learner.counter_branch_name + "_mask"
        assert logger.stats["train_gate/" + main_name + "/valid_slot_count"][-1][1] == 840
        assert "train_gate/" + main_name + "/mean" in logger.wandb_current_data
        assert rendered and len(rendered) == len(frames)
        for name, history in frames.items():
            assert [frame[0] for frame in history] == [0, 1, 2, 3]
            assert all(frame[1].shape == (4, 30) for frame in history)
            assert "train_gate_heatmap/" + name in logger.wandb_current_data
        if label != "kl80":
            prefix = "aux_fixed80" if label == "relation_random80" else "aux_" + learner.kl_auxiliary_tag
            for (_, main), (_, sampled), (_, combined) in zip(
                    frames[prefix + "_main_mask"], frames[prefix + "_mask"], frames[prefix + "_combined_mask"]):
                assert th.equal(main * sampled, combined)
                assert ((sampled > 0) & (sampled < 1)).any()
            assert logger.stats["train_gate/" + prefix + "_probability/valid_slot_count"][-1][1] == 840
            if label == "relation_random80":
                assert logger.stats["train_gate/aux_fixed80_probability/std"][-1][1] < 1e-7
        # Scalar diagnostics can continue without creating another image set.
        rendered.clear()
        learner.train(batch, t_env=300001, episode_num=2)
        assert not rendered
    return state, rng


if __name__ == "__main__":
    th.set_num_threads(1)
    check_collector()
    for label in ("kl80", "relation_kl80aux", "relation_random80", "relation_kl50aux", "relation_kl30aux", "linear_relation_kl80aux"):
        plain, plain_rng = run(label, False)
        logged, logged_rng = run(label, True)
        assert th.equal(plain_rng, logged_rng), label
        assert all(th.equal(plain[key], logged[key]) for key in plain), label
        print(label + ": valid-only stats, rendered heatmaps, identical update and RNG OK")
