#!/usr/bin/env python3
"""Real learner/plot checks; no game binary or scheduler is required."""
import os
import torch as th

from smoke_test_counter_transformer_nine import check, make_case
from ozstar_submit_trans9_multiscene import LABELS, SCENES


def check_smac_semantics(scene):
    mac, _, batch, _ = make_case("relation_kl80aux", scene)
    capturer = mac.agent.rpg_relation_capturer
    layout = capturer.observation_layout
    obs = batch["obs"][:, 0].clone()
    # Raw slicing and slot labels must use the simulator's exact flat order.
    ramp = th.arange(capturer.expected_obs_dim).float().expand_as(obs)
    own, allies, enemies = capturer._split_smac_obs(ramp)
    start = layout["move_dim"]
    assert enemies[0, 0, 0, 0] == start
    assert capturer.semantic_names[start].startswith("enemy_0_")
    start += layout["n_enemies"] * layout["enemy_feat_dim"]
    assert allies[0, 0, 0, 0] == start
    assert capturer.semantic_names[start].startswith("ally_0_")
    assert th.equal(own[..., :layout["move_dim"]], ramp[..., :layout["move_dim"]])
    assert th.equal(own[..., layout["move_dim"]:], ramp[..., -layout["own_dim"]:])

    # Invisible enemy, then an enemy visible but outside attack range.
    start = layout["move_dim"]
    obs[..., start:start + layout["enemy_feat_dim"]] = 0
    obs[..., start + layout["enemy_feat_dim"]] = 0
    capturer(obs, None)
    assert not capturer._smac_entity_mask[..., mac.args.n_agents].any()
    assert capturer._smac_entity_mask[..., mac.args.n_agents + 1].all()
    expected_visibility = capturer._smac_entity_mask.clone()
    # Gates cannot reinterpret a masked-out feature as an invisible unit.
    capturer._dynamic_branch_gate_t_env = 300000
    capturer._semantic_test_mode = True
    with th.no_grad():
        last = capturer.dynamic_branch_gate.gate_network[-1]
        last.weight.zero_()
        last.bias.fill_(-10)
        assert th.isfinite(capturer(obs, None)[0]).all()
        assert th.equal(capturer._smac_entity_mask, expected_visibility)
        assert (capturer.latest_dynamic_branch_gates_graph[1] == 0).all()
        assert th.isfinite(capturer(th.zeros_like(obs), None)[0]).all()
        # Reopen the main gate to test whether auxiliary corruption leaks.
        last.bias.fill_(3)
    capturer._semantic_test_mode = False
    def forward(test=False):
        capturer._semantic_test_mode = test
        th.manual_seed(123)
        return capturer(obs, None)[0].detach()
    before = forward()
    with th.no_grad():
        capturer.kl80_auxiliary_gate.gate_network[-1].bias.fill_(-10)
    assert th.equal(before, forward())
    capturer.kl80_auxiliary_enabled = True
    assert not th.allclose(before, forward())
    test_aux_on = forward(True)
    assert capturer.latest_kl80_auxiliary_mask is None
    capturer.kl80_auxiliary_enabled = False
    assert th.equal(test_aux_on, forward(True))
    # Available-action filtering must remain in force for the larger head.
    batch["avail_actions"][:] = 0
    batch["avail_actions"][..., 0] = 1
    mac.init_hidden(batch.batch_size)
    assert (mac.select_actions(batch, 0, 300000, test_mode=True) == 0).all()
    print(scene + ": raw layout, visibility, zero/dead obs, aux isolation, action mask OK", flush=True)


def main():
    th.set_num_threads(1)
    selected = os.environ.get("SCENES", " ".join(SCENES)).split()
    if not selected or set(selected) - set(SCENES):
        raise ValueError("SCENES must select from " + " ".join(SCENES))
    for key in selected:
        scene, domain, _ = SCENES[key]
        for label in LABELS:
            check(label, scene)
        if domain == "smac":
            check_smac_semantics(scene)
    # Optional observation features must also preserve runtime dimensions.
    if "3s5z" in selected:
        mac, _, batch, _ = make_case("relation_kl80aux", "3s5z_vs_3s6z", {
            "obs_last_action": True, "obs_timestep_number": True,
            "obs_pathing_grid": True, "obs_terrain_height": True})
        mac.init_hidden(batch.batch_size)
        assert th.isfinite(mac.forward(batch, 0)).all()
    print("Selected paired scene tests passed; simulators/Slurm NOT exercised", flush=True)


if __name__ == "__main__":
    main()
