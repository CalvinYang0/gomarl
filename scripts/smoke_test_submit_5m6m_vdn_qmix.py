#!/usr/bin/env python3
"""Read-only 5m6m VDN/QMIX plan checks; no scheduler or W&B writes."""
from pathlib import Path

from ozstar_submit_5m6m_vdn_qmix_10m_3seeds import build_plans, GROUP
from ozstar_submit_5m6m_linear_baseline_recheck_3seeds import build_plans as obs_plans
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_plans as other_map_plans
from smoke_test_submit_8m9m_vdn_qmix import extras
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = build_plans(repo)
    reference = {p["seed"]: p for p in obs_plans(repo)}
    assert len(plans) == len({p["job_name"] for p in plans}) == 6
    assert {p["job_name"] for p in plans}.isdisjoint(
        {p["job_name"] for p in other_map_plans(repo)})
    for method in ("vdn", "qmix"):
        assert {p["seed"] for p in plans if p["label"] == method} == {1, 2, 3}
    for plan in plans:
        exports = plan["exports"]
        assert plan["scene"] == "smac_5m6m" and plan["map_name"] == "5m_vs_6m"
        assert plan["job_name"] == "smac_5m6m_{}_baseline_10m_s{}_valuediag".format(
            plan["label"], plan["seed"])
        assert exports["RUN_NAME"] == plan["job_name"] and exports["GROUP_NAME"] == GROUP
        assert exports["MODEL_TYPE"] == "qmix_minimal" and plan["memory"] == "24G"
        assert "--time=2-00:00:00" in plan["sbatch_args"]
        expected = reference[plan["seed"]]["exports"]
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "SEED", "T_MAX",
                    "TEST_INTERVAL", "BATCH_SIZE_RUN", "BATCH_SIZE", "BUFFER_SIZE",
                    "USE_CUDA", "WANDB_MODE"):
            assert exports[key] == expected[key], key
        assert exports["T_MAX"] == "10050000"
        actual = extras(plan)
        assert actual["mixer"] == plan["label"] and actual["test_greedy"] == "True"
        assert actual["test_value_diagnostics"] == "True"
        assert actual["test_battle_videos"] == "True"
        assert actual["test_battle_video_interval"] == "1000000"
        assert actual["test_battle_video_episodes"] == "10"
        for key, value in extras(reference[plan["seed"]]).items():
            assert actual[key] == value, key
    for seed in (1, 2, 3):
        vdn, qmix = [p for p in plans if p["seed"] == seed]
        assert {k: v for k, v in extras(vdn).items() if k != "mixer"} == {
            k: v for k, v in extras(qmix).items() if k != "mixer"}
    validate_config_keys(repo, plans)
    print("PASS: 5m_vs_6m VDN/QMIX x three seeds, fresh 10M names, matched settings and videos")


if __name__ == "__main__":
    main()
