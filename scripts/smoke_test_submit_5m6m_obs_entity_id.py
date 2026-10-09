#!/usr/bin/env python3
"""Validate exact three seeds, matched settings and diagnostics; no scheduler writes."""
from pathlib import Path
import shlex

from ozstar_submit_5m6m_obs_entity_id_10m_3seeds import build_plans, LABEL
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as obs_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = build_plans(repo)
    reference = {p["seed"]: p for p in obs_plans(repo) if p["label"] == "linear_baseline"}
    assert len(plans) == 3 and {p["seed"] for p in plans} == {1, 2, 3}
    assert len({p["job_name"] for p in plans}) == 3
    for plan in plans:
        exports = plan["exports"]
        assert plan["label"] == LABEL and plan["map_name"] == "5m_vs_6m"
        assert plan["memory"] == "24G" and "--time=2-00:00:00" in plan["sbatch_args"]
        assert exports["T_MAX"] == "10050000"
        assert plan["job_name"] == "smac_5m6m_linear_obs_entity_id_10m_s{}_entityidcond".format(plan["seed"])
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "BATCH_SIZE_RUN", "BATCH_SIZE",
                    "BUFFER_SIZE", "TEST_INTERVAL", "SEED", "USE_CUDA", "WANDB_MODE"):
            assert exports[key] == reference[plan["seed"]]["exports"][key], key
        overrides = dict(arg.split("=", 1) for arg in shlex.split(exports["EXTRA_ARGS"]))
        expected = dict(arg.split("=", 1) for arg in shlex.split(reference[plan["seed"]]["exports"]["EXTRA_ARGS"]))
        for key, value in expected.items():
            if key != "clean_model_type":
                assert overrides[key] == value, key
        assert overrides["test_battle_videos"] == "True"
        assert overrides["test_battle_video_episodes"] == "10"
        assert overrides["test_battle_video_interval"] == "1000000"
        assert exports["MODEL_TYPE"] == "smac_single_linear_suite_obs_entity_id_baseline_hypercond"
    validate_config_keys(repo, plans)
    print("PASS: 5m6m Obs + entity ID x seeds 1/2/3, Obs-matched TD/Double-Q/optimizer, 10M/24G/48h, value diagnostics and videos")


if __name__ == "__main__":
    main()
