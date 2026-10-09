#!/usr/bin/env python3
"""Validate six exact matched plans and video flags; no scheduler writes."""
from pathlib import Path
import shlex

from ozstar_submit_5m6m_ones_timestep_10m_3seeds import build_plans, CONTROLS
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as obs_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = build_plans(repo)
    reference = {p["seed"]: p for p in obs_plans(repo) if p["label"] == "linear_baseline"}
    assert len(plans) == 6
    for label, suffix, _ in CONTROLS:
        assert {p["seed"] for p in plans if p["label"] == label} == {1, 2, 3}
    for plan in plans:
        exports = plan["exports"]
        assert plan["map_name"] == "5m_vs_6m" and plan["memory"] == "24G"
        assert "--time=2-00:00:00" in plan["sbatch_args"]
        assert exports["T_MAX"] == "10050000"
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "BATCH_SIZE_RUN", "BATCH_SIZE",
                    "BUFFER_SIZE", "TEST_INTERVAL", "SEED", "USE_CUDA", "WANDB_MODE"):
            assert exports[key] == reference[plan["seed"]]["exports"][key], key
        overrides = dict(arg.split("=", 1) for arg in shlex.split(exports["EXTRA_ARGS"]))
        expected = dict(arg.split("=", 1) for arg in shlex.split(reference[plan["seed"]]["exports"]["EXTRA_ARGS"]))
        for key, value in expected.items():
            assert overrides[key] == value, key
        assert overrides["test_battle_videos"] == "True"
        assert overrides["test_battle_video_episodes"] == "10"
        assert overrides["test_battle_video_interval"] == "1000000"
        assert "single_linear_suite" in exports["MODEL_TYPE"]
    validate_config_keys(repo, plans)
    print("PASS: two profiles x three seeds, Obs-matched optimizer/rollout/Double-Q flags, 10M/48h/24G, declared video flags")


if __name__ == "__main__":
    main()
