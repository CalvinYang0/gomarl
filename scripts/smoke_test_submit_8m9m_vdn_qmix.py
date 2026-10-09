#!/usr/bin/env python3
"""Read-only submission-plan checks; no scheduler, simulator or W&B writes."""
from pathlib import Path
import shlex

from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_plans
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as obs_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(arg.split("=", 1) for arg in shlex.split(plan["exports"]["EXTRA_ARGS"]))


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = build_plans(repo)
    reference = {p["seed"]: p for p in obs_plans(repo, "smac_8m9m", "8m_vs_9m")
                 if p["label"] == "linear_baseline"}
    assert len(plans) == len({p["job_name"] for p in plans}) == 6
    for method in ("vdn", "qmix"):
        assert {p["seed"] for p in plans if p["label"] == method} == {1, 2, 3}
    for plan in plans:
        exports = plan["exports"]
        expected = reference[plan["seed"]]["exports"]
        assert exports["MODEL_TYPE"] == "qmix_minimal"
        assert plan["map_name"] == "8m_vs_9m" and plan["memory"] == "24G"
        assert "--time=2-00:00:00" in plan["sbatch_args"]
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "SEED", "T_MAX",
                    "TEST_INTERVAL", "BATCH_SIZE_RUN", "BATCH_SIZE", "BUFFER_SIZE",
                    "USE_CUDA", "WANDB_MODE"):
            assert exports[key] == expected[key], key
        actual = extras(plan)
        assert actual["mixer"] == plan["label"] and actual["test_greedy"] == "True"
        assert actual["test_battle_videos"] == "True"
        assert actual["test_battle_video_interval"] == "1000000"
        assert actual["test_battle_video_episodes"] == "10"
        assert actual["test_nepisode"] == "32"
        for key, value in extras(reference[plan["seed"]]).items():
            assert actual[key] == value, key
    for seed in (1, 2, 3):
        vdn, qmix = [p for p in plans if p["seed"] == seed]
        assert {k: v for k, v in extras(vdn).items() if k != "mixer"} == {
            k: v for k, v in extras(qmix).items() if k != "mixer"
        }
    validate_config_keys(repo, plans)
    print("PASS: 8m_vs_9m, VDN/QMIX x seeds 1/2/3, only mixer differs; Obs-matched settings and video flags")


if __name__ == "__main__":
    main()
