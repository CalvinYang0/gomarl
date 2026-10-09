#!/usr/bin/env python3
"""Read-only plan checks: exact maps/names and settings matched to Obs."""
from pathlib import Path

from ozstar_submit_linear_id_8m9m_6h8z_10m_3seeds import build_plans, SCENES
from ozstar_submit_linear_id_5m6m_8m9m_10m_3seeds import build_plans as previous_plans
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as obs_plans


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = build_plans(repo)
    assert len(plans) == 6
    previous = {p["job_name"]: p for p in previous_plans(repo) if p["scene"] == "smac_8m9m"}
    for scene, map_name in SCENES:
        selected = [p for p in plans if p["scene"] == scene]
        assert {p["seed"] for p in selected} == {1, 2, 3}
        references = {p["seed"]: p for p in obs_plans(repo, scene, map_name)
                      if p["label"] == "linear_baseline"}
        for plan in selected:
            exports = plan["exports"]
            assert exports["MODEL_TYPE"] == "smac_single_linear_suite_id_baseline_hypercond"
            assert exports["MAP_NAME"] == map_name
            assert plan["job_name"] == f"{scene}_linear_id_baseline_10m_s{plan['seed']}_valuediag"
            reference = references[plan["seed"]]["exports"]
            for key in ("CONFIG", "ENV_CONFIG", "SEED", "T_MAX", "TEST_INTERVAL",
                        "BATCH_SIZE_RUN", "BATCH_SIZE", "BUFFER_SIZE", "EXTRA_ARGS"):
                assert exports[key] == reference[key], (key, exports[key], reference[key])
            assert exports["T_MAX"] == "10050000"
            assert "--time=2-00:00:00" in plan["sbatch_args"]
            if scene == "smac_8m9m":
                assert plan["job_name"] in previous
                for key, value in previous[plan["job_name"]]["exports"].items():
                    if key != "GROUP_NAME":
                        assert exports[key] == value, key
    assert all(p["scene"] != "smac_5m6m" for p in plans)
    print("PASS: six exact 8m9m/6h8z jobs, Obs-matched settings, existing corrected 8m9m names retained")


if __name__ == "__main__":
    main()
