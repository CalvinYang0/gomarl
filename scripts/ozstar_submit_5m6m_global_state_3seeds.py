#!/usr/bin/env python3
"""Submit the 10M, three-seed 5m_vs_6m global-state-conditioned Linear run.

Only the generated-head condition changes: the policy GRU still receives local
observations, while all agents receive the same global-state hypercondition.
Plan-only by default; SUBMIT=YES enables validated Slurm submission.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "smac_5m6m_linear_global_state_10m_3seeds"
SCENE = "smac_5m6m"
MAP = "5m_vs_6m"
LABEL = "linear_global_state_baseline"
SEEDS = (1, 2, 3)
MIN_HOME_FREE_GIB = 2.0


def build_plans(repo):
    profiles = _load_profiles(repo)
    flags = profiles.ALL_PROFILES[LABEL]
    expected_flags = {
        "branch": "linear",
        "hyper_condition": "global_state_linear",
    }
    if flags != expected_flags:
        raise RuntimeError("Unexpected global-state profile: " + repr(flags))

    plans = []
    for seed in SEEDS:
        name = "{}_linear_global_state_10m_s{}_statecond".format(SCENE, seed)
        plan = _plan(
            repo, profiles, SCENE, MAP, "smac", LABEL, seed,
            name, "24G", GROUP,
        )
        plan["exports"]["T_MAX"] = "10050000"
        plan["exports"]["TEST_INTERVAL"] = "10000"
        plan["exports"]["EXTRA_ARGS"] = _extra_args(
            profiles, LABEL, "smac"
        ) + " test_value_diagnostics=True test_value_diagnostics_interval=100000"
        plan["sbatch_args"] = [
            "--time=2-00:00:00" if arg.startswith("--time=") else arg
            for arg in plan["sbatch_args"]
        ]
        plan["profile_flags"] = flags
        plans.append(plan)

    expected_model = profiles.model_type_for(LABEL, "smac")
    if "single_linear" not in expected_model or "global_state_baseline" not in expected_model:
        raise RuntimeError("Unexpected resolved model type: " + expected_model)
    if any(plan["exports"]["MODEL_TYPE"] != expected_model for plan in plans):
        raise RuntimeError("Resolved model type changed between seeds")
    if len(plans) != 3 or len({plan["job_name"] for plan in plans}) != 3:
        raise RuntimeError("Expected exactly three unique state-conditioned jobs")
    return plans


def main():
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = ("scripts/smoke_test_linear_global_state.py",)
    submitter.build_plans = build_plans

    if os.environ.get("SUBMIT") != "YES":
        os.environ["DRY_RUN"] = "YES"
        submitter.main()
        return

    free_gib = submitter.home_quota_free_gib()
    print("/home quota free: {:.2f} GiB".format(free_gib), flush=True)
    if free_gib < MIN_HOME_FREE_GIB:
        raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry

    params = get_smac_map_registry().get(MAP)
    if params is None:
        raise RuntimeError("Map not registered in installed SMAC: " + MAP)
    actual = (params["n_agents"], params["n_enemies"], params["map_type"])
    if actual != (5, 6, "marines"):
        raise RuntimeError("Unexpected SMAC map definition: {}".format(actual))
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    if not any(maps_root.rglob(MAP + ".SC2Map")):
        raise RuntimeError("Missing map file under {}: {}".format(maps_root, MAP))
    print("Map preflight passed: {} {}".format(MAP, actual), flush=True)
    os.environ.pop("DRY_RUN", None)
    submitter.main()


if __name__ == "__main__":
    main()
