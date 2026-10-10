#!/usr/bin/env python3
"""Three exact health jobs and scheduler safety; no actual submission."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_health_only_10m_3seeds as study
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as obs_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    reference = {p["seed"]: p for p in obs_plans(repo) if p["label"] == "linear_baseline"}
    assert len(plans) == len({p["job_name"] for p in plans}) == 3
    assert {p["seed"] for p in plans} == {1, 2, 3}
    for plan in plans:
        exports = plan["exports"]
        args = dict(a.split("=", 1) for a in shlex.split(exports["EXTRA_ARGS"]))
        baseline = reference[plan["seed"]]["exports"]
        assert exports["T_MAX"] == "10050000" and plan["target_steps"] == 10000000
        assert plan["map_name"] == "5m_vs_6m" and plan["label"] == study.LABEL
        assert plan["job_name"] == "smac_5m6m_linear_health_only_10m_s{}_healthcond".format(plan["seed"])
        assert plan["memory"] == "24G" and "--time=2-00:00:00" in plan["sbatch_args"]
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "BATCH_SIZE_RUN", "BATCH_SIZE",
                    "BUFFER_SIZE", "TEST_INTERVAL", "SEED", "USE_CUDA", "WANDB_MODE"):
            assert exports[key] == baseline[key], key
        for key, value in dict(a.split("=", 1) for a in shlex.split(baseline["EXTRA_ARGS"])).items():
            if key != "clean_model_type":
                assert args[key] == value, key
        assert exports["MODEL_TYPE"] == "smac_single_linear_suite_health_baseline_hypercond"
        assert args["test_visualizations_required"] == "True"
        for key in ("test_battle_videos", "test_policy_importance", "test_hyper_obs_importance"):
            assert args[key] == "True"
        for prefix in ("test_battle_video", "test_policy_importance", "test_hyper_obs_importance"):
            assert args[prefix + "_interval"] == "1000000" and args[prefix + "_episodes"] == "10"
    validate_config_keys(repo, plans)
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-health-") as temp:
        calls = []
        def fake_run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_commit"
            assert command[:2] == ["sbatch", "--parsable"]
            assert calls.count("test-only") == 3
            calls.append("submit")
            return str(99000 + calls.count("submit"))
        def preflight(command, **kwargs):
            if command[0] == "sbatch":
                assert command[1] == "--test-only" and "submit" not in calls
                calls.append("test-only")
            else:
                assert command[1] in study.SMOKE_SCRIPTS
        cwd, original = Path.cwd(), study.submitter.build_plans
        try:
            with redirect_stdout(io.StringIO()), \
                 patch.dict(os.environ, SUBMIT="YES", REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch.object(study, "validate_installed_maps"), \
                 patch.object(study.submitter, "home_quota_free_gib", return_value=10), \
                 patch.object(study.submitter, "guard_and_route", return_value={"logs": Path(temp)}), \
                 patch.object(study.submitter, "active_jobs", return_value={}) as active, \
                 patch.object(study.submitter, "completed_jobs", return_value={}) as complete, \
                 patch.object(study.submitter.subprocess, "run", side_effect=preflight), \
                 patch.object(study.submitter, "run", side_effect=fake_run):
                study.main()
                assert calls.count("submit") == 3
                active.return_value = {plans[0]["job_name"]: "synthetic_running"}
                complete.return_value = {p["job_name"]: "synthetic_completed" for p in plans[1:]}
                calls.clear()
                study.main()
                assert not calls
                active.return_value, complete.return_value = {}, {}
                with patch.object(study.submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "synthetic")):
                    try:
                        study.main()
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        raise AssertionError("Failed preflight must stop all submission")
                assert not calls
        finally:
            os.chdir(cwd)
        assert study.submitter.build_plans is original
    print("PASS: exact three health-only 10M plans; matched settings; required videos/importance; all-before-any preflight; retain active/completed; idempotence")


if __name__ == "__main__":
    main()
