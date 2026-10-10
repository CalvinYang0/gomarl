#!/usr/bin/env python3
"""Exact 24-job plan and mocked all-before-any/idempotent Slurm submission."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_head_input_24jobs as study
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p["job_name"] for p in plans}) == 24
    expected = {(scene, label) for scene, _, labels in study.SCENES for label in labels}
    assert len(expected) == 8 and {(p["scene"], p["label"]) for p in plans} == expected
    for scene, label in expected:
        assert {p["seed"] for p in plans if (p["scene"], p["label"]) == (scene, label)} == {1, 2, 3}
    for plan in plans:
        exports = plan["exports"]
        args = dict(a.split("=", 1) for a in shlex.split(exports["EXTRA_ARGS"]))
        assert exports["T_MAX"] == "10050000" and plan["target_steps"] == 10000000
        assert plan["memory"] == "24G" and "--time=2-00:00:00" in plan["sbatch_args"]
        assert exports["TEST_INTERVAL"] == "10000" and args["test_nepisode"] == "32"
        assert exports["MODEL_TYPE"].startswith("smac_single_linear_suite_")
        assert plan["job_name"].endswith("_vizcoverage")
        assert args["test_value_diagnostics"] == args["test_visualizations_required"] == "True"
        for prefix in ("test_battle_video", "test_policy_importance", "test_hyper_obs_importance"):
            assert args[prefix + "_interval"] == "1000000" and args[prefix + "_episodes"] == "10"
        assert args["test_battle_videos"] == args["test_policy_importance"] == "True"
        assert args["test_hyper_obs_importance"] == "True"
    assert {p["map_name"] for p in plans if p["scene"] == "smac_3svs5z"} == {"3s_vs_5z"}
    assert {p["map_name"] for p in plans if p["scene"] == "smac_8m9m"} == {"8m_vs_9m"}
    validate_config_keys(repo, plans)
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-head-inputs-") as temp:
        paths = {"logs": Path(temp) / "logs"}
        active_names = {plans[0]["job_name"]: "synthetic_running"}
        completed_names = {plans[1]["job_name"]: "synthetic_completed"}
        calls, submitted = [], {}

        def fake_run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_test_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_test_commit"
            assert command[:2] == ["sbatch", "--parsable"], command
            assert sum(kind == "test-only" for kind, _ in calls) == 22
            name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--job-name="))
            calls.append(("submit", name))
            submitted[name] = str(99000 + len(submitted))
            return submitted[name]

        def fake_preflight(command, **kwargs):
            if command[0] == "sbatch":
                assert command[1] == "--test-only"
                assert sum(kind == "smoke" for kind, _ in calls) == len(study.SMOKE_SCRIPTS)
                calls.append(("test-only", command))
            else:
                assert command[1] in study.SMOKE_SCRIPTS
                calls.append(("smoke", command))

        original_cwd, original_builder = Path.cwd(), study.submitter.build_plans
        try:
            with redirect_stdout(io.StringIO()), \
                 patch.dict(os.environ, SUBMIT="YES", REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch.object(study, "validate_installed_maps"), \
                 patch.object(study.submitter, "home_quota_free_gib", return_value=10), \
                 patch.object(study.submitter, "guard_and_route", return_value=paths), \
                 patch.object(study.submitter, "active_jobs", return_value=active_names) as active, \
                 patch.object(study.submitter, "completed_jobs", return_value=completed_names) as completed, \
                 patch.object(study.submitter.subprocess, "run", side_effect=fake_preflight), \
                 patch.object(study.submitter, "run", side_effect=fake_run):
                study.main()
                assert len(submitted) == 22 and set(submitted).isdisjoint(active_names | completed_names)
                record = json.loads(next(paths["logs"].glob(study.GROUP + "_*.json")).read_text())
                assert len(record["plans"]) == 24 and record["submitted"] == submitted
                assert record["retained"] == dict(active_names, **completed_names)
                completed.return_value = dict(completed_names, **submitted)
                calls.clear()
                study.main()
                assert not any(kind in {"submit", "test-only"} for kind, _ in calls)
                active.return_value, completed.return_value = {}, {}
                with patch.object(study.submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "synthetic preflight")):
                    try:
                        study.main()
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        raise AssertionError("Failed preflight must prevent submission")
                assert len(submitted) == 22
        finally:
            os.chdir(original_cwd)
        assert study.submitter.build_plans is original_builder
    print("PASS: exactly 24 new 10M jobs/eight cohorts, explicit videos/importance, maps/seeds, main-only Linear models; retain live/completed; repeat dedup; preflight stops")


if __name__ == "__main__":
    main()
