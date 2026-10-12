#!/usr/bin/env python3
"""Twelve plans and mocked Slurm safety; no real submission."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_slot_input_controls_12jobs as study
import ozstar_submit_linear_directkl_four_model_3seeds as history
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p["job_name"] for p in plans}) == 12
    assert {p["label"] for p in plans} == set(study.VARIANTS)
    for label in study.VARIANTS:
        assert {p["seed"] for p in plans if p["label"] == label} == {1, 2, 3}
    for p in plans:
        e = p["exports"]
        extra = dict(v.split("=", 1) for v in shlex.split(e["EXTRA_ARGS"]))
        assert p["map_name"] == "5m_vs_6m" and p["scene"] == "smac_5m6m"
        assert e["T_MAX"] == "10050000" and p["target_steps"] == 10000000
        assert e["MODEL_TYPE"] == study.VARIANTS[p["label"]]
        assert extra["agent"] == "slot_group_hyper" and extra["mac"] == "slot_group_mac"
        assert extra["learner"] == "slot_group_learner" and extra["double_q"] == "True"
        assert extra["slot_group_interval"] == "1000" and extra["slot_importance_ema"] == "0.8"
        assert extra["slot_demote_ratio"] == "0.5"
        assert extra["test_visualizations_required"] == "True"
        assert extra["test_battle_videos"] == extra["test_policy_importance"] == "True"
        assert extra["test_value_diagnostics"] == "True" and extra["test_hyper_obs_importance"] == "False"
        assert not any("kl" in k and k.endswith("_coef") and v not in {"0", "0.0"} for k, v in extra.items())
        if "static10" in p["label"]:
            assert extra["clean_hard_gate_initial_keep_probability"] == extra["clean_hard_gate_threshold"] == "0.1"
        assert p["memory"] == ("24G" if "static10" in p["label"] else "48G")
        assert "--time=2-00:00:00" in p["sbatch_args"]
    validate_config_keys(repo, plans)
    # Failed attempts must not be treated as already-completed replicas.
    records = "\n".join("{}|{}|FAILED|1:0".format(99000 + i, p["job_name"])
                        for i, p in enumerate(plans))
    with patch.object(history, "run", return_value=records):
        assert history.completed_jobs("synthetic_user") == {}
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-slots-") as temp:
        calls = []
        submitted_names = []
        expected_missing = 12
        def run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_commit"
            assert command[:2] == ["sbatch", "--parsable"]
            assert calls.count("test-only") == expected_missing
            calls.append("submit")
            submitted_names.append(next(arg.split("=", 1)[1] for arg in command
                                        if arg.startswith("--job-name=")))
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
                 patch.object(study.submitter, "home_quota_free_gib", return_value=3.33), \
                 patch.object(study.submitter, "guard_and_route", return_value={"logs": Path(temp)}), \
                 patch.object(study.submitter, "active_jobs", return_value={}) as active, \
                 patch.object(study.submitter, "completed_jobs", return_value={}) as completed, \
                 patch.object(study.submitter.subprocess, "run", side_effect=preflight), \
                 patch.object(study.submitter, "run", side_effect=run):
                study.main()
                assert calls.count("submit") == 12
                # Match the reported cluster state: static seeds active,
                # nine grouped seeds failed. Retry exactly those nine.
                static = [p for p in plans if p["label"] == "linear_static10_nokl"]
                active.return_value = {p["job_name"]: "synthetic_running" for p in static}
                calls.clear()
                submitted_names.clear()
                expected_missing = 9
                study.main()
                assert calls.count("submit") == 9
                assert set(submitted_names) == {p["job_name"] for p in plans
                                                if p["label"] != "linear_static10_nokl"}
                active.return_value = {plans[0]["job_name"]: "synthetic_running"}
                completed.return_value = {p["job_name"]: "synthetic_complete" for p in plans[1:]}
                calls.clear()
                with patch.object(study.submitter, "home_quota_free_gib", return_value=3.0):
                    study.main()
                assert not calls
                active.return_value, completed.return_value = {}, {}
                with patch.object(study.submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "synthetic")):
                    try:
                        study.main()
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        raise AssertionError("Failed preflight must prevent all jobs")
                assert not calls
                with patch.object(study.submitter, "home_quota_free_gib", return_value=2.99):
                    try:
                        study.main()
                    except RuntimeError as exc:
                        assert "no jobs submitted" in str(exc)
                        assert "2.99 GiB free" in str(exc) and "3.0 GiB" in str(exc)
                    else:
                        raise AssertionError("Quota guard must stop submission")
        finally:
            os.chdir(cwd)
        assert study.submitter.build_plans is original
    print("PASS: twelve matched 10M plans; retain 3 active static seeds and retry exactly 9 failed grouped seeds; required media; 3.33/3.00 GiB accepted, 2.99 blocked; preflight before any submission; idempotence")


if __name__ == "__main__":
    main()
