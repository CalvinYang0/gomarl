#!/usr/bin/env python3
"""Exact nine plans and fake scheduler safety; never submits real jobs."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shlex
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_mixer_input_controls_9jobs as study
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p["job_name"] for p in plans}) == 9
    for label in study.VARIANTS:
        assert {p["seed"] for p in plans if p["label"] == label} == {1, 2, 3}
    for p in plans:
        e = p["exports"]
        extras = dict(a.split("=", 1) for a in shlex.split(e["EXTRA_ARGS"]))
        assert p["map_name"] == "5m_vs_6m" and e["T_MAX"] == "10050000"
        assert e["MODEL_TYPE"] == "qmix_minimal" and extras["mixer"] == "qmix"
        assert extras["learner"] == "mixer_input_learner"
        assert extras["mixer_input_control"] == study.VARIANTS[p["label"]]
        assert extras["double_q"] == "True" and extras["slot_group_interval"] == "1000"
        assert extras["mixer_input_initial_probability"] == extras["mixer_input_gate_threshold"] == "0.1"
        assert extras["test_visualizations_required"] == extras["test_battle_videos"] == extras["test_policy_importance"] == "True"
        assert extras["test_hyper_obs_importance"] == "False"
        assert "--job-name=" + p["job_name"] in p["sbatch_args"]
        assert e["RUN_NAME"] == p["job_name"]
    validate_config_keys(repo, plans)
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-mixer-") as temp:
        calls = []
        retained = {}
        def fake_run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_commit"
            assert command[:2] == ["sbatch", "--parsable"]
            assert calls.count("preflight") == 9 - len(retained)
            calls.append("submit")
            return str(99000 + calls.count("submit"))
        def preflight(command, **kwargs):
            if command[0] == "sbatch":
                assert command[1] == "--test-only" and "submit" not in calls
                calls.append("preflight")
            else:
                assert command[1] in study.SMOKE_SCRIPTS
        cwd = Path.cwd()
        original = study.submitter.build_plans
        try:
            with redirect_stdout(io.StringIO()), \
                 patch.dict(os.environ, SUBMIT="YES", REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch("ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map"), \
                 patch.object(study.submitter, "home_quota_free_gib", return_value=2.28), \
                 patch.object(study.submitter, "guard_and_route", return_value={"logs": Path(temp)}), \
                 patch.object(study.submitter, "active_jobs", return_value=retained) as active, \
                 patch.object(study.submitter, "completed_jobs", return_value={}), \
                 patch.object(study.submitter.subprocess, "run", side_effect=preflight), \
                 patch.object(study.submitter, "run", side_effect=fake_run):
                study.main()
                assert calls.count("submit") == 9
                retained.update({p["job_name"]: "synthetic_running" for p in plans})
                active.return_value = retained
                calls.clear()
                study.main()
                assert not calls
                with patch.object(study.submitter, "home_quota_free_gib", return_value=2.0):
                    study.main()
                assert not calls
                with patch.object(study.submitter, "home_quota_free_gib", return_value=1.99):
                    try:
                        study.main()
                    except RuntimeError as exc:
                        assert "no jobs submitted" in str(exc)
                    else:
                        raise AssertionError("Quota guard must prevent submission")
                assert not calls
        finally:
            os.chdir(cwd)
        assert study.submitter.build_plans is original
    from plot_5m6m_mixer_input_controls_3seeds import build_plans
    plotting = build_plans(repo)
    assert len(plotting) == 21 and len({p["job_name"] for p in plotting}) == 21
    assert {p["label"] for p in plotting} == set(study.VARIANTS) | {"qmix", "historical_qmix", "vdn", "historical_vdn"}
    print("PASS: 9 exact ordinary-head jobs; declared config; all preflights before submit; idempotence; 2.28/2.00 GiB accepted, 1.99 blocked; isolated plotting")


if __name__ == "__main__":
    main()
