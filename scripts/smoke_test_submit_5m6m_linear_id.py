#!/usr/bin/env python3
"""Exact plans and idempotent missing-only 5m6m ID submission; no scheduler writes."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_5m6m_linear_id_10m_3seeds as study
from ozstar_submit_5m6m_linear_baseline_recheck_3seeds import build_plans as obs_plans
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p["job_name"] for p in plans}) == 3
    assert {p["seed"] for p in plans} == {1, 2, 3}
    references = {p["seed"]: p for p in obs_plans(repo)}
    for plan in plans:
        exports = plan["exports"]
        assert plan["scene"] == "smac_5m6m" and plan["map_name"] == "5m_vs_6m"
        assert plan["job_name"] == "smac_5m6m_linear_id_baseline_10m_s{}_valuediag".format(plan["seed"])
        assert exports["MODEL_TYPE"] == "smac_single_linear_suite_id_baseline_hypercond"
        assert exports["T_MAX"] == "10050000" and exports["GROUP_NAME"] == study.GROUP
        assert plan["memory"] == "24G" and "--time=2-00:00:00" in plan["sbatch_args"]
        reference = references[plan["seed"]]["exports"]
        for key in ("CONFIG", "ENV_CONFIG", "MAP_NAME", "SEED", "T_MAX", "TEST_INTERVAL",
                    "BATCH_SIZE_RUN", "BATCH_SIZE", "BUFFER_SIZE", "USE_CUDA"):
            assert exports[key] == reference[key], key
        actual = dict(a.split("=", 1) for a in shlex.split(exports["EXTRA_ARGS"]))
        expected = dict(a.split("=", 1) for a in shlex.split(reference["EXTRA_ARGS"]))
        assert all(actual[k] == value for k, value in expected.items())
        assert actual["test_value_diagnostics"] == "True"
        assert actual["test_battle_videos"] == "True"
        assert actual["test_battle_video_interval"] == "1000000"
        assert actual["test_battle_video_episodes"] == "10"
    validate_config_keys(repo, plans)
    original_builder = study.submitter.build_plans
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-5m6m-id-") as tmp:
        paths = {"logs": Path(tmp) / "logs"}
        calls, submitted = [], {}
        active_names = {plans[0]["job_name"]: "synthetic_running_id"}
        completed_names = {plans[1]["job_name"]: "synthetic_completed_id"}

        def fake_run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_test_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_test_commit"
            assert command[:2] == ["sbatch", "--parsable"]
            assert sum(c[0] == "test-only" for c in calls) == 1
            name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--job-name="))
            assert name == plans[2]["job_name"]
            calls.append(("submit", name))
            submitted[name] = "synthetic_new_id"
            return submitted[name]

        def fake_preflight(command, **kwargs):
            if command[0] == "sbatch":
                assert command[1] == "--test-only"
                assert sum(c[0] == "smoke" for c in calls) == len(study.SMOKE_SCRIPTS)
                calls.append(("test-only", command))
            else:
                assert command[1] in study.SMOKE_SCRIPTS
                calls.append(("smoke", command))

        initial_cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), \
                 patch.dict(os.environ, SUBMIT="YES", REPO_DIR=str(repo), RUNTIME_ROOT=tmp), \
                 patch.object(study, "validate_installed_map"), \
                 patch.object(study.submitter, "guard_and_route", return_value=paths), \
                 patch.object(study.submitter, "home_quota_free_gib", return_value=10), \
                 patch.object(study.submitter, "active_jobs", return_value=active_names), \
                 patch.object(study.submitter, "completed_jobs", return_value=completed_names) as completed, \
                 patch.object(study.submitter.subprocess, "run", side_effect=fake_preflight), \
                 patch.object(study.submitter, "run", side_effect=fake_run):
                study.main()
                assert len(submitted) == 1
                records = list(paths["logs"].glob(study.GROUP + "_*.json"))
                record = json.loads(records[0].read_text())
                assert record["submitted"] == submitted
                assert record["retained"] == dict(active_names, **completed_names)
                assert len(record["plans"]) == 3
                completed.return_value = dict(completed_names, **submitted)
                calls.clear()
                study.main()
                assert not any(c[0] in {"test-only", "submit"} for c in calls)
                completed.return_value = completed_names
                with patch.object(study.submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "synthetic preflight")):
                    try:
                        study.main()
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        raise AssertionError("Failed preflight must stop submission")
                assert len(submitted) == 1
        finally:
            os.chdir(initial_cwd)
    assert study.submitter.build_plans is original_builder
    print("PASS: only 5m6m Linear-ID x three seeds, 10M, matched settings/videos, missing-only submit, manifest and repeat dedup")


if __name__ == "__main__":
    main()
