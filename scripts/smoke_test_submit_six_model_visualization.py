#!/usr/bin/env python3
"""Exact plans and simulated missing-job submission; no real scheduler writes."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_six_model_visualization_18jobs as study
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p["job_name"] for p in plans}) == 18
    assert sum(p["map_name"] == "5m_vs_6m" for p in plans) == 12
    assert sum(p["map_name"] == "8m_vs_9m" for p in plans) == 6
    for plan in plans:
        exports = plan["exports"]
        args = dict(a.split("=", 1) for a in shlex.split(exports["EXTRA_ARGS"]))
        assert exports["T_MAX"] == "10050000" and plan["memory"] == "24G"
        assert "--time=2-00:00:00" in plan["sbatch_args"]
        assert exports["TEST_INTERVAL"] == "10000" and args["test_nepisode"] == "32"
        assert args["test_value_diagnostics"] == "True"
        assert args["test_battle_videos"] == "True"
        assert args["test_battle_video_interval"] == "1000000"
        assert args["test_battle_video_episodes"] == "10"
        if plan["label"] == "linear_baseline":
            assert "_recheck" in plan["job_name"]
            assert args["test_hyper_obs_importance"] == "True"
            assert args["test_hyper_obs_importance_interval"] == "1000000"
        elif plan["label"] in {"vdn", "qmix"}:
            assert exports["MODEL_TYPE"] == "qmix_minimal"
            assert args["mixer"] == plan["label"]
    validate_config_keys(repo, plans)
    entities = [p for p in plans if p["label"] == "linear_obs_entity_id_baseline"]
    retained = {p["job_name"]: str(18342166 + i) for i, p in enumerate(entities)}
    with tempfile.TemporaryDirectory(prefix="gomarl-submit-six-") as tmp:
        paths = {"logs": Path(tmp) / "logs"}
        calls = []
        submitted = {}

        def fake_run(command, **kwargs):
            if command == ["id", "-un"]:
                return "synthetic_test_user"
            if command == ["git", "rev-parse", "HEAD"]:
                return "synthetic_test_commit"
            assert command[:2] == ["sbatch", "--parsable"], command
            # ALL missing-job scheduler tests must pass before first submit.
            assert sum(c[0] == "test-only" for c in calls) == 15
            calls.append(("submit", command))
            name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--job-name="))
            submitted[name] = str(90000 + len(submitted))
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
                 patch.object(study.submitter, "guard_and_route", return_value=paths), \
                 patch.object(study.submitter, "home_quota_free_gib", return_value=10), \
                 patch("ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map"), \
                 patch.object(study, "validate_8m9m"), \
                 patch.object(study.submitter, "active_jobs", return_value=retained) as active, \
                 patch.object(study.submitter, "completed_jobs", return_value={}) as completed, \
                 patch.object(study.submitter.subprocess, "run", side_effect=fake_preflight), \
                 patch.object(study.submitter, "run", side_effect=fake_run):
                study.main()
                assert len(submitted) == 15 and set(submitted).isdisjoint(retained)
                record = json.loads(next(paths["logs"].glob(study.GROUP + "_*.json")).read_text())
                assert record["retained"] == retained and record["submitted"] == submitted
                assert len(record["plans"]) == 18
                # Re-run: pending/running AND completed names are retained.
                active.return_value = retained
                completed.return_value = submitted.copy()
                calls.clear()
                study.main()
                assert not any(c[0] in {"test-only", "submit"} for c in calls)
                # Failed preflight stops the entire batch before submission.
                active.return_value = {}
                completed.return_value = {}
                before = len(submitted)
                with patch.object(study.submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "synthetic preflight")):
                    try:
                        study.main()
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        raise AssertionError("Preflight failure must stop submission")
                assert len(submitted) == before
        finally:
            os.chdir(initial_cwd)
    assert study.submitter.build_plans is study.obs_plans
    print("PASS: 18 exact jobs, 12x5m6m/6x8m9m, explicit visualization flags, "
          "retain 18342166/67/68, only 15 missing submissions, repeat no duplicates, preflight failure stops")


if __name__ == "__main__":
    main()
