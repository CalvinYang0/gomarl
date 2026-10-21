#!/usr/bin/env python3
"""Mocked Slurm checks: dry run, quota, all-before-submit and idempotency."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import ozstar_submit_official_qmix_6jobs as suite


def main():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="official-qmix-submit-test-") as scratch:
        root = Path(scratch)
        paths = {k: root / k for k in ("logs", "results", "wandb")}
        events, submitted = [], {}
        names = [p["job_name"] for p in suite.build_plans(repo)]
        retained = {names[0]: "101", names[1]: "102"}
        original_cwd = Path.cwd()

        def fake_run(command, **kwargs):
            if command[:2] == ["id", "-un"]:
                return "kyang"
            if command[:2] == ["git", "rev-parse"]:
                return "synthetic-test-commit"
            assert command[:2] == ["sbatch", "--parsable"], command
            events.append("submit")
            assert events.count("preflight") == 4
            name = kwargs["env"]["RUN_NAME"]
            submitted[name] = str(1000 + len(submitted))
            return submitted[name]

        def fake_subprocess(command, **kwargs):
            if command[0] == "sbatch":
                assert command[1] == "--test-only"
                events.append("preflight")
            else:
                assert Path(command[-1]).name in {"smoke_test_official_qmix.py", "smoke_test_submit_official_qmix.py"}
            return subprocess.CompletedProcess(command, 0)

        try:
            with patch.dict(os.environ, {"REPO_DIR": str(repo), "SUBMIT": "NO"}), \
                    patch.object(suite, "run", side_effect=AssertionError("dry run called Slurm")), \
                    redirect_stdout(io.StringIO()):
                suite.main()
            assert not any(root.iterdir())
            with patch.dict(os.environ, {"REPO_DIR": str(repo), "SUBMIT": "YES"}), \
                    patch.object(suite, "route_runtime", return_value=paths), \
                    patch.object(suite, "run", side_effect=fake_run), \
                    patch.object(suite, "home_quota_free_gib", return_value=0.5), \
                    redirect_stdout(io.StringIO()):
                try:
                    suite.main()
                except RuntimeError as error:
                    assert "GiB" in str(error)
                else:
                    raise AssertionError("Low quota did not block submission")
            assert not submitted and not any(root.iterdir())
            with patch.dict(os.environ, {"REPO_DIR": str(repo), "SUBMIT": "YES"}), \
                    patch.object(suite, "route_runtime", return_value=paths), \
                    patch.object(suite, "run", side_effect=fake_run), \
                    patch.object(suite, "home_quota_free_gib", return_value=10), \
                    patch.object(suite, "validate_maps"), \
                    patch.object(suite, "active_jobs", return_value={names[0]: "101"}), \
                    patch.object(suite, "completed_jobs", return_value={names[1]: "102"}), \
                    patch.object(suite.subprocess, "run", side_effect=fake_subprocess), \
                    redirect_stdout(io.StringIO()):
                suite.main()
            assert events == ["preflight"] * 4 + ["submit"] * 4
            receipts = list(paths["logs"].glob("*.json"))
            assert len(receipts) == 1
            record = json.loads(receipts[0].read_text())
            assert record["retained"] == retained and record["submitted"] == submitted
            # Repeat retains all six; no new sbatch calls or receipt overwrite.
            events.clear()
            with patch.dict(os.environ, {"REPO_DIR": str(repo), "SUBMIT": "YES"}), \
                    patch.object(suite, "route_runtime", return_value=paths), \
                    patch.object(suite, "run", side_effect=fake_run), \
                    patch.object(suite, "home_quota_free_gib", return_value=10), \
                    patch.object(suite, "validate_maps"), \
                    patch.object(suite, "active_jobs", return_value={}), \
                    patch.object(suite, "completed_jobs", return_value=dict(retained, **submitted)), \
                    patch.object(suite.subprocess, "run", side_effect=fake_subprocess), \
                    redirect_stdout(io.StringIO()):
                suite.main()
            assert not events and len(list(paths["logs"].glob("*.json"))) == 2
        finally:
            os.chdir(original_cwd)
    print("PASS: dry-run read-only, quota block, all preflights first, exact-name retention and receipts")


if __name__ == "__main__":
    main()
