#!/usr/bin/env python3
"""Safety tests: mocked Slurm only; never contacts or cancels real jobs."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submit_nine", ROOT / "scripts/ozstar_submit_counter_transformer_nine.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name).resolve()
        target = self.repo / "src/modules/agents"
        target.mkdir(parents=True)
        shutil.copyfile(ROOT / "src/modules/agents/counter_transformer_suite.py", target / "counter_transformer_suite.py")
        self.calls = []
        self.env = patch.dict(os.environ, {"REPO_DIR": str(self.repo), "CANCEL_OLD": "YES",
                                         "DRY_RUN": "NO", "SEED": "1", "RUN_SUFFIX": ""})
        self.env.start()

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.env.stop()
        self.temporary.cleanup()

    def query(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "id":
            return "kyang"
        if argv[0] == "squeue":
            return "11|old_counter|RUNNING\n12|unrelated|RUNNING\n13|grf_counter_trans9_baseline_s1|PENDING"
        if argv[0] == "scontrol":
            directory = "/other/project" if argv[-1] == "12" else str(self.repo)
            return "JobId=" + argv[-1] + " WorkDir=" + directory
        if argv[0] == "git":
            return "test-commit"
        if argv[0] == "sbatch":
            return str(100 + len(self.calls))
        raise AssertionError(argv)

    def action(self, argv, **kwargs):
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    def test_scope_preflight_and_reuse(self):
        with patch.object(module.subprocess, "check_output", side_effect=self.query), \
             patch.object(module.subprocess, "run", side_effect=self.action):
            module.main()
        cancellations = [c for c in self.calls if c[0] == "scancel"]
        self.assertEqual(cancellations, [["scancel", "11"]])
        self.assertEqual(sum(c[:2] == ["sbatch", "--test-only"] for c in self.calls), 8)
        self.assertEqual(sum(c[:2] == ["sbatch", "--parsable"] for c in self.calls), 8)
        cancel_index = self.calls.index(cancellations[0])
        self.assertTrue(all(self.calls.index(c) < cancel_index for c in self.calls if "--test-only" in c))
        manifest = json.loads(next((self.repo / "ozstar_logs").glob("*.json")).read_text())
        self.assertEqual(len(manifest["submitted"]), 8)
        self.assertEqual(manifest["retained"], {"grf_counter_trans9_baseline_s1": "13"})

    def test_smoke_failure_never_cancels(self):
        with patch.object(module.subprocess, "check_output") as query, \
             patch.object(module.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "smoke")):
            with self.assertRaises(subprocess.CalledProcessError):
                module.main()
        query.assert_not_called()

    def test_sbatch_preflight_failure_never_cancels(self):
        def action(argv, **kwargs):
            self.calls.append(argv)
            if "--test-only" in argv:
                raise subprocess.CalledProcessError(1, argv)
        with patch.object(module.subprocess, "check_output", side_effect=self.query), \
             patch.object(module.subprocess, "run", side_effect=action):
            with self.assertRaises(subprocess.CalledProcessError):
                module.main()
        self.assertFalse(any(c[0] == "scancel" for c in self.calls))
        self.assertFalse(any("--parsable" in c for c in self.calls))


if __name__ == "__main__":
    unittest.main()
