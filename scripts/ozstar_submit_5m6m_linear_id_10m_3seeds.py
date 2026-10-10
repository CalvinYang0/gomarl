#!/usr/bin/env python3
"""Submit only matched 5m6m Linear ID, three seeds, 10M, with battle videos.

Retain existing exact-name active/completed jobs. Share the older two-map
suite's lock; never submit 8m9m, cancel jobs, or reuse attention/5M ID runs.
"""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_id_5m6m_8m9m_10m_3seeds import (
    build_plans as previous_plans, GROUP as PREVIOUS_GROUP,
)
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map

GROUP = "linear_id_5m6m_value_diagnostics_10m_3seeds"
SMOKE_SCRIPTS = (
    "scripts/smoke_test_submit_5m6m_linear_id.py",
    "scripts/smoke_test_5m6m_linear_id.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    plans = [p for p in previous_plans(repo) if p["scene"] == "smac_5m6m"]
    for plan in plans:
        plan["exports"]["GROUP_NAME"] = GROUP
        plan["exports"]["EXTRA_ARGS"] += (
            " test_battle_videos=True test_battle_video_interval=1000000"
            " test_battle_video_episodes=10"
        )
    if len(plans) != 3 or {p["seed"] for p in plans} != {1, 2, 3}:
        raise RuntimeError("Expected exactly three matched 5m6m Linear-ID jobs")
    return plans


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    runtime = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    with ExitStack() as stack:
        if os.environ.get("SUBMIT") == "YES":
            if submitter.home_quota_free_gib() < 2.0:
                raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
            validate_installed_map()
            paths = submitter.guard_and_route(build_plans(repo), runtime)
            paths["logs"].mkdir(parents=True, exist_ok=True)
            # The old two-map command can submit these same names.
            lock = stack.enter_context((paths["logs"] / ("." + PREVIOUS_GROUP + ".lock")).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__),
                           SMOKE_SCRIPTS=SMOKE_SCRIPTS, build_plans=build_plans)
        original = {key: getattr(submitter, key) for key in replacement}
        try:
            for key, value in replacement.items():
                setattr(submitter, key, value)
            submitter.main()
        finally:
            for key, value in original.items():
                setattr(submitter, key, value)


if __name__ == "__main__":
    main()
