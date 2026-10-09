#!/usr/bin/env python3
"""Fill missing jobs in the agreed six-model, two-map visualization batch.

5m6m: fresh Obs, all-ones, timestep, Obs+entity-ID (12 jobs).
8m9m: fixed-head VDN and QMIX (6 jobs). No cancellation or old-Obs reuse.
Plan-only unless SUBMIT=YES. Exact-name active/completed jobs are retained.
"""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_ones_timestep_10m_3seeds import (
    build_plans as signal_plans, GROUP as SIGNAL_GROUP,
)
from ozstar_submit_5m6m_obs_entity_id_10m_3seeds import (
    build_plans as entity_plans, GROUP as ENTITY_GROUP,
)
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import (
    build_plans as mixer_plans, GROUP as MIXER_GROUP, validate_installed_map as validate_8m9m,
)

GROUP = "six_model_visualization_18jobs_10m"
OBS_GROUP = submitter.GROUP
obs_plans = submitter.build_plans
SMOKE_SCRIPTS = (
    "scripts/smoke_test_submit_six_model_visualization.py",
    "scripts/smoke_test_linear_single_head_suite.py",
    "scripts/smoke_test_hyper_obs_importance.py",
    "scripts/smoke_test_5m6m_ones_timestep.py",
    "scripts/smoke_test_submit_5m6m_ones_timestep.py",
    "scripts/smoke_test_5m6m_obs_entity_id.py",
    "scripts/smoke_test_submit_5m6m_obs_entity_id.py",
    "scripts/smoke_test_submit_8m9m_vdn_qmix.py",
    "scripts/smoke_test_8m9m_fixed_head_baselines.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    plans = obs_plans(repo)
    for plan in plans:
        # Explicit flags: a new run to collect visualization, not the old
        # completed scalar-only Obs run. Do not change training/loss settings.
        plan["exports"]["EXTRA_ARGS"] += (
            " test_value_diagnostics=True test_value_diagnostics_interval=100000"
            " test_battle_videos=True test_battle_video_interval=1000000"
            " test_battle_video_episodes=10 test_hyper_obs_importance=True"
            " test_hyper_obs_importance_interval=1000000"
            " test_hyper_obs_importance_episodes=10"
        )
    plans.extend(signal_plans(repo))
    plans.extend(entity_plans(repo))
    plans.extend(mixer_plans(repo))
    expected = {
        "linear_baseline": "5m_vs_6m", "linear_ones_baseline": "5m_vs_6m",
        "linear_timestep_baseline": "5m_vs_6m", "linear_obs_entity_id_baseline": "5m_vs_6m",
        "vdn": "8m_vs_9m", "qmix": "8m_vs_9m",
    }
    if len(plans) != 18 or len({p["job_name"] for p in plans}) != 18:
        raise RuntimeError("Expected eighteen distinct visualization jobs")
    if {p["label"] for p in plans} != set(expected):
        raise RuntimeError("Unexpected model in six-model batch")
    for label, map_name in expected.items():
        selected = [p for p in plans if p["label"] == label]
        if {p["seed"] for p in selected} != {1, 2, 3}:
            raise RuntimeError("Incorrect seeds: " + label)
        if any(p["map_name"] != map_name or p["exports"]["T_MAX"] != "10050000"
               for p in selected):
            raise RuntimeError("Incorrect map/budget: " + label)
    return plans


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    runtime = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    with ExitStack() as stack:
        if os.environ.get("SUBMIT") == "YES":
            from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
            if submitter.home_quota_free_gib() < 2.0:
                raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
            validate_installed_map()
            validate_8m9m()
            paths = submitter.guard_and_route(build_plans(repo), runtime)
            paths["logs"].mkdir(parents=True, exist_ok=True)
            # Share the existing individual-suite locks, so invoking an old
            # submit command concurrently cannot duplicate the same jobs.
            for group in sorted((OBS_GROUP, SIGNAL_GROUP, ENTITY_GROUP, MIXER_GROUP)):
                handle = stack.enter_context((paths["logs"] / ("." + group + ".lock")).open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX)
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
