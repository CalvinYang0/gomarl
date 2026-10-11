#!/usr/bin/env python3
"""Check only the 8m9m health cohort; mocked scheduler, no real jobs."""
from pathlib import Path

import ozstar_submit_8m9m_health_only_10m_3seeds as study
from ozstar_submit_5m6m_health_only_10m_3seeds import build_plans as old_plans
from smoke_test_submit_5m6m_health_only import check_study


def main():
    repo = Path(__file__).resolve().parents[1]
    assert {p["job_name"] for p in study.build_plans(repo)}.isdisjoint(
        {p["job_name"] for p in old_plans(repo)})
    check_study(study, "smac_8m9m", "8m_vs_9m")


if __name__ == "__main__":
    main()
