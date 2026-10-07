#!/usr/bin/env python3
"""Update only RUNNING Slurm jobs from the fresh 5m6m ID baseline suite.

Read their local records with --local-only (recommended for periodic updates).
No historical controls, completed replicas, pending jobs or other maps are read.
Available seed count is labelled honestly if fewer than three remain running.
"""
import subprocess

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as suite_plans


def select_running(plans, running_names):
    return [p for p in plans
            if p["scene"] == "smac_5m6m" and p["label"] == "hyper_hypermarl_id"
            and p["job_name"] in running_names]


def build_plans(repo):
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    output = subprocess.check_output(
        ["squeue", "-u", user, "-t", "RUNNING", "-h", "-o", "%j"], text=True,
    )
    plans = select_running(suite_plans(repo), set(output.splitlines()))
    for plan in plans:
        print("Updating running job: " + plan["job_name"], flush=True)
    return plans


if __name__ == "__main__":
    charts.GROUP = "running_5m6m_id_10m_valuediag"
    charts.DEFAULT_OUTPUT_SUBDIR = "running_5m6m_id_10m"
    charts.RESULTS_TITLE = "currently running ID baseline seeds"
    charts.SCENES = {"smac_5m6m": ("5m vs. 6m", "test_battle_won_mean")}
    charts.LABELS = {"hyper_hypermarl_id": ("ID-based (running seeds only)", "#ff7f0e")}
    charts.build_plans = build_plans
    charts.main()
