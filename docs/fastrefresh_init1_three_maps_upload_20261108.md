# Audit/upload the Nov6 parent-transfer batch

Scope: `fastrefresh_and_projected_init1_three_maps_10m_18jobs_20261106`.
Only `*_parents20261106` jobs: faststages8m soft-onefill and projected init1,
on corridor, MMM2 and 5m6m, seeds 1/2/3. Exclude the Nov1 3s5z parents and the
Nov4 projected+fast combination. The initial receipt recorded 20069067–20069084;
the entry uses actual manifests and latest accounting, not hardcoded IDs.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_fastrefresh_init1_three_maps_results.py
```

Append `--status-only` for a read-only scheduler audit. Otherwise, the entry
checks exact latest attempts by sacct Submit time, job name and WorkDir; syncs
readable original records of ended jobs, including TIMEOUT; and publishes
selectable unsmoothed mean3 only when all three latest jobs are COMPLETED/0:0
and the three job-owned test histories actually reach 10M. Running jobs are
not synced. Incomplete histories never masquerade as full mean3.

Each complete group has local PNG/PDF and seed CSV/inventory. Local plots use
the shared observation interval and centered smoothing over 10 test points.
Mean3 uses the original unsmoothed histories on the common 10k-step grid.
No broad W&B scan; if local curves are unavailable, only proven original IDs
may supply cloud histories. A missing original file is reported, not proof of
missing cloud data. No new image workspace, re-run, cleanup or video upload.

Report: `runtime/figures/fastrefresh_init1_three_maps_backfill_20261108/upload_report.json`.
It separates job states, original sync, full mean3, and skipped/failed groups.
Repeated uploads reuse matching mean3 identities.

`scripts/smoke_test_upload_fastrefresh_init1_three_maps.py` checks the exact
cohort, latest attempt filtering, terminal versus active sync scope, and real
synthetic multi-group plots and mean3 preparation with mocked cloud/scheduler.
