# November 4 combined-model job audit and W&B backfill

This entry handles ONLY `projected_init1_fastrefresh8m_four_maps_10m_12jobs_20261104`:
`qmix_projected_init1_faststages8m`, seeds 1/2/3 on 3s5z_vs_3s6z, MMM2,
corridor and 5m_vs_6m. It does not upload the Nov1 parent-model batches.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_projected_fastrefresh_results.py
```

Append `--status-only` for receipt/accounting checks without W&B access, local
output creation, uploading or cleanup. An empty squeue alone is not completion.

The entry reads cohort receipts and verifies exact job name, WorkDir, state,
exit code, elapsed time, Submit/Start/End via sacct. For repeated submissions it
selects the latest Submit time, never a previous successful or better attempt.
Missing or unverifiable accounting is reported rather than guessed.

Confirmed terminal jobs can upload available original records, even when failed
or interrupted. Running/pending jobs are not uploaded by this one-shot entry.
Original IDs are preserved. Already-current originals can skip raw sync based
on cloud scalar progress; this is NOT a complete per-file/media backup audit.

Per-map full mean3 requires all three latest jobs COMPLETED / 0:0 AND readable
test histories through 10M. History selection is tied to those exact job logs'
W&B IDs or Sacred paths. No parent curves, best seeds, old attempts or extrapolated
tails. Cleaned local histories may be recovered only by proven original cloud
IDs, not a project-wide search.

Local output per map includes PNG/PDF, individual seeds, inventory and raw CSV.
Aggregate figures use mean +/- sample std and centered window=10 test points;
selectable mean3 scalar runs are unsmoothed for W&B postprocessing. Identity/digest
and cloud scalar checks make repeated mean3 uploads idempotent. This entry uploads
original training runs and scalar mean3; it does not publish a new image workspace.

Report:
`/home/kyang/gomarl-runtime/gomarl-dual-branch/figures/projected_fastrefresh_four_maps_backfill_20261105/upload_report.json`.
It records job statuses, original-sync results, full mean3 results and errors.
Partial data is not reported as a full result. No videos, training, deletion,
automatic retry loop, environment upgrades or cloud-run deletion occur.

Synthetic verification:
`scripts/smoke_test_upload_projected_fastrefresh.py` exercises exact12 scoping,
mixed job states, latest attempts, four-map rendering/CSV, short-seed refusal and
idempotent mean3, with scheduler/cloud calls mocked. It does not verify real jobs
or upload synthetic fixtures to W&B.
