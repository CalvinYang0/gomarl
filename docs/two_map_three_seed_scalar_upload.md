# Selectable model-level three-seed curves and ended-run backfill

Run once on the server, not in the periodic tmux loop:

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_update_two_map_three_seed.py
```

1. Check exact original IDs of terminal jobs from the configured suite manifests,
   restricted to the 48 planned QMIX/VDN runs on 5m6m/8m9m. Confirm Slurm terminal
   state and WorkDir, hold the existing once-sync lock, and check cloud visibility
   and scalar progress. Append local records only when cloud is missing/behind
   or cannot be verified; never create replacement training IDs. Missing local
   sources and failures are reported. This is not a full media/history audit.
2. Refresh both local-available figure sets and upload their analysis snapshots.
   Plot data reads only local logs and verified retained CSVs, not cloud history.
3. Create one **three-seed-aggregate analysis run per available model/map**,
   labelled `5m6m | <model> | mean3 | <snapshot>` or the 8m9m equivalent. No training
   runs are overwritten. Missing seed 1/2/3 means no mean3 run for that model.
   Three available partial seeds are shown only through their shared interval.

In W&B select the desired `mean3` runs, plot `test_battle_won_mean` against
`env_steps`, and apply smoothing/axis limits in the UI. Values are fractions
in [0,1], not percentages. Uploads contain **no smoothing** or downsampling:
10k-step linear alignment within the shared seed interval, mean, sample std
(`test_battle_won_mean_std`, ddof=1), clipped mean±std bounds, `seed_count=3`,
and `seed_1/test_battle_won_mean` through `seed_3/test_battle_won_mean`.
The bounds are not confidence intervals. Source names/IDs, run attempt times,
endpoints and raw aggregate CSV/provenance artifacts accompany each run.

Unchanged content retries resume the same content-addressed analysis run;
changed data creates a new snapshot rather than rewriting an old mean curve.
Filter W&B by `jobType=three-seed-aggregate` or tag `three-seed-aggregate` to
hide original training and image-only runs. Groups are
`smac_5m6m_three_seed_scalars` and `smac_8m9m_three_seed_scalars`.

Scalar upload results are saved under runtime
`figures/three_seed_scalar_runs/upload_report.json`. Failures do not count as
success; unavailable old logs cannot be reconstructed from a win-rate CSV.
`--skip-ended-sync` skips only the original-run audit; `--no-upload` skips all
cloud writes/audits but refreshes local figure/CSV previews. No training,
cleanup, tmux changes, or new recurring uploader is created.

Synthetic QA: `python scripts/smoke_test_three_seed_scalar_runs.py`.
