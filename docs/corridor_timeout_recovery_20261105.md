# Corridor timeout records, without requiring full10M

The reported Nov4 jobs are corridor seed1 `19916903` TIMEOUT, seed2 `19916904`
COMPLETED, seed3 `19916905` TIMEOUT. The previous original-run audit could not
discover local W&B files for seeds1/3. TIMEOUT itself never prohibited uploading;
source discovery was the separate failure. Missing W&B files do not establish
that Sacred/checkpoints are also absent.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_corridor_timeout_records.py
```

The entry rechecks exact latest job receipts/accounting for this combined-model
corridor cohort. It selects TIMEOUT only, not completed seed2 or other maps.
IDs above are evidence from the report, not hardcoded selection rules.
`--status-only` prints job status without uploads/output writes.

Recovery order:

1. Read exact job logs, strip ANSI colors, accept both offline-run and run W&B
   paths under the known runtime/repo W&B roots. If an original binary exists,
   sync complete readable records through the verified uploader, preserving ID.
   A complete10M history or clean shutdown marker is not required. Scalar
   progress verification is not a full media backup audit.
2. If original records are unavailable, use only the selected job's logged
   Sacred observer root + run number. Validate exact run name, map and seed.
   Recover all saved finite scalar/step pairs in info.json and metrics.json;
   conflicting overlaps fail. Invalid/nonfinite values are counted as omitted.
   Ambiguous same-name retries are not guessed from reusable numeric paths.
3. Publish Sacred recovery as a separate content-addressed run labelled
   `corridor | ... | s1/s3 | TIMEOUT partial`, job_type
   `recovered-partial-history`, with incomplete_training=True and source job,
   name/path, actual test endpoint and snapshot artifact. Repeating unchanged
   data skips it after verifying digest, counts/end and full test-win history.
   This is recovered scalar data, NOT the original media/checkpoint/run.
4. When local data is unavailable, query only immutable original IDs proven by
   job logs. If already present, report its actual available cloud test history.
   Never scan the whole project or invent data for an absent source.

Only actual recorded points are uploaded: no10M cutoff requirement, padding,
extrapolation, smoothing, best-seed substitution or fabricated complete mean3.
This small entry only restores the timeout single-seed records, not a partial
three-seed aggregate. Existing complete seed2 and the other maps stay untouched.

Output/audit lives under runtime
`figures/corridor_projected_fastrefresh_timeout_recovery_20261105/`.
It holds the shared upload/cleanup lock. Sources are read-only; no deletion,
resubmission, videos, checkpoint edits, model changes or dependency upgrades.
Unavailable sources produce PARTIAL, not a claimed successful upload.

`scripts/smoke_test_corridor_timeout_records.py` uses synthetic fixtures and fake
W&B/scheduler clients; it verifies timeout-only scope, original-ID preservation,
fallback from a broken native file, partial endpoints, exact scalar pairing,
conflict/refusal safeguards, idempotency and unchanged source files. No synthetic
fixtures are uploaded to the real project.
