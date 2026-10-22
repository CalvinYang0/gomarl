# October 22 fresh rerun after the quota failure

This is a new 18-run cohort, not continuation of the October 21 partial histories.
It never cancels jobs or deletes data. Repeating the command retains active or
completed runs with this new cohort's exact names. Failed rerun attempts can be
submitted again, but do not stitch their histories together.

| Map | Models | Seeds |
| --- | --- | --- |
| 5m6m | qmix_qtot_sum_bothones, qmix_qtot_sum_p10, qmix_qtot_alpha10_p10, qmix_soft_onefill_p10 | 1, 2, 3 each |
| MMM2 | qmix_onefill_p10 | 1, 2, 3 |
| corridor | qmix_onefill_p10 | 1, 2, 3 |

Source job IDs: 18944777–18944791 and 18944951–18944953. Source submission receipts
and unsuccessful terminal accounting must still agree before submitting anything.

Every plan copies the source training overrides, media settings, seed, map,
10.05M stopping budget and 24h resource request. In particular the historical
`qmix_minimal` GRU + two-layer ELU action head is NOT replaced by the corrected
official QMIX head. No checkpoint resume, gate/objective change or silent media
disablement is introduced. Run names end in `_rerun20261022`; the new group is
`qtot_soft_fresh_rerun_10m_18jobs_20261022`. Each plan records `rerun_of` provenance.

## Capacity checks

The dedicated launcher requires at least **5 GiB AND 50,000 free inodes** from
the user's raw `/home` quota (`quota -v`), rejecting missing/ambiguous quota data.
These conservative startup floors are NOT a measured full-batch budget or a
guarantee against subsequent exhaustion. All 18 full-length runs retain videos
and diagnostics; other jobs and uploads share the same quota.

The supplied October 22 server output reports 15,845,771 / 20,969,520 KiB and
169,467 / 200,000 inodes: about 4.886 GiB and 30,533 free inodes. This is below
BOTH startup floors, so fixing the parser does not authorize submission. The
parser now accepts OzSTAR's appended `(Disk usage is at ... Inode usage is at
...)` explanation, but still derives all quotas from the raw integer columns.
This exact server output is a regression fixture; unrelated trailing columns
and ambiguous quota rows remain rejected.
Do not bypass the guard if blocked. The launcher does not clean or relocate data.

## Server command

After deploying this commit to the existing server checkout:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_resubmit_qtot_soft_18jobs_20261022.py
```

Without `SUBMIT=YES`, only print the 18 planned jobs. Submission still uses the
existing map/config/smoke/Slurm test-only validation and exact-name duplicate
protection. The manifest is saved under the new group in runtime `ozstar_logs`.
The local agent preparing this launcher has no authenticated remote execution
channel and cannot claim these jobs were submitted without actual Slurm receipts.

The earlier ad-hoc cleanup command protected ONLY the old 18-job cohort. Do NOT
reuse it after starting this new cohort: it does not know the new retention scope.
The earlier original-log/mean3 upload commands also select only the old cohort;
the new cohort needs its own exact run-name selection when uploading results.
