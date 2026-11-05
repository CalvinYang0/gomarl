# Transfer the two original Nov1 models to three maps

Requested models are the two separate parents shown in the screenshots:

| Label | Exact original behavior |
| --- | --- |
| `qmix_soft_onefill_p10_refresh_faststages8m_evallive` | Sigmoid per-field gate initialized at .1; train mixer-state refresh `K=1+floor((L-1)*(min(t,8M)/8M)^2)`; test always live |
| `qmix_onefill_projected_init1` | Direct per-field coefficient initialized at exactly1; project into [0,1] after each optimizer step; NO refresh, mixer state always live |

These are NOT the Nov4 combined `qmix_projected_init1_faststages8m` model.
Neither freezes/binarizes the gate at5M. Closed fields fill1, and all four
QMIX state paths use the same controlled state. Actor observations remain live.

Maps: corridor, MMM2, 5m_vs_6m; each model/map has seeds1/2/3, for18 fresh jobs.
Preserve the entire parent EXTRA_ARGS byte-for-byte, including the historical
GRU + two-layer ELU action head, joint TD, optimizer/learning settings and
single complete QMIX. Only map, cohort/run identity and walltime change.

Each job uses28 CPU,32G memory, CPU-only, nominal10M environment steps
(`T_MAX=10050000`, the same historical completion margin). No GPU migration is
implied by the preceding GPU discussion. Videos disabled, static diagnostics kept.
Corridor requests48h after Nov4 seeds1/3 hit24h; MMM2 and5m6m request24h.
48h is a requested limit, not a promise of runtime; Slurm test-only validation
must accept all missing jobs before any real submission.

## Submit

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_fastrefresh_init1_three_maps_18jobs.py
```

Without SUBMIT=YES, plan-only. Optional --scene smac_corridor/smac_mmm2/smac_5m6m
selects6 jobs. Optional --variant faststages/init1 selects9 jobs. Together they
select3 jobs. Exact cohort:
`fastrefresh_and_projected_init1_three_maps_10m_18jobs_20261106`.
Fresh names end in `parents20261106`; original and Nov4 names are not reused.

The entry checks home byte+inode reserves (5GiB AND50,000 free), installed
SC2/map assets and identities, config keys, synthetic learner/startup smoke tests
and Slurm --test-only. Repeated invocation retains exact active/completed names.
Submission manifests under runtime ozstar_logs preserve each actual job ID.
No cleanup, upload, environment changes, parent-model edits or old-job restarts.
Do not delete unfinished corridor recovery sources to make room automatically.

## Verification

`scripts/smoke_test_fastrefresh_init1_three_maps.py` checks exact18 coverage,
unchanged parent EXTRA_ARGS, distinct identities, per-map walltime, CPU resources,
all18 actual no-video startup preflights, quota/map scopes and config validation;
then runs both real learners on all three maps' production feature dimensions
with Adam and RMSProp using synthetic padded episodes. It checks original actor
head, live actor observations, held mixer-state metadata, unchanged parent soft
TD on identical held inputs, exact1 versus sigmoid.1 gate initialization, bounded
updates, gate trainability after5M and target gate copying. No simulator launched
and no jobs/uploads performed by these tests.
