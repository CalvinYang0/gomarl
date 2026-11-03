# Exact-one projected gate + faststages8M

## Scope

Combine the purple `qmix_onefill_projected_init1` with the orange-red
`qmix_soft_onefill_p10_refresh_faststages8m_evallive`. This is a new combined
variant, not a rerun of either parent and not the 5M binary-freeze variant.

| Scene | Installed map | Seeds |
| --- | --- | --- |
| 3s5z | `3s5z_vs_3s6z` | 1, 2, 3 |
| mmm2 | `MMM2` | 1, 2, 3 |
| corridor | `corridor` | 1, 2, 3 |
| 5m6m | `5m_vs_6m` | 1, 2, 3 |

12 fresh jobs; each uses 28 CPU, 32G memory, 24h, nominal 10M environment
steps (`T_MAX=10050000`, same historical completion margin). Model label:
`qmix_projected_init1_faststages8m`. Group:
`projected_init1_fastrefresh8m_four_maps_10m_12jobs_20261104`.

## Forward and training

1. Actor observations always update every timestep. Preserve the historical
   `qmix_minimal` GRU + two-layer ELU action head; do not switch to the corrected
   official-baseline actor.
2. At episode/reset rollout collection progress `t`, maximum episode length `L`,
   select `K = 1 + floor((L-1)*(min(t,8000000)/8000000)^2)` segments. Segment
   starts are `floor(j*L/K)`. Hold the raw mixer state from each segment start.
   `K` stays fixed for that rollout; early episode termination does not rescale
   the partition. This is the existing quadratic faststages rule, not successive
   fixed 2M stages. The inherited `stage_steps=2000000` is not used by this rule.
3. Gate the held state: `x = p * held_state + (1-p)`. One direct parameter per
   state scalar, shared across episodes/timesteps; every parameter starts at
   **exactly 1**. No sigmoid, sampling, straight-through estimator or threshold.
   `p=1` passes actual/held state; `p=0` fills 1, not 0.
4. A single complete QMIX uses `x` for all four state-conditioned parameter
   paths: `W1`, `W2`, `b1`, `b2`. Agent Q inputs are unchanged. No dual branch.
5. The gate and existing networks train jointly with the unchanged parent
   projected-onefill TD loss/optimizer. After each optimizer step clamp the
   direct gate parameter into `[0,1]`, before any scheduled target copy.
   No extra KL/MC/importance loss, no 5M freeze or binarization.
6. At and after 8M, new training rollouts are live each timestep, including the
   terminal state. Earlier replay episodes keep their original held states and
   collection metadata; do not reinterpret them using the latest schedule.
7. Test rollouts are always live each timestep, from the first test onward.
   Evaluation only changes the mixer-state collection; the decentralized actor
   does not consume QMIX output to choose actions.

For the installed 3s5z maximum length 170, example counts at 0/2/4/6/8M are
1/11/43/96/170. Other maps use their own installed maximum episode lengths.

Gate heatmaps, scalar refresh diagnostics and required static policy diagnostics
remain enabled. Videos are disabled. Existing upload/cleanup policy is unchanged;
this submission command does not upload or delete any data.

## Run on OzSTAR

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_projected_fastrefresh_four_maps_12jobs.py
```

Optional `--scene smac_3s5z`, `smac_mmm2`, `smac_corridor` or `smac_5m6m`
selects just that map's three seeds. Without `SUBMIT=YES`, only plans are printed.

The entry checks at least 5 GiB and 50,000 free home inodes, installed SC2/map
identities/assets, registered config keys, synthetic smoke tests and Slurm
`--test-only` for all missing jobs before submitting. Exact matching active or
completed names are retained. Receipts are persisted under runtime `ozstar_logs`;
look for `Submitted ... job=...` and `Manifest: ...` to confirm actual submission.
Do not treat provisional `--test-only` IDs as submitted jobs.

## Verification

`scripts/smoke_test_projected_fastrefresh.py` checks all 12 real startup configs
without an encoder/video dependency, map assets and quota guards; real episode
and parallel runner refresh behavior with a synthetic environment; all four maps'
production actor/state dimensions with Adam and RMSProp; exact-one inward
learning and bounded projection; all four hypernetwork paths; equality to the
parent projected TD updates on identical held batches; trainability after 5M;
mixed-stage replay, target copying and checkpoint contract rejection. These are
synthetic checks, not experimental results or simulator runs.
