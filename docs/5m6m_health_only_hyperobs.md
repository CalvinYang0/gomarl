# 5m6m health-only hypernetwork observation control

Three new seeds (1/2/3), nominal 10M environment steps; `t_max=10050000` allows
the final 10M diagnostic milestone. Single linear branch, GRU-generated Q head,
same initialization, encoder dimensions, mixer, optimizer and Double-Q/main TD
objective as Linear Obs. No learned gate, KL, auxiliary TD or new warmup.

Only hypernetwork input changes: preserve the existing 55D local Obs layout,
retain 11 health slots, zero the other 44 fields. Main policy still receives
the full local Obs, previous action and agent ID. Mixer still receives its
original global state. Training, target TD, behaviour and evaluation use the
same filter; previous/next condition Obs are filtered too.

Zero-based retained raw Obs columns: `8,13,18,23,28,33` (six enemies),
`38,43,48,53` (four allies), `54` (self). The filter derives these from the
production adapter's semantic fields, not hardcoded offsets. This is local
observed normalized health, NOT omniscient health: out-of-sight/dead slots keep
the source Obs zeros. No positions, distances, visibility flags, shields or
entity IDs are added to the hypernetwork. Health patterns may nevertheless
implicitly indicate visibility/life, so this experiment is not evidence that
health is free of those correlations.

Every 1M: ten ordinary test battle videos (positions/actions/health), common
policy parameter/Obs sensitivities, plus hyper-path-only head/decision/group
diagnostics. The latter explicitly includes the health filter: all non-health
hyper-only derivatives are zero. Whole-policy Obs sensitivity can remain
nonzero on other fields through the unchanged GRU. Missing required media
dependencies abort before training; offline files are not proof of cloud upload.

Job names: `smac_5m6m_linear_health_only_10m_s{1,2,3}_healthcond`.
24G, 48h each. New three-job suite; does not change/cancel the previous 24 jobs.
Submission checks all missing jobs before any scheduler writes and retains
exact-name pending/running/completed jobs on repeat. Failed attempts may retry.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_5m6m_health_only_10m_3seeds.py
```

Both `plot_5m6m_head_condition_3seeds.py` and the seven-map plot include this
new cohort by exact run names; absent histories remain marked missing, never
substituted with older Obs/ID curves. Local tests use synthetic data and no
StarCraft II launch, real scheduler submission or W&B upload.
