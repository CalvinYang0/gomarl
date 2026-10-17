# 8m9m: update the locally available six-model comparison

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_8m9m_local_available_3seeds.py
```

Expected models, three planned seeds each: fresh ordinary QMIX, fresh VDN,
fresh static state gate p0=.1/no KL, all-ones mixer condition, static gate plus
probability L1, static gate plus variance-weighted probability loss. No 5m6m
experiments or historical 5M substitutions are included.

Data comes only from current local offline/Sacred logs and verified raw
`figures/*/seed_curves.csv` snapshots with sibling inventories. Exact run names,
IDs, attempts and deduplication use the same rules as the local 5m6m uploader.
No cloud run/history reads occur. Upload is enabled by default; use
`--no-upload` for a local-only preview. Models with zero readable seeds are
omitted from figures; partial and missing seeds are labelled honestly.

Output: runtime `figures/8m9m_local_available_qmix_variants_3seeds`; other source
CSVs are preserved. The W&B figure run contains an overview and readable
per-model seed plots in the 8m9m section beside Media. Printed run/workspace
URLs identify the new snapshot. Training and tmux sync are not modified.

Synthetic regression (cloud history forbidden, upload mocked):
`python scripts/smoke_test_plot_8m9m_local_available.py`.
