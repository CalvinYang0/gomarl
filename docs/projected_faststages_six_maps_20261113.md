# Projected init1 faststages8M vs QMIX vs all-ones

Fresh 10M runs (historical `t_max=10050000` margin), seeds 1/2/3, three
models per map, 54 jobs total. Each job requests 28 CPUs, 32G and 24h.

| Scene key | Environment map | Win metric |
| --- | --- | --- |
| `smac_6h8z` | `6h_vs_8z` | `test_battle_won_mean` |
| `smac_10m11m` | `10m_vs_11m` | `test_battle_won_mean` |
| `smac_3s_vs_5z` | `3s_vs_5z` (not `3s5z_vs_3s6z`) | `test_battle_won_mean` |
| `grf_counter` | `academy_counterattack_easy` | `test_game_win_mean` |
| `grf_pass` | `academy_pass_and_shoot_with_keeper` | `test_game_win_mean` |
| `grf_3v1` | `academy_3_vs_1_with_keeper` | `test_game_win_mean` |

- `qmix`: live global mixer state every timestep.
- `qmix_state_ones`: same clean learner and mixer, but all four parameter
  paths use the same-size vector of ones instead of global state.
- `qmix_projected_init1_faststages8m`: parent projected gate/TD recipe;
  trainable field coefficients start at exactly 1, are clamped to [0,1]
  after optimizer steps, and interpolate held mixer state toward one.
  Training starts with episode-start state, reaches live state at 8M;
  evaluation is live from the start. No binary freeze or auxiliary loss.

Keep the historical GRU and two-layer ELU action head and live agent
observations in all three variants. Preserve each environment's original
reward and features. Disable video, GRF episode/goal dumps and checkpoints;
retain W&B offline scalars and applicable diagnostic images.

## OzSTAR

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_projected_faststages_six_maps_54jobs.py
```

Omit `SUBMIT=YES` for a plan. `--scene` selects one map (9 jobs);
`--domain smac` or `--domain grf` selects 27 jobs. All selections share the
same job identities, receipt group and duplicate-protection lock. The existing
5 GiB/50,000 free home inode check is a startup floor, not a guarantee of
space for 54 complete runs. For limited space, run one scene at a time and
recheck quota before the next wave. No upload, cleanup or cancellation is
part of this submission.

Selected SC2 asset checks and isolated GRF reset/step probes precede the
synthetic training smoke test and Slurm `--test-only`. Confirm submission with
real `Submitted ... job=...` lines and a manifest in runtime `ozstar_logs`.
An already-active or completed exact job is retained on retry.
