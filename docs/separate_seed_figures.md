# Combined three-seed model panels

The shared SMAC plotter restores one image per map, with one panel per model
and all three seed curves overlaid in that panel. Seeds use distinct colors;
raw curves are faint and centered rolling means are solid. The separate
three-seed aggregate still compares models using their shared observed interval.

Files and W&B keys: `<scene>_individual_seeds.png` / matching stem.
Each seed's rolling mean uses 100 test points by default and ends at its actual observed progress,
capped at the requested budget. No interpolation from a different seed/cohort.
Missing histories get an explicit unavailable-history placeholder, **not** a
claim that the job was unsubmitted, pending or failed. Slurm status must be
checked separately. Old standalone seed files may remain locally but are not
uploaded by the restored plotter. 5m6m upload dedup includes the layout version,
so unchanged curve data can still receive the restored combined format once.

Health-only uses a separate submitter and is not included in the 24-job suite.
It is registered in both 5m6m and seven-map figure inventories by exact names:
`smac_5m6m_linear_health_only_10m_s{1,2,3}_healthcond`.

The inventory contains seven maps, 25 current map/model cohorts plus two
explicit 5m6m mixer fallback cohorts: 81 exact names (36 for 5m6m).
For 5m6m VDN and QMIX independently, show the historical 5M paper cohort
only when no seed of the new 10M cohort has test history. Labels explicitly
say historical 5M; never merge seeds or extend the old curve to 10M. If any
new 10M seed has data, show only the new cohort and retain its missing seeds.
If both cohorts are unavailable, retain the new 10M missing-data panel.
Raw CSV inventories retain both cohorts for audit, while figures apply the
whole-cohort selection. No training job is submitted by this change.
Superseded replicas, legacy attention-ID and 5M ID remain excluded.
Keep the latest 5m6m visualization ID and
3s5z visualization Obs/ID/ones cohorts. KL80 remains an explicitly labelled
5M experiment; current cohorts otherwise request 10M. Missing current data
is never filled from other excluded old runs. This change deletes no training
records or previously uploaded W&B snapshots.

Tests (synthetic data and mocked uploads only):
`smoke_test_separate_seed_figures.py`,
`smoke_test_plot_5m6m_head_condition_3seeds.py`,
`smoke_test_plot_smac_five_maps_3seeds.py`.
