# Separate seed images

The shared SMAC plotter uploads one image for each **map/model/seed**, rather
than a multi-model panel or three overlaid seeds. The separate three-seed
aggregate still compares models using their shared observed step interval.

Files and W&B keys: `<scene>__<model>__seed_<1|2|3>.png` / matching stem.
Each image contains only that seed's raw curve and its centered rolling mean
(100 test points by default). It ends at that seed's actual observed progress,
capped at the requested budget. No interpolation from a different seed/cohort.
Missing histories get an explicit unavailable-history placeholder, **not** a
claim that the job was unsubmitted, pending or failed. Slurm status must be
checked separately. Old composite files may remain locally but are not uploaded
by the new plotter. 5m6m upload dedup includes the new layout version, so existing
unchanged curve data can still receive the updated separate-image format once.

Health-only uses a separate submitter and is not included in the 24-job suite.
It is registered in both 5m6m and seven-map figure inventories by exact names:
`smac_5m6m_linear_health_only_10m_s{1,2,3}_healthcond`.

The current upload inventory contains seven maps, 25 selected map/model
cohorts and 75 exact seed names (30 names for the ten 5m6m models).
Superseded replicas, legacy attention-ID, 5M ID and historical paper VDN/QMIX
are excluded from discovery, CSV inventories, figures and new uploads, even
when their records still exist. Keep the latest 5m6m visualization ID and
3s5z visualization Obs/ID/ones cohorts. KL80 remains an explicitly labelled
5M experiment; all other selected cohorts request 10M. Missing current data
is never filled from an excluded old run. This change deletes no training
records or previously uploaded W&B snapshots.

Tests (synthetic data and mocked uploads only):
`smoke_test_separate_seed_figures.py`,
`smoke_test_plot_5m6m_head_condition_3seeds.py`,
`smoke_test_plot_smac_five_maps_3seeds.py`.
