# Corridor baseline replicas

Six fresh runs on `corridor`: Linear observation-conditioned single-head
baseline and historical ID-conditioned baseline, each with seeds 1/2/3.
The ID model retains its historical attention encoder; this is not a matched
Linear-obs versus Linear-ID architecture ablation.

Both use main TD only, no gate, KL or QME. Settings match the 5m6m value
diagnostic suite: 10,050,000 environment steps, two-day Slurm limit, 24G/32G,
eight rollout workers, batch 128, buffer 5000; 32 test episodes every 10K steps
and scalar value diagnostics every 100K steps. Checkpoints, trajectories,
parameter PCA and battle traces are disabled; runtime stays under `/home`.
See `5m6m_value_diagnostics.md` for diagnostic definitions and limitations.

Submission entry point:
`scripts/ozstar_submit_corridor_value_diagnostics_10m_3seeds.py`.
Default invocation validates and prints plans. `SUBMIT=YES` additionally checks
quota, runs model smoke tests and validates all Slurm requests before submitting.
Exact-name active/completed jobs are retained, preventing duplicate submissions.
Existing 5m6m jobs are not cancelled or changed. One GiB minimum free quota is
only a guard, not a guarantee against later disk exhaustion.

## Fixed-head controls

`scripts/ozstar_submit_corridor_vdn_qmix_10m_3seeds.py` adds six jobs: VDN and
QMIX, each with seeds 1/2/3. Both use the historical paper `qmix_minimal`
shared recurrent agent with a fixed two-layer MLP action head. No generated
agent head or gate/KL/QME losses. Only their mixer differs (`vdn` / `qmix`).
They share the 10M/two-day budget, 24G RAM, rollout/optimization, evaluation,
home storage and scalar diagnostics of the obs/ID suite. For VDN the mixing
sensitivity is always one and no learned mixer-weight statistics exist.
QMIX still has a state-conditioned hypernetwork in its mixer; this comparison
is about generated agent heads, not removing every hypernetwork.

Submission checks real Corridor dimensions with synthetic padded episodes for
both fixed-head learners and diagnostic paths. No StarCraft launch is performed
by the smoke test; actual training remains a cluster-side verification.
New runs use separate names/group and do not cancel or duplicate obs/ID jobs.
