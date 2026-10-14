# Grouping media: native logging and historical backfill

The two new 5m6m ablations (Q-change/isolated and MC-error/shared, three seeds
each) retain the original heatmap/regroup intervals. Both now log these keys:

- `mixer_input/grouping_heatmap`: exact important/unimportant membership.
- `mixer_input/grouping_membership`: a slot/group/ranking table.
- `mixer_input/important_slot_names`, `unimportant_slot_names`: latest names in
  the run summary, with `grouping_t_env` indicating the snapshot step.

The original `mixer_input/partition_probability_heatmap` remains unchanged.
JSON snapshots also include the actual W&B run ID/name, seed and map. Existing
head-group experiments use the corresponding `slot_input/` prefix.

## Backfill previously saved partitions

From `/home/kyang/code/gomarl-dual-branch`:

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_backfill_grouping_heatmaps.py
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_backfill_grouping_heatmaps.py --upload
```

Scope: the nine old head-group runs, six old QMIX group runs, three recent
MC-error/isolated runs, and six new ablations. Gates and baselines are not group
experiments and are not selected. Only locally present sources are processed;
`NO LOCAL SOURCE` distinguishes not-yet-run/deleted sources from successes.
Missing saved partitions are reported as failures, not reconstructed from final
weights or fabricated. A missing PNG can be regenerated only from its actual
archived JSON membership.

Backfill uploads all saved PNG/JSON snapshots to the **original training run**,
then adds latest heatmap, latest membership table, a `grouping_evolution` image,
and a `grouping_history` table containing every saved step and its heatmap file
path (all referenced PNGs are uploaded in Files).
Where original offline grouping-media files still exist, the tool also uploads
them at their exact original `media/images/...` paths, repairing old history
image references without adding new history rows. The original heatmap summary
key is also populated with the latest archived partition.
Look in the run's `mixer_input` or `slot_input` media; raw snapshots are under
Files → `grouping_backfill`. No new experiment run is created. Public API
file/summary updates do not resume training, insert history rows, change the
run's status, or alter win-rate/Q curves. A source ID/name mismatch or ambiguous
legacy attempt is rejected. Running cloud runs get file uploads only; their
summary is deferred to prevent concurrent summary writes. Rerun after completion.

Uploads share the existing synchronization/cleanup lock. If locked, retry later.
Successful file fingerprints are saved in `grouping_backfill/uploaded_files.json`;
repeating the command skips unchanged files and repairs the summary. Do not
delete the generated `grouping_backfill` folder or this ledger while uploads
are pending. Up to four files upload concurrently, without downloading training
histories. PNG staging uses symlinks rather than duplicating the large image
archives on quota-limited `/home`; generated tables and evolution images are
small additional files. Keep the original snapshots until upload finishes.
Previously removed local data cannot be backfilled by this tool.

Already-running Python processes will not reload this logging change. The
backfill tool supports their saved JSON too; new starts include the native keys.
The six-job submission smoke checks now exercise actual W&B Image construction
and SDK-readable table files for both ablations, plus offline-only backfill
checks before sbatch. Native tables use W&B's run-file JSON format to avoid the
server-introspection request that SDK 0.18.7 can make even in offline Table logging.
