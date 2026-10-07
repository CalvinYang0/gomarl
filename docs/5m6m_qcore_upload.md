# Individual core-Q histories

`scripts/ozstar_upload_5m6m_qcore.py` reads the latest offline attempt for each
of the six fresh obs/ID 5m6m runs, including completed jobs. It preserves other
finite scalar histories and keeps only ten `test_value/` diagnostics: joint Q
mean/std, pooled agent Q mean, MC bias/absolute error means, mixing sensitivity
mean, two mixer-weight means, natural termination fraction and calibration count.
No per-agent statistics, max-absolute statistics, media or checkpoints.

Each source run gets a separate, stable-ID `_qcore` mirror in the same project.
Original training runs are not overwritten or deleted. Source names/IDs and
filter keys are recorded in config; the mirrors are filtered history copies,
not new experiments or seed aggregate runs. Uploads resume from the cloud's
next history step. Local fingerprints are saved atomically only after successful
finish; unchanged files (usually completed jobs) are skipped on future cycles.
Late final writes are included even after a job leaves `squeue`.

Future training code emits only these ten diagnostic keys. Existing processes
keep their imported code, so their old 88-key histories are filtered at upload.
This does not change training losses, forward computation or seeds.

Tests: diagnostic unit smoke and scalar-filter/name selection checks. Real W&B
authentication/network upload must be verified on the cluster; no actual cloud
upload was performed locally.
