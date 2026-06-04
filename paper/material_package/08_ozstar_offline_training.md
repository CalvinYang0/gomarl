# OzSTAR Offline Training Mode

Use this workflow when compute nodes cannot resolve external services such as W&B.

## 1. Smoke Test

Submit a short CPU job from the login/test node:

```bash
cd /home/kyang/code/gomarl
MAP_NAME=3m MODEL_TYPE=qmix_minimal SEED=1 T_MAX=10000 TEST_INTERVAL=5000 BATCH_SIZE_RUN=1 BATCH_SIZE=8 BUFFER_SIZE=50 USE_WANDB=False sbatch scripts/ozstar_train_offline.sbatch
```

Check logs:

```bash
squeue -u kyang
tail -f ozstar_logs/gomarl_<job_id>.out
tail -f ozstar_logs/gomarl_<job_id>.err
```

A successful smoke test should show SC2 launch messages, `Game has started`, `Recent Stats`, and no Python traceback.

## 2. Offline W&B Training

For real experiments, keep W&B offline on compute nodes:

```bash
cd /home/kyang/code/gomarl
MAP_NAME=corridor MODEL_TYPE=rpg_linear_interaction_hypercond SEED=1 T_MAX=10050000 BATCH_SIZE_RUN=1 BATCH_SIZE=32 BUFFER_SIZE=500 USE_WANDB=True WANDB_MODE=offline RUN_NAME=corridor_rpg_linear_interaction_hypercond_ozstar_s1 sbatch scripts/ozstar_train_offline.sbatch
```

The job writes stdout/stderr under `ozstar_logs/` and W&B offline files under `wandb/`.

## 3. Sync After Training

Run sync on a login/test node with network access:

```bash
cd /home/kyang/code/gomarl
bash scripts/ozstar_sync_wandb.sh
```

If needed, narrow the sync target:

```bash
PATTERN='wandb/offline-run-20260604*' bash scripts/ozstar_sync_wandb.sh
```

## 4. Notes

- The script directly uses `/home/kyang/.conda/envs/marl_cpu/bin/python`, so it does not require `conda activate` or `mamba` on compute nodes.
- The default `SC2PATH` is `/home/kyang/StarCraftII`.
- Compute nodes should use `WANDB_MODE=offline`; use the login/test node to upload results later.
- If `SC2 binary exists` prints `False`, fix the StarCraft II installation before submitting long jobs.

