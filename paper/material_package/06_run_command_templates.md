# Run Command Templates

These commands follow the preferred human input order: open tmux, enter the repo, update code, activate environment, export runtime variables, then run one experiment.

Use one server/GPU per experiment.

## V100-16GB Dynamic Linear on Corridor

```bash
tmux new -s linear_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_linear_interaction_hypercond seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_linear_interaction_hypercond_v100_amp64_s1 wandb_run_name=corridor_rpg_linear_interaction_hypercond_v100_amp64_s1
```

## V100-16GB Fixed Linear Control on Corridor

```bash
tmux new -s fixed_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_fixed_linear_structured_maker seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_fixed_linear_structured_maker_v100_amp64_s1 wandb_run_name=corridor_rpg_fixed_linear_structured_maker_v100_amp64_s1
```

## V100-16GB Residual Variant on Corridor

```bash
tmux new -s residual_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_residual_interaction_hypercond seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_residual_interaction_hypercond_v100_amp64_s1 wandb_run_name=corridor_rpg_residual_interaction_hypercond_v100_amp64_s1
```

## V100-16GB FiLM Variant on Corridor

```bash
tmux new -s film_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_film_interaction_hypercond seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_film_interaction_hypercond_v100_amp64_s1 wandb_run_name=corridor_rpg_film_interaction_hypercond_v100_amp64_s1
```

## V100-16GB MoE Variant on Corridor

```bash
tmux new -s moe_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_moe_interaction_head seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_moe_interaction_head_v100_amp64_s1 wandb_run_name=corridor_rpg_moe_interaction_head_v100_amp64_s1
```

## V100-16GB Smooth Variant on Corridor

```bash
tmux new -s smooth_v100_amp_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_smooth_linear_interaction_hypercond seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 name=corridor_rpg_smooth_linear_interaction_hypercond_v100_amp64_s1 wandb_run_name=corridor_rpg_smooth_linear_interaction_hypercond_v100_amp64_s1
```

## CPU 8-Core Template

```bash
tmux new -s corridor_cpu_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export WANDB_MODE=online
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python3 src/main.py --config=clean_hyper --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_linear_interaction_hypercond seed=1 use_cuda=False use_wandb=True wandb_mode=online batch_size_run=4 name=cpu_corridor_rpg_linear_interaction_hypercond_s1 wandb_run_name=cpu_corridor_rpg_linear_interaction_hypercond_s1
```

## Visualization Run Template

Use this only for selected baseline/fixed runs because trace videos add test-time overhead.

```bash
tmux new -s trace_linear_s1
cd /home/vipuser/code/clone/gomarl
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
conda activate benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/main.py --config=clean_hyper_gpu_v100 --env-config=sc2 with env_args.map_name=corridor clean_model_type=rpg_linear_interaction_hypercond seed=1 use_cuda=True use_wandb=True wandb_mode=online batch_size=64 save_battle_trace=True battle_trace_interval=1000000 battle_trace_frame_stride=1 name=corridor_rpg_linear_interaction_hypercond_v100_trace_s1 wandb_run_name=corridor_rpg_linear_interaction_hypercond_v100_trace_s1
```

