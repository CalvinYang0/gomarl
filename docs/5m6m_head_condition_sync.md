# 5m6m head-condition 三种子图与 tmux 同步

`scripts/plot_5m6m_head_condition_3seeds.py` 比较四组精确 run 名称：

| 组别 | 名称模板 | 预算 |
| --- | --- | --- |
| 原线性 obs baseline | `smac_5m6m_linear_obs_baseline_10m_s{seed}_valuediag` | 10M |
| 新线性 ID baseline | `smac_5m6m_linear_id_5m_s{seed}_idkl80fix` | 5M |
| 新线性 direct KL80 | `smac_5m6m_linear_kl80_direct_linearonly_5m_s{seed}_idkl80fix` | 5M |
| global-state baseline | `smac_5m6m_linear_global_state_10m_s{seed}_statecond` | 10M |

每组 seed 为 1/2/3。使用 `test_battle_won_mean`，同一 seed 只取最新 attempt；
平均曲线只画当前可用 seed 的共同区间，图例明确写 `n/3 seeds`。
缺少的数据保留为空，不以历史 attention ID 补位。
输出包含 mean ± sample std 的 PNG/PDF、四组分别展开的种子图，以及原始曲线和 seed inventory CSV。
平滑沿用 centered window=100 test points；需要看未平滑曲线可以加 `--mean-window 1`。

默认目录：`/home/kyang/gomarl-runtime/gomarl-dual-branch/figures/5m6m_head_condition_comparison_3seeds`。
W&B analysis run 名称：`smac_5m6m_head_condition_comparison_3seeds_figures`。
每次成功上传保存数据指纹；曲线、种子清单或平滑配置相同则跳过上传，上传失败留待下轮重试。
两个 tmux 同步器共享绘图锁，避免同时生成和上传。

## 立即生成并上传

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/plot_5m6m_head_condition_3seeds.py --local-only
```

`--local-only` 从本地 W&B/Sacred 读取训练数据，图仍上传到 W&B。
加 `--no-upload` 仅生成本地图片。没有任何测试数据时正常跳过，不上传空图。

## 更新正在运行的 tmux 同步器

先更新仓库，然后在原 tmux 所在登录节点重启同步会话。
`restart` 更新的是同步会话，Slurm 训练继续运行。

```bash
if tmux has-session -t '=recent-wandb-sync' 2>/dev/null; then
  bash scripts/ozstar_recent_wandb_sync_tmux.sh restart
else
  bash scripts/ozstar_wandb_sync_tmux.sh restart
fi
```

两种同步器均在每轮原始 run 同步之后更新三种子图，默认 600 秒启动间隔。
`UPDATE_5M6M_FIGURES=NO` 可关闭自动绘图；`FIGURE_TIMEOUT=600` 控制单轮绘图及上传超时。
自定义会话需通过原来的 `SESSION_NAME` 启动参数重启。

本地回归检查：

```bash
python scripts/smoke_test_plot_5m6m_head_condition_3seeds.py
python scripts/smoke_test_counter_sync_tmux.py
```

绘图检查使用合成 Sacred 数据，包括 NumPy scalar 的字典序列化形式。
验证 PNG/PDF、正确的三种子来源、无数据跳过、重复上传跳过和失败重试；不调用真实 W&B。
