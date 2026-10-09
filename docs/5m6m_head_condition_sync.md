# 一次性三种子图与独立 tmux 作业同步

tmux 只周期同步本仓库当前 `RUNNING` 作业的原始 W&B 数据。
不周期绘图，不扫描已结束作业，不执行最终上传/删除本地记录。
原 `recent-wandb-sync` 启动器也使用同一当前作业同步逻辑，保留会话名方便重启。

## 五张地图一次性出图

`scripts/plot_smac_five_maps_3seeds.py` 一次读取 `3m`、`8m`、`5m_vs_6m`、
`3s_vs_5z`、`6h_vs_8z` 的三种子历史数据并上传同一个 analysis run。
每张地图分别生成 mean ± sample std 的 PNG/PDF 和个别 seed 曲线 PNG，
不把不同地图混成一条胜率曲线。包含运行中和已结束的数据，不受 Slurm 状态过滤。
五图都有原线性 obs baseline；仅 5m6m 额外包括新的线性 ID、KL80、global-state。
其他地图没有提交这些条件，因此不为它们生成多余的空模型面板。
旧 attention ID 不混入新线性 ID。

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/plot_smac_five_maps_3seeds.py --local-only
```

每次手动调用生成一个新快照，只上传一次，然后退出；不会启动循环。
W&B run：`smac_five_maps_obs_head_comparison_3seeds_figures`。
目录：`/home/kyang/gomarl-runtime/gomarl-dual-branch/figures/smac_five_maps_latest_3seeds`。
缺少种子时明确显示 n/3，排队的 global-state 无测试数据时标为 0/3。

## 单独查看 5m6m 四组条件

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
绘图脚本的锁避免手动重复调用时同时生成和上传；tmux 不调用此脚本。

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

两种同步器均只同步当前运行作业，默认 600 秒启动间隔。
已移除自动绘图，旧环境中的 `UPDATE_5M6M_FIGURES` 不再生效。
自定义会话需通过原来的 `SESSION_NAME` 启动参数重启。

本地回归检查：

```bash
python scripts/smoke_test_plot_5m6m_head_condition_3seeds.py
python scripts/smoke_test_plot_smac_five_maps_3seeds.py
python scripts/smoke_test_counter_sync_tmux.py
```

绘图检查使用合成 Sacred 数据，包括 NumPy scalar 的字典序列化形式。
验证五地图的来源、各地图模型选择、缺少种子、一次性上传、PNG/PDF，
以及 5m6m 独立脚本的重复上传跳过和失败重试；不调用真实 W&B。
