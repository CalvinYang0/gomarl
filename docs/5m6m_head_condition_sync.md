# 一次性三种子图与独立 tmux 作业同步

以后启动的 SMAC 实验默认每 1M 步记录前 10 局正常测试视频。
更新和编码依赖检查见 [测试战斗视频](test_battle_videos.md)；不重启现有作业。

## 8m9m / 6h8z 的正确线性 ID 对照

入口 `scripts/ozstar_submit_linear_id_8m9m_6h8z_10m_3seeds.py` 只包含这两张地图，
每张 seed 1/2/3、10M、48 小时、24G。沿用 Obs 的主网络和训练设置；
GRU 输入保留局部 obs、上一步动作、agent ID，生成头条件只来自线性编码的 ID。
仅主 TD 损失，无 gate、KL、attention 策略替代或特殊 HyperMARL 初始化。

名称为 `smac_8m9m_linear_id_baseline_10m_s{seed}_valuediag` 和
`smac_6h8z_linear_id_baseline_10m_s{seed}_valuediag`。
前者沿用之前正确线性 ID 的名称；同名运行中/成功完成作业会保留，不重复提交。
历史 `*_id_baseline_*` attention 组不复用、不取消。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/ozstar_submit_linear_id_8m9m_6h8z_10m_3seeds.py
```

不设置 `SUBMIT=YES` 只预览计划。提交前自动校验地图、磁盘配额、配置、
真实地图尺寸及生产模型尺寸的 GRU/ID 前向与梯度路径，再做 Slurm 预检。
tmux 已按本仓库运行中任务动态筛选，不需要另加固定作业 ID。
现有六地图图入口暂不自动纳入这两组新 ID；不能以空面板或旧 ID 充当新结果。

tmux 只周期同步本仓库当前 `RUNNING` 作业的原始 W&B 数据。
不周期绘图，不扫描已结束作业，不执行最终上传/删除本地记录。
原 `recent-wandb-sync` 启动器也使用同一当前作业同步逻辑，保留会话名方便重启。

## 六张地图一次性出图

`scripts/plot_smac_six_maps_3seeds.py` 一次读取 `3m`、`8m`、`8m_vs_9m`、`5m_vs_6m`、
`3s_vs_5z`、`6h_vs_8z` 的三种子历史数据并上传同一个 analysis run。
每张地图分别生成 mean ± sample std 的 PNG/PDF 和个别 seed 曲线 PNG，
不把不同地图混成一条胜率曲线。包含运行中和已结束的数据，不受 Slurm 状态过滤。
六图都有原线性 obs baseline；仅 5m6m 额外包括修正后的 5M 线性 ID、KL80、global-state、全 1、单局 timestep、Obs+实体 ID，以及明确标注为历史 5M 的 VDN/QMIX 对照。
`8m` 是 8 vs 8，`8m_vs_9m` 是 8 vs 9，按精确 run 名称分别读取，标题和输出文件独立。
旧入口 `plot_smac_five_maps_3seeds.py` 保持兼容，也会生成全部六张地图。
其他地图没有提交这些条件，因此不为它们生成多余的空模型面板。
旧 attention ID 不混入新线性 ID。

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/plot_smac_six_maps_3seeds.py --local-only
```

每次手动调用生成一个新快照，只上传一次，然后退出；不会启动循环。
W&B run：`smac_six_maps_obs_head_comparison_3seeds_figures`。
目录：`/home/kyang/gomarl-runtime/gomarl-dual-branch/figures/smac_six_maps_latest_3seeds`。
缺少种子时明确显示 n/3，排队的 global-state 无测试数据时标为 0/3。

## 单独查看 5m6m 九组对照

`scripts/plot_5m6m_head_condition_3seeds.py` 比较九组精确 run 名称：

| 组别 | 名称模板 | 预算 |
| --- | --- | --- |
| 原线性 obs 可视化新跑 | `smac_5m6m_linear_singlehead_baseline_10m_s{seed}_recheck` | 10M |
| 正确线性 ID（5M 独立对照） | `smac_5m6m_linear_id_5m_s{seed}_idkl80fix` | 5M |
| 新线性 direct KL80 | `smac_5m6m_linear_kl80_direct_linearonly_5m_s{seed}_idkl80fix` | 5M |
| global-state baseline | `smac_5m6m_linear_global_state_10m_s{seed}_statecond` | 10M |
| 全 1 超网络输入 | `smac_5m6m_linear_ones_10m_s{seed}_signalcond` | 10M |
| 单局 timestep 超网络输入 | `smac_5m6m_linear_timestep_10m_s{seed}_signalcond` | 10M |
| Obs + 实体 ID | `smac_5m6m_linear_obs_entity_id_10m_s{seed}_entityidcond` | 10M |
| VDN（历史 paper 对照） | `smac_5m6m_paper_vdn_5m_s{seed}` | 5M |
| QMIX（历史 paper 对照） | `smac_5m6m_paper_qmix_5m_s{seed}` | 5M |

VDN/QMIX 只读取这六个 5m6m 的历史精确名称，不混入 8m9m 或 Corridor 的新运行，不延伸到 10M。
它们是不同训练批次的参考对照，不是严格同批次、同预算的消融。若历史本地记录已清理，
省略 `--local-only` 可同时查找 W&B 云端，仍不回退到其他模型或预算的记录。

每组 seed 为 1/2/3。使用 `test_battle_won_mean`，同一 seed 只取最新 attempt；
平均曲线只画当前可用 seed 的共同区间，图例明确写 `n/3 seeds`。
缺少的数据保留为空，不以历史 attention ID 补位。
ID 只展示已有完整三种子结果的修正后 5M 线性组，明确标注实际预算，不延伸到 10M。
当前不选择没有测试数据的 10M ID 组，因此不生成该空面板；不删除其训练记录或修改提交配置。
Obs 使用专门的可视化重跑记录；旧的已完成 Obs 不代替新的缺失数据。
输出包含 mean ± sample std 的 PNG/PDF、九组分别展开的种子图，以及原始曲线和 seed inventory CSV。
九组种子面板采用三列、三行排列；汇总图将图例放在图外，避免遮住曲线。
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
验证六地图的来源、8 vs 8 / 8 vs 9 不混用、各地图模型选择、缺少种子、一次性上传、PNG/PDF，
以及 5m6m 独立脚本的重复上传跳过和失败重试；不调用真实 W&B。
