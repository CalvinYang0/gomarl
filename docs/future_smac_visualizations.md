# 后续 SMAC 实验的可视化覆盖

规则接在公共的 `run_sequential` 训练入口，不依赖某个提交脚本或地图。
更新代码后启动的新训练任务默认启用；不修改正在运行的进程，不补录过去战斗，
也不取消、重启或提交作业。已有独立 checkpoint evaluation 分支不在本次调度范围。

## 记录什么

每跨过一个 1M 环境步节点，在随后第一轮**普通测试**中取前 10 局，包含输局和赢局，
不挑高奖励样本，不增加测试 rollout。每个 seed 单独保存。

- 战斗视频：模拟器坐标、单位血量、采取的动作及目标，用现有俯视视频记录。
  录像和普通胜率测试使用相同的动作，不是重新生成的一套战斗。
- 所有当前模型共有的决策参数敏感度图：逐个已学习的 policy 参数块计算
  `mean |parameter × d(top1-top2 Q)/d(parameter)|`，另存未乘参数的梯度均值。
- 所有当前模型共有的 Obs 字段 × agent 图：当前自身 Obs 对两个最优可用动作
  Q 差距的绝对梯度，乘本次有效状态字段标准差。它包含主网络和超网络的真实路径，
  **不是单独的超网络输入归因**。字段标签无可用真实布局时只标 `obs_0...`。
- policy 确实使用全局 state 时，另有 `state_sensitivity` 图。按真实 state 数组索引
  标记，展示未按标准差缩放的梯度；不能直接与 Obs 色条比较。
- 无 gate 的原始 Linear Obs 额外保留原来的超网络专属三张图：生成头敏感度、
  固定主网络隐藏状态后的决策敏感度、字段组均值替换干预。

公共诊断已覆盖 Obs、ID、全 1、episode timestep、global-state、Obs+entity-ID、
直接 KL、attention，以及固定头 VDN/QMIX；也验证了 MMM2 的实际网络维度。
ID/常量方案仍有**主网络**的 Obs 依赖，不把固定超网络输入伪装成动态 Obs 归因。
VDN/QMIX 的图衡量 agent policy，不衡量 mixer 权重或训练 TD 信用分配；
普通去中心化 policy 不使用全局 state，不会人为给它加 state 输入。

公共诊断每节点最多 64 个有效决策探针，按 agent 分层、确定性抽样，排除死亡、
padding 和终止后的状态。用相同配置/权重的独立 MAC 回放记录的测试历史，固定
每步传入的 recurrent hidden state，不反传历史，不改原模型缓存/梯度/权重/RNG。
原 hyper-only 诊断仍使用最多 256 个探针。

**解释边界：**这些是局部连续敏感度，不是胜率因果贡献、语义理解证据或信用分配。
参数化和尺度会影响结果；零梯度不证明无用，常量字段的标准差缩放分数会为零。
比较前后阶段时应读原始数值并固定色条，不能只看颜色。

## W&B 和本地记录

请在对应的**实际训练 run**里找 Media，不是在三种子汇总曲线的 figure run 里。

- `test_battle_video/episode_01` 至 `episode_10`
- `test_policy_importance/parameter_sensitivity`
- `test_policy_importance/observation_sensitivity`
- `test_policy_importance/state_sensitivity`（policy 使用 state 且有非零局部梯度时）
- `test_hyper_obs_importance/head_sensitivity`、`decision_sensitivity`、`group_ablation`
  （仅上面明确支持的 raw Linear Obs）

状态标量：`test_battle_video/{collected,rendered,failed}`、
`test_policy_importance/{enabled,samples,episodes,failed}`、
`test_hyper_obs_importance/{enabled,samples,episodes,failed}`。

本地 `<local_results_path>` 下有 `test_battle_videos` 的 MP4/轨迹 JSON、
`policy_importance` 的 PNG/JSON/参数 CSV，以及 `test_visualization_inventory` 的
逐节点清单。都按 unique token、seed、run-name hash 分开。
原始 JSON/CSV 不自动成为 W&B artifact；视频和 PNG 通过正常 W&B media 路径记录。
**离线记录成功不等于云端上传完成**，清单明确标 `cloud_upload_verified=False`；
仍须完成该 run 的正常 W&B 同步。新增代码未宣称现有集群 run 已上传媒体。

## 不允许静默遗漏

默认 `test_visualizations_required=True`。缺少编码依赖、关闭公共可视化开关或实际普通
测试局数不足 10，会在 SC2 启动前报错。测试局数按当前 runner 的 batch 向下取整
检查，例如 `test_nepisode=10,batch_size_run=8` 实际只有 8 局，不会私自增开测试。
节点上录像/诊断失败或缺图时先写清单再报错，不再悄悄跑完但没有可视化。

这是诊断覆盖要求，不修改训练损失、权重、head 输入、gate 设置或探索调度。
诊断会增加计算/绘图时间和媒体磁盘用量，提交预算需保留余量。
短调试任务可明确设 `test_visualizations_required=False`，再关闭相应记录开关；
正式后续实验不要关闭。

## 更新和验证（不提交作业）

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
/home/kyang/.conda/envs/marl_cpu/bin/python -m pip install 'imageio>=2.9' imageio-ffmpeg Pillow 'matplotlib>=3.5' &&
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/smoke_test_policy_importance.py &&
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/smoke_test_periodic_battle_videos.py
```

测试不启动 SC2、不联网上传，不生成真实实验结果。验证生产模型回放 Q 一致、
参数/缓存/梯度/RNG 隔离、真实 PNG/MP4 编码、调度和失败保护。
真实计算节点上的 SC2 录像及 W&B 云端同步需要后续新任务验证。
