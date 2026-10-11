# 8m9m health-only：三种子、10M

补充目的：验证 5m6m 的 health-only 收益是否适用于 8m_vs_9m。
这组只改变生成行动头的条件，不改变主 GRU 或 mixer 的真实环境输入。

- 地图：`8m_vs_9m`，8 Marines vs. 9 Marines；不是 `8m`。
- 种子：1/2/3；预算 10M，`t_max=10050000`；每作业 24G、48h、28 CPU。
- 条件输入仍为 85D；保留 9 敌军、7 友军和自己的 17 个 health 字段，其他 68 个字段置零。
- 保留字段由模型的真实语义布局确定，不从 5m6m 拷贝固定下标。该地图零基下标应为
  `8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,83,84`。
- 血量来自本地实际 Obs，不补充视野外的真实血量；0 仍可能暗含不可见/死亡信息。
- 主 GRU 保留完整 Obs、上一步动作、agent ID，mixer 保留原全局 state。
- 同原 Linear Obs 初始化、生成头、主 TD、Double-Q、优化器和采样设置；无 gate、KL 或辅助 TD。
- 每 10K 步测试 32 局；每 100K 价值诊断；每 1M 的前 10 局正常测试战斗视频、
  policy 输入/参数敏感度和 hyper-only 诊断，包括 `decision_sensitivity`。
- 媒体依赖缺失或预检失败时，不静默关闭诊断后继续训练。

精确运行名：`smac_8m9m_linear_health_only_10m_s{1,2,3}_healthcond`。
新提交入口复用 5m6m health-only 的匹配设置，但使用独立地图与名称。
重复运行保留 exact-name pending/running/completed 任务，只提交缺失任务；不取消其他作业。
配置、真实地图维度、模型梯度/target/TD/诊断及全部缺失作业的 Slurm 预检通过后才提交。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_8m9m_health_only_10m_3seeds.py
```

不设 `SUBMIT=YES` 只预览，不启动训练。至少要求 2 GiB /home 配额余量，这只是启动门槛，
不是录像总容量保证。只有 `Submitted ... job=...` 或 Slurm 记录才代表已提交；
代码推送、模拟调度检查、注册到绘图清单，都不代表取得了真实实验结果。

七地图入口的 8m9m 已加这组三种子，8m9m 当前为 Obs、ID、ones、health、VDN、QMIX 六组，
共 18 个运行。沿用上一轮按 `smac_8m9m` 过滤的独立出图命令，也会纳入此组；
没有测试数据时保留缺失标记，不拿 5m6m health 数据或其他模型补位。
仅上传已生成旧 PNG 的命令不会自动刷新这组，需要重新读取/绘图。

本地配置/模拟调度测试：

```bash
python scripts/smoke_test_submit_8m9m_health_only.py
python scripts/smoke_test_submit_5m6m_health_only.py
python scripts/smoke_test_plot_smac_five_maps_3seeds.py
```

真实尺寸的模型检查需要集群已有的 PyTorch、SMAC/PySC2 环境，但不启动 SC2 或上传数据：

```bash
python scripts/smoke_test_8m9m_health_only.py
```

当前本地无法通过集群 SSH 自动认证；这里只准备提交器和测试，不宣称新作业已启动。
