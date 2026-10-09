# 5m6m：常量与单局时钟超网络输入

两组各 seed 1/2/3、10M（提交预算 10,050,000）、48h、24G。
主网络仍输入原始局部 obs、上一动作和 agent ID，并输出 64D GRU 隐状态。
mixer 仍用原始全局 state。只把进入原 55→64 线性条件编码器的 obs 换掉：

| 组别 | 超网络的原始条件输入 | Run 名称 |
| --- | --- | --- |
| 全 1 | 55 维全为 1；所有 agent 和局相同 | `smac_5m6m_linear_ones_10m_s{seed}_signalcond` |
| 单局时钟 | 55 维全为当前决策步 t；每局 0,1,2,… 重置 | `smac_5m6m_linear_timestep_10m_s{seed}_signalcond` |

使用原始 t，不除以回合上限，不用 t_env，不添加 ID/state/动作到超网络条件。
在线策略、测试、replay 学习和 target MAC 使用同一规则；下一步目标使用 t+1。
两组保留原 Obs 的全部模块、参数数量和初始化，只改变条件输入。
只有原主 TD 损失；没有 gate、KL、辅助 TD、attention 替代或特殊初始化。
主 TD/Double-Q/target 更新、探索、replay 设置和 QMIX 优化逻辑均不变。
每 1M 步记录正常测试前 10 局的视频；提交前检查编码依赖。

“常量输入”不等于冻结超网络：同一个 checkpoint 内头不随环境/时钟变化，
但参数会随训练更新。时钟组在同一 checkpoint 的同一 t 对所有 agent 生成相同头，
而主 GRU 隐状态仍不同，因此不强迫它们选择同一个动作。
原始 t 的幅度会比常见归一化 obs 大；如果时钟组变差，也可能是尺度/优化问题，
不能单凭它失败就证明时间依赖不适用。
时钟携带战斗阶段的间接信息，所以它表现好也不等于模型完全不需要环境信息：
主网络仍看环境，并且 t 与战斗阶段可能相关。

## 更新并提交（不取消旧任务）

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
/home/kyang/.conda/envs/marl_cpu/bin/python -m pip install \
  'imageio>=2.9,<3' 'imageio-ffmpeg>=0.4,<0.7' Pillow &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/ozstar_submit_5m6m_ones_timestep_10m_3seeds.py
```

不设置 SUBMIT=YES 只打印计划。提交前检查地图、配额、声明的配置键、
实际 5m6m/生产模型尺寸、训练/测试输入及梯度、视频编码与 Slurm 预检。
重复调用保留同名运行中/完成任务，不重启历史实验，不把旧对照当新组。
tmux 当前作业筛选会自动纳入启动后的任务，无固定 job ID 列表需要修改。
两个新组已加入 5m6m 单独图和六地图一次性图；无历史时明确显示 0/3。
数据和图片上传仍需在集群运行对应命令，不在本地上传合成测试结果。

```bash
python scripts/smoke_test_5m6m_ones_timestep.py
python scripts/smoke_test_submit_5m6m_ones_timestep.py
```
