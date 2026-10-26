# Corridor：补齐单 mixer all-ones 三种子

只跑 `corridor` 的 `qmix_state_ones`，seeds 1/2/3，共三个全新 job。
严格从已补跑普通 QMIX 的 `ozstar_submit_corridor_qmix_3jobs.py` 复制配置，
除实验名称/分组外，只将 `qmix_condition_input=state` 改为 `ones`。

相同 CleanLearner、单个 QMixer、参数量和初始化、actor 实时局部观测、GRU + 两层 ELU
行动头、TD/优化器/采集/评估及标量诊断；不切换官方 PyMARL2 行动头。
online 与 target 的 W1/W2/b1/b2 四条路径全读 `ones_like(state)`，
不是权重固定为 1，不是 VDN，也不是行动头超网络输入 ones。
无 gate、one-fill、随机 mask、额外损失、第二 mixer 或刷新 curriculum。

总预算 t_max=10,050,000（名义 10M），每 job 28 CPU/24G/24h。
沿用普通 Corridor baseline 的标量诊断和媒体设置，不额外开启录像或 checkpoint。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_corridor_ones_3jobs.py
```

任务名：`smac_corridor_qmix_state_ones_10m_s{1,2,3}_ones20261027`。
批次：`corridor_mixer_ones_10m_3jobs_20261027`。
无 SUBMIT=YES 只预览。保留 5 GiB / 50,000 inode 门槛、Corridor 定义/地图/二进制检查，
真实尺寸模型合成 TD 更新、全部缺失任务 Slurm 预检、逐 job 落盘回执和重复提交保护。
不重复提交普通 QMIX/one-fill，不取消其他任务，不清理数据或降低配额门槛。
必须得到实际 job ID 才表示已提交；本入口不自动补传数据或扩展旧 tmux 范围。

测试验证 baseline 与 ones 的参数/RNG 初始化、行动头和 learner 一致，只有
条件开关变化；训练及 target 四条 W/b 路径均读全 1，actor 仍用真实观测。
合成及模拟调度测试不代表已经跑过真实 SC2 训练。
