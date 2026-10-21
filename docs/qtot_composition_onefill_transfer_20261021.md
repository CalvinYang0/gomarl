# Q_tot 输出组合与 onefill_p10 跨地图，15 作业

本批严格保留历史 `qmix_minimal` agent：局部 obs + 上一动作 + agent ID，
编码层 + GRU-64 + `Linear(64,64) → ELU → Linear(64,n_actions)`。
**不切换到刚新增的官方 PyMARL2 单层行动头基线。**

## 实验清单

| 地图 | 模型 | seed | 时限 |
| --- | --- | --- | --- |
| 5m_vs_6m | qmix_qtot_sum_bothones | 1/2/3 | 24h |
| 5m_vs_6m | qmix_qtot_sum_p10 | 1/2/3 | 24h |
| 5m_vs_6m | qmix_qtot_alpha10_p10 | 1/2/3 | 24h |
| MMM2 | 原单分支 qmix_onefill_p10 | 1/2/3 | 24h |
| corridor | 原单分支 qmix_onefill_p10 | 1/2/3 | 24h |

三种子、10M（t_max=10,050,000）；每作业 28 CPU，5m6m 24G、MMM2/corridor
32G。24h 是 Slurm 上限，不保证更长地图必定在一天内完成，不自动取消其他任务。

## 组合层次

同一套 agent 的执行动作 utility 向量 q，输入两个**完整且独立**的 QMIX：

```
Qmain = QMIX_main(q, ones)
Qsub  = QMIX_secondary(q, x)
sum:       Qtot = Qmain + Qsub
alpha_sum: Qtot = alpha*Qmain + (1-alpha)*Qsub
```

`bothones` 的 x 为全 1，无生效 gate；其余 x=m*state+(1-m)，初始 keep probability
0.1，训练使用历史 hard-ST Binary-Concrete，target/诊断使用软概率，无测试阈值。
alpha 是两输出共用的单个可学习全局标量，严格位于 (0,1)，初始值 0.1：
**初始主 Q 占 0.1、次 Q 占 0.9**。

每个 QMIX 都独立生成自己的 W1/W2/b1/b2。各自内部权重取 abs 保持单调；
**最终 Qtot 不取 abs、不中心化、不乘固定 0.1、不相加权重。**
一个联合 Qtot 计算一个历史 TD 平方误差，联合训练共享 actor、两 mixer、gate
和 alpha（如有）。没有单独的分支 TD、KL、MC 或额外梯度截断。

主分支和 gate 的初始值沿用普通 one-fill；第二个完整 QMIX 独立按现有 QMixer
默认方式初始化。没有沿用旧权重残差的零输出头：独立 QMIX 的 abs 零权重会导致
次分支权重梯度为零。这个初始化差异已明确记录，不能将新旧结果归因于唯一一个
纯代数因素。两个分支不共享参数，不以相同参数初始化来维持人为对称。

MMM2/corridor 直接复制历史单 mixer onefill_p10 的 learner、头结构、门控和损失，
只改地图、动作输出维度（由环境决定）、作业资源与 run 身份。不改成双分支。

## 诊断与验证

保留普通 gate 概率图、常规测试和媒体配置。新增两分支原始 Qtot、加权贡献与
合成 Qtot 的标量及图，alpha 单独记录。训练图来自最后一次真实随机门控前向；
测试值诊断使用软概率；二者不是同一个量，不声称 soft diagnostic 是训练期望。
没有人为捏造“合并后的 W1/W2”（两个非线性 mixer 的输出和一般没有这种权重表示）。

```
python scripts/smoke_test_qtot_composition.py
python scripts/smoke_test_submit_qtot_map_suite.py
```

测试不启动 SC2、不上传云端：验证三种输出组合公式、负 Qtot、单调性、两分支
W1/W2/b1/b2 梯度、actor 参数不变、联合更新、target 同步、门控/alpha 与 checkpoint，
以及 MMM2/corridor 实际维度下原 one-fill 的一次 TD 更新。
旧 one-fill/权重组合有回归测试，保留原默认行为和历史 checkpoint。

## 提交

在服务器已登录终端执行：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_qtot_and_onefill_15jobs.py
```

不设置 SUBMIT=YES 只显示计划。正式提交复用现有配额/地图/网络预检、全部
`sbatch --test-only` 通过后才提交、同名 active/completed 保留和 JSON 回执流程。
作业名后缀为 `_qtot20261021`，不会覆盖/合并此前权重组合或单分支历史结果。
SSH 认证失败不等于提交成功，必须以真实 job ID 和回执为准。
