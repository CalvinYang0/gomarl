# qmix_qtot_sum_p10 四地图扩展

用户指定将 5m6m 的 `qmix_qtot_sum_p10`（mean3 afa70d）扩展到：

| 场景 | 实际地图 | seeds |
| --- | --- | --- |
| smac_8m9m | 8m_vs_9m | 1/2/3 |
| smac_corridor | corridor | 1/2/3 |
| smac_mmm2 | MMM2 | 1/2/3 |
| smac_3s5z | 3s5z_vs_3s6z | 1/2/3 |

共 12 个全新 job，每个约 10M 环境步（t_max=10,050,000），28 CPU/32G/24h。
最后一张不是 `3s_vs_5z` 或普通 `3s5z`。不重复提交 5m6m、不夹带 baseline、all-ones、
alpha 版、权重相加版或单 mixer one-fill；不取消其他任务或恢复旧 checkpoint。

## 严格复制的模型和训练

从 `ozstar_submit_qtot_and_onefill_15jobs.py` 只选择三份 5m6m `qmix_qtot_sum_p10`
模板；所有 EXTRA_ARGS、learner、优化器、TD、探索、评估及媒体配置逐项保留，
只改地图（维度由环境决定）、内存和实验身份。

```
Qmain = QMIX_main(Qi, ones)
Qsub  = QMIX_secondary(Qi, mask*state + (1-mask))
Qtot  = Qmain + Qsub
```

两个完整独立 QMIX 各有 W1/W2/b1/b2，输入同一套 actor Qi。
主分支 all-ones；次分支使用历史 hard-ST Binary-Concrete 门控，p0=.1、temperature=.5，
关闭填 1、无测试阈值；target/诊断沿用软概率。不是新版确定性 soft one-fill。
一个联合历史 TD 损失更新 actor、两 mixer 和 gate；无额外 TD/KL/MC、alpha、
输出 abs、固定 .1 倍次分支、残差中心化或刷新 curriculum。
原 GRU + 两层 ELU 行动头保持不变，不切换官方 PyMARL2 单线性头。
保留现有 gate 热图、分支 Qtot 图/标量及普通诊断和测试视频；输出占比不人为平衡。

## 提交

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_qtot_sum_four_maps_12jobs.py
```

无 SUBMIT=YES 时只预览；没有实际 job ID 不声称已提交。
批次为 `qtot_sum_p10_four_maps_10m_12jobs_20261026`，任务名为
`{scene}_qmix_qtot_sum_p10_10m_s{1,2,3}_qtotsum20261026`。
保留 5 GiB / 50,000 inode 启动门槛（非全程空间保证），四地图定义/文件及 SC2 二进制检查，
真实模型合成前后向测试、全部缺失 job 的 sbatch --test-only 及逐任务落盘提交回执。
同名活动/已成功作业保留；不清理空间、不绕过锁或门槛。

新测试覆盖四地图实际维度、原 actor/head、直接 Qtot 相加、actor/两分支/gate 更新、
target 同步、诊断 JSON、12 份配置及模拟 Slurm 的幂等/回执；不启动 SC2 或提交实际任务。
本入口不上传 W&B，不扩展旧上传循环；后续汇总须按这个新批次核对三种子。
