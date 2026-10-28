# Soft one-fill 与 refresh 2000k：三个地图，18 个 job

地图为 `3s5z_vs_3s6z`、`MMM2`、`corridor`。每个地图两个模型，各 seeds 1/2/3：

1. `qmix_soft_onefill_p10`：旧单 mixer soft one-fill 直接转移地图。
2. `qmix_soft_onefill_p10_refresh_2000k`：同一模型加 mixer 原始 state 的 refresh curriculum。

两组均保留历史 `qmix_minimal` GRU + 两层 ELU 行动头，单完整 QMIX、静态逐字段
sigmoid 系数 p0=0.1，输入 `p*state+(1-p)`；同一 joint TD、优化器、target 更新、
采集/评估和原有媒体设置。无随机硬门控/ST/温度/测试阈值/KL/MC/额外评分或第二 mixer。
不是 qsum，也没有换成官方行动头或普通 CleanLearner 的 TD。

组合版先缓存真实 raw state，再由 soft gate 插值；不缓存门控后的输入，也不冻结 gate
或 mixer 权重。actor obs 每步更新。门控按当前在线/target 参数分别计算，规则相同。

沿用最大 episode 长度 L 分段。K=min(L,1+floor(采集时全局环境步数/2,000,000))：

| 采集步数 | 每局分段数 | 刷新位置 |
|---|---|---|
| 0–2M | 1 | 仅开局 |
| 2–4M | 2 | 开局、L/2 |
| 4–6M | 3 | 开局、L/3、2L/3 |
| 6–8M | 4 | 每 L/4 段起点 |
| 8–10M | 5 | 每 L/5 段起点 |

位置向下取整；短 episode 未达到后续边界就不会刷新，t=L 不额外刷新。
K 每局/并行 rollout reset 时固定。名义10M，实际 t_max=10,050,000，因此越过10M后
进入 K=6；不强制恢复到每 timestep。Replay 保存采集时 held state、raw/source/阶段元数据，
在线与 target 都按已存状态训练，不按当前训练阶段重新解释旧 replay。

`MixerStateRefreshAuditMixin` 只复用校验、诊断、checkpoint 合同；组合 learner
继承 `SoftOneFillGateLearner`，真实 TD 更新与把同一个 held batch 交给旧 soft learner
逐参数验证一致。原普通刷新 learner 的旧合同和行为保留，使用回归测试验证。

每 job 28 CPU / 32G / 24h；预算和评估沿用原 soft。脚本要求 /home 至少5GiB且50,000个
可用 inode。只是启动门槛，不保证18个带原有媒体的任务全程容量；不会自动清理/降低门槛。
先做地图/二进制、合成模型训练和全部缺失 job 的 Slurm test-only，再实际提交。
已有同名运行/成功 job 保留，逐次记录回执，不取消旧任务或拼接旧 checkpoint。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_soft_onefill_refresh_three_maps_18jobs.py
```

批次 `soft_onefill_refresh_three_maps_10m_18jobs_20261028`，任务名以
`softrefresh20261028` 结尾。去掉 SUBMIT=YES 只预览，不查询调度器。
收到18个实际 job ID 才表示全部提交成功；本地测试不是实际训练。
本入口不改变旧 tmux 范围、不上传历史结果。
