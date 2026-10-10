# 5m_vs_6m：VDN / QMIX 三种子 10M 对照

目的：在同一训练设置下观察 VDN、QMIX 是否出现后期退化，不能用历史 5M 终点代替 10M。
两组各 seed 1/2/3，共六个新作业。历史结果不覆盖、不续接，不取消其他作业。

| 设置 | 两组共用 |
| --- | --- |
| 地图 | 5m_vs_6m，5 Marines 对 6 Marines |
| 预算 | 10M；t_max=10,050,000，跨过最终测试节点 |
| 行动网络 | 共享 GRU-64 + 固定 64→64→12 ELU 头，不是超网络生成头 |
| 主网络输入 | 局部 obs（55）+ 上一步动作（12）+ agent ID（5） |
| 训练 | 原有 clean_hyper：Adam、lr=0.001、Double-Q、TD(λ)=0.6、γ=0.99 |
| 损失 | 有效转移上的主 TD 平方误差，系数 1；无 gate、KL 或辅助损失 |
| 采样 | batch_run=8、batch=128、buffer=5000，沿用原有探索日程 |
| 测试 | greedy，每 10K steps 测 32 局 |
| 可视化 | 每 1M 正常测试的前 10 局视频；每 100K 测试价值诊断 |
| 资源 | 每作业 28 CPU、24G、48 小时 |

两组唯一模型差异：`mixer=vdn`（求和）/ `mixer=qmix`（全局 state 生成正权重）。
沿用本仓库及现有 8m9m 对照的固定两层头，不声称逐项复现官方 PyMARL。
没有新加热身、正则或探索调整；历史 5M 组保持单独标签。

运行名：`smac_5m6m_{vdn,qmix}_baseline_10m_s{1,2,3}_valuediag`。
重复调用保留同名 active/completed 作业，不重复提交；提交清单记录实际 job ID 和代码版本。

## 提交六个新作业

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_5m6m_vdn_qmix_10m_3seeds.py
```

省略 `SUBMIT=YES` 只打印计划，不提交。提交前核验地图、参数、固定头及两种 mixer 的真实
维度和反向传播、视频编码路径，再对全部缺失作业执行 Slurm test-only。
视频依赖缺失时先按 `docs/test_battle_videos.md` 安装，不静默关闭视频。
只有服务器返回六个 job ID 才代表已提交；本地检查不启动 SC2，不上传合成结果。

## 绘图

新两组已接入 `scripts/plot_5m6m_head_condition_3seeds.py` 和六地图入口；历史 5M 保留为
独立参考。新组没有数据时明确显示 0/3，不使用历史结果补位。
观察 `test_battle_won_mean`，同时看原始数据和逐种子曲线，比较峰值窗口与末期窗口，
不要只比较单次最佳 checkpoint。默认 centered window=100 可用 `--mean-window 1` 关闭。
现有 tmux 按 RUNNING 作业发现 W&B 同步，无需新增名称白名单。
