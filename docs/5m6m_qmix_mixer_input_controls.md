# 5m6m：普通 agent，mixer state 门控／双指标分组

三个模型，每组 seed 1/2/3、10M，共九个独立新作业。普通 agent 复用本仓库
VDN/QMIX baseline 的 `qmix_minimal`（GRU＋固定两层 ELU head），不是生成 head。
主网络仍看完整局部 Obs、上一步动作和 ID。QMIX 原生 state-conditioned mixer
超网络保留；所有 W1/W2/b1/V 路径均只读受控 state。

| label | 机制 |
| --- | --- |
| qmix_state_static10 | 静态 Binary-Concrete 门控，p0=0.1，无 KL，评估 p>0.1 |
| qmix_state_group_qchange | 正向互补 state 字段分组，按 mean abs ΔQtot 排名，双 TD |
| qmix_state_group_tderror | 同样分组，按 max(0, mean(扰动平方 TD 误差−原误差)) 排名，双 TD |

5m6m 标准 state 有 98 项：5×4 ally 字段、6×3 enemy 字段、5×12 上次动作。
两组都读取同一套 agent utilities；只分 state 字段，不分 agent，也没有第二行动头。
分组初始重要组为空，每 1000 次梯度更新，当前所属 mixer 为其字段评分；从
不重要组转入最高正分项，重要组最低分低于最高的一半时转出（重要组至少留一项）。
EMA=0.8，转移时重置该字段的 EMA 初始化标记，不搬网络参数或重置优化器。
在线／target membership 立即同步，参数仍按原 target_update_interval 更新。

两个 TD target 分别由自己的 target mixer 计算；下一动作都由唯一在线 agent
Double-Q 选择，target agent 评价。总 loss=(L重要+L不重要)/2，不是对两个 Qtot
求平均后只算一次 TD。无 KL、L0、蒸馏、附加正则。没有测试“选择分支”问题：
执行不使用 mixer；默认价值诊断评估重要 mixer，并额外记录两组校准和敏感度。

排名固定当前 agent Q、当前 mask 和原来的 own-branch TD target，在训练 replay
有效非 padding／非 terminal 后探针中逐字段均值替换；两个分组模型同时记录两种
分数，只用于排序的指标不同。均值扰动可能破坏 state 字段关联，分数不是胜率归因。
门控训练采样连续 Concrete mask（temperature=.5）；target 用滞后概率的软 mask，
评估／重要性诊断用严格硬阈值。这一点沿用前一轮静态门控训练／测试约定。

每 100k 步保存字段语义热图及两种重要性 JSON；迁移时额外保存迁移记录／热图。
每 1M 步前十局正常测试战斗视频及普通策略参数／Obs 重要性。没有生成头重要性。
默认保留三种子曲线、两组损失、Qtot/Qi、校准误差及敏感度。既有实验不修改／停止。

## 提交

```bash
cd /home/kyang/code/gomarl-dual-branch &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_5m6m_mixer_input_controls_9jobs.py
```

省略 SUBMIT 只预览。最低 3 GiB 配额，不保证所有未来媒体空间；精确同名活动或
成功任务跳过，失败任务可重试。所有模型／配置／地图检查与 sbatch --test-only
通过后才提交，日志 manifest 记录每个 job ID。

## 专用三种子图

```bash
cd /home/kyang/code/gomarl-dual-branch &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/plot_5m6m_mixer_input_controls_3seeds.py --local-only --mean-window 10
```

仅这九个新实验和三种子 QMIX baseline；新 QMIX 完全无数据才整组显示历史
5M，不补单个 seed、不拼接预算。加 --no-upload 仅本地出图。删掉 --local-only
仅精确查询这十五个名称，查询失败不扫描全项目。原 hyper-head 实验不混入新组。
