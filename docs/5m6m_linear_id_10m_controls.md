# 5m6m 线性 ID 三种子 10M 补提交

入口：`scripts/ozstar_submit_5m6m_linear_id_10m_3seeds.py`。
只包含 `5m_vs_6m`，不会同时启动 8m9m 或取消其他任务。

三个精确名称为 `smac_5m6m_linear_id_baseline_10m_s{1,2,3}_valuediag`。
预算 10M（停止阈值 10,050,000），24G、48h。保持 Obs 基线的 GRU 主网络、
主网络局部 obs/上一步动作/agent ID 输入、生成头、mixer、TD/Double-Q、优化器与采样设置。
唯一模型变化是超网络条件改为线性编码的 agent ID：同一 agent 的条件不随 obs 或局内时间改变，
不同 agent 条件不同。没有 attention 策略替代、gate、KL 或额外辅助损失。

显式开启每 1M 的 10 局正常测试战斗视频和每 100K 的价值诊断。
已运行任务不追溯补录过去的视频，也不会因为视频配置而重启。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_5m6m_linear_id_10m_3seeds.py
```

不带 `SUBMIT=YES` 仅预览。提交前检查地图文件、剩余配额、真实地图尺寸与生产模型尺寸的
前向/梯度/learner 更新、视频及 Slurm 预检。所有缺项完成预检后才开始实际提交。
同名 active/completed 作业保留；缺项才提交。与旧两地图 ID 入口共享锁，防止并发重复。
5M ID 和旧 attention-ID 不视为这三个任务已经完成。

日志目录逐项保存 manifest；只有输出的 `Submitted` job ID 或 Slurm 查询结果能证明任务已提交。
本地测试、代码推送、配置文件或图中 0/3 都不能证明任务已启动或从未启动。
六地图图及独立 5m6m 图已注册这三个名称，无数据时明确显示 0/3，不回退至其他架构或预算。
