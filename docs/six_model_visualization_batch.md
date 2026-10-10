# 六模型可视化补提交

只补本轮六组、三种子、10M，共 18 个准确名称的任务：

| 地图 | 模型 | 数量 |
| --- | --- | --- |
| 5m_vs_6m | 原始 Linear Obs 新跑、全 1、episode timestep、Obs+实体 ID | 12 |
| 8m_vs_9m | 固定头 VDN、固定头 QMIX | 6 |

已知实体 ID 任务 18342166/18342167/18342168 为正确的排队任务，应保留。
所有同名 active/completed 作业都会保留；其他缺项经过模型、地图、视频和
Slurm test-only 检查后才提交。不取消任何任务，不复用旧 scalar-only Obs 名称。
补提交时同时持有四个原分组的锁，防止与单独提交入口并发重复。
训练框架、输入实验、损失和预算不变；原始 Obs 新跑显式打开观测重要性图，
所有 18 个任务显式打开每 1M 十局正常测试视频和每 100K 价值诊断。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/ozstar_submit_six_model_visualization_18jobs.py
```

不带 SUBMIT=YES 只预览，不连接调度器。结果输出 Submitted/Retained 和逐项保存
的 manifest；本地预检和推送不是集群提交成功，须以实际 job ID 为准。

原始 Obs 选择 smac_5m6m_linear_singlehead_baseline_10m_s{1,2,3}_recheck。
5m6m 汇总图改为新 Obs，并选择真正的 10M Linear-ID 组
smac_5m6m_linear_id_baseline_10m_s{1,2,3}_valuediag，而非旧 5M idkl80fix。
六地图图的 5m6m 面板同步使用这些选择；没有新组历史时显示缺失，不用旧数据代替。
ID-only、global state、KL80 不在这次 18-job 入口的补提交范围；KL80 仍按实际 5M 标注。
注意：存在 ID 10M 配置并不证明它已经提交。此前“保留已有任务”的表述不能充当 job ID 证据。
现在可用 `scripts/ozstar_submit_5m6m_linear_id_10m_3seeds.py` 单独补齐 5m6m 线性 ID 三种子 10M；
只有同名 active/completed 作业才保留，其余通过预检后提交，实际成功仍须以返回的 job ID 为准。
