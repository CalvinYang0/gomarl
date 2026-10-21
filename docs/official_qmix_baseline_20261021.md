# 修正后的官方 PyMARL2 QMIX 基线

## 范围与来源

采用 `hijkzzz/pymarl2` 固定提交
`8ccac7c5aa134422a2e3009be735d23cdf8ce2f8` 的 QMIX 模型和学习器。
这不是原始 PyMARL 的 RMSProp/一步 TD 配方，也不能宣称是 GoMARL
论文未公开的两套 baseline 作业配置的逐项复现。

直接引入官方 `n_rnn_agent.py`、`nmix.py`、`nq_learner.py`、controller、
动作选择器和相关数值工具。源码位于 `src/third_party/pymarl2_qmix/`，
包含上游 Apache-2.0 许可证、固定 URL、原始 SHA-256、接入后的 SHA-256
及算法体的跨 Python 版本 AST 校验值。只有 import 路径适配和删除两个
本次不支持的 Qatten/matrix-game import；所有算法类和函数体保留。
注册包装器只做参数检查，不重写官方 `train()`。

## 新基线设置

- 5m_vs_6m、8m_vs_9m，各 seed 1/2/3，共六个新作业。
- Actor 为 `Linear → ReLU → GRU-64 → Linear(actions)`；无额外条件编码器、
  两层行动头、超网络行动头、gate 或辅助损失。
- Mixer 使用官方 state 条件化的 QMIX，隐藏层 32、超网络隐藏层 64、abs 正权重。
- Adam lr=0.001，TD(lambda)=0.6，gamma=0.99，官方 `0.5 * mean(TD_error^2)`。
- 8 个并行环境，batch=128、buffer=5000，每次采集一次更新，target 每 200 episodes。
- epsilon 1→0.05，100K 环境步衰减；每 10K 环境步贪心测试 32 局。
- 10M（t_max=10,050,000），每作业 28 CPU、24 GiB、48 小时。
- 记录原始训练/测试标量到 Sacred 与离线 W&B；本批关闭自定义重要性探测、
  gate 图和视频，避免将研究诊断逻辑接入严格基线。

实验名：

```
smac_5m6m_qmix_official_pymarl2_10m_s{1,2,3}_corrected20261021
smac_8m9m_qmix_official_pymarl2_10m_s{1,2,3}_corrected20261021
```

旧 `clean_hyper`、旧 QMIX/VDN/one-fill/双分支实现与历史 run 名不修改、不覆盖。
本次只重跑 QMIX，不提交 VDN。

训练外壳仍复用当前项目的 Sacred/W&B、ReplayBuffer 和容错 SC2 runner。
作业启动日志记录实际 checkout commit、依赖版本、SC2 选择版本、地图哈希及
关键项目文件哈希。不能把官方模型代码等同于整个环境/运行框架逐字一致。

## 验证

```
python scripts/smoke_test_official_qmix.py
python scripts/smoke_test_official_qmix.py --sources-only --verify-upstream
python scripts/smoke_test_submit_official_qmix.py
```

测试不启动 SC2、不上传 W&B、不提交 Slurm：

- 校验固定官方源码与算法体；比对官方 yaml 的训练超参数。
- 两地图真实输入维度、单层输出头、无多余 agent 参数。
- 相同 seed 下通过注册器构建和官方 agent 直接构建的初始化/RNG 状态相同。
- 独立计算含 padding/终止掩码的官方半平方 TD 损失及一次裁剪后 Adam 更新，
  对比实际 learner 参数逐张量一致。
- target 更新与完整项目训练循环/ReplayBuffer/普通评估接线测试（合成数据）。

这些检查证明接入一致性，不代表已经完成真实 SC2 训练或已上传结果。

## 提交

在 OzSTAR 登录节点运行：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_official_qmix_6jobs.py
```

不设 `SUBMIT=YES` 只显示计划，不写运行目录、不提交作业。
正式提交检查个人 /home 配额（至少 2 GiB 空余）、两地图、源码和网络更新测试，
然后先对所有缺失作业执行 `sbatch --test-only`，全部通过才提交。
同名运行/完成作业保留，不取消任何作业；失败/中断作业可重新提交。
每次实际提交写 JSON 回执，记录项目 commit、上游 commit、配置哈希、计划与 job ID。

未取得 job ID 就不能说作业已经提交。SSH 认证失败时需用户在已登录终端执行。
