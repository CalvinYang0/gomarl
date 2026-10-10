# Obs 的哪些部分影响超网络？

仅针对当前 **无 gate 的 SMAC 单分支 Linear Obs baseline**。
不是 mask 保留概率/注意力热图，也不是对训练信用分配或胜率的因果归因。
不会修改原测试动作、采样数据、模型参数、损失、随机状态、已有参数梯度或 MAC 缓存。
不启动新 rollout，不用训练奖励优化这个诊断。

## 三种图

1. **生成头敏感度 `head_sensitivity`**：每行一个原始 Obs 字段，列是生成头的
   W1、b1、W2、b2。计算每个参数块对该字段的导数 RMS，再乘本次测试样本中该字段的标准差。
   原始、未按标准差缩放的精确 Jacobian L2 值保存在 CSV/JSON。
2. **决策敏感度 `decision_sensitivity`**：每行一个 Obs 字段，每列一个 agent。
   固定该状态原本的主网络 GRU 输出和真实可用动作，计算原 top1/top2 两个 Q 的差值对
   超网络 Obs 的梯度绝对值，再乘字段标准差并对探针状态平均。它是局部连续敏感度，
   对 visibility 等二进制字段也按连续网络延拓计算，不能解释为实际翻转二进制的效果。
3. **字段组干预 `group_ablation`**：只把超网络输入中的一组字段替换为本次测试样本均值。
   主网络的当前/历史 Obs、GRU 输出、环境可用动作 mask 都不变。
   显示贪心动作改变比例与原 top1/top2 Q 差距的平均绝对变化。
   字段组包括自身可移动信息、自身血量、敌军/友军几何信息、血量、visibility/attackability 等；
   图中括号说明组内字段数量，不默认不同大小的组有可比因果贡献。

均值替换可能产生物理上不可能的观测组合，只是离线模型探针，不将修改的 Obs 送进模拟器。
动作改变比例不是胜率下降，也不是删除信息后重新训练的性能。
主网络也读取原 Obs；这里专门把那条路径固定，避免将主网络敏感度混入超网络敏感度。
`ally_0` 表示该观察者的第一个友军槽，未必是绝对 agent 0。

## 为什么 Linear Obs 可以精确计算头的敏感度

代码路径是 `dual_linear_encoder(obs)`，接着四个 `nn.Linear` 生成行动头参数。
无 gate、无非线性 condition 输出编码时，θ(o)=A o+b，因此 dθ/do=A。
这个 Jacobian 在同一个 checkpoint 下不随当前 Obs、agent 或 timestep 改变。
图的变化来自训练后参数 A 改变以及测试样本的字段标准差改变。
生成参数对字段敏感，不意味着该字段一定改变决策；主网络隐藏状态和头里的 ELU 会影响 Q 排序。
参数敏感度也依赖参数化/尺度，不可直接拿不同模型的数值当重要性排名。
常量字段的 std 缩放分数为 0，不等于模型结构上忽略该字段；应结合未缩放值查看。

## 调度、文件与 W&B

更新代码后，未来符合上述条件的 Obs baseline 默认启用：

- 每 1M 环境训练步，使用跨节点后的第一轮正常测试中的前 10 局。
- 从有效且存活、至少有两个可用动作的 agent 状态中，按 agent 分层、确定性等距抽取最多 256 个探针。
- 字段均值和标准差由所记录的十局全部有效 agent 状态计算，不只是探针。
- 所有字段标签来自实际 SMAC capturer 的布局，不手工猜测 5m6m 的维度。
- ID、全 1、timestep、global-state、固定头、attention 和有 gate 的方案明确跳过
  **这一套 hyper-only 诊断**，但仍有公共的 policy 参数/输入敏感度图，见
  [未来 SMAC 可视化默认规则](future_smac_visualizations.md)。

W&B Media 键：

- `test_hyper_obs_importance/head_sensitivity`
- `test_hyper_obs_importance/decision_sensitivity`
- `test_hyper_obs_importance/group_ablation`

状态标量：同一前缀的 `samples`、`episodes`、`failed`。
原始结果保存为 `importance.json`、`features.csv` 和三个 PNG：
`<local_results_path>/hyper_obs_importance/<unique_token>/seed_<seed>_<run-hash>/step_<actual_t_env>/`。
JSON 包含每个探针的 episode/timestep/agent 和逐字段决策分数，便于和测试视频对照。
本地文件并不自动变成 W&B artifact；上述 PNG 通过普通离线 W&B 媒体同步。
图没有按列或行归一化，跨节点比较时还应读取原值并固定色条范围，不能仅凭颜色判断趋势。
正常测试只有不足十局时记录实际局数，不新增测试局。
默认 `test_visualizations_required=True` 时，公共入口写出失败清单后会报错，
避免新任务静默遗漏可视化。显式关闭 mandatory 检查的调试运行才只警告继续。

设置：`test_hyper_obs_importance=False` 可关闭；`test_hyper_obs_importance_interval=100000`
可提高到每 100K 一次以检查峰值之前的变化。这个间隔只控制诊断，不是训练热身。
已运行的进程不会自动加载新代码；旧的 Q/mixer 均值不能补算字段重要性。
需要对应 checkpoint 和真实观测/历史，或在之后的新实验里记录。这里不取消或重启现有作业。

## 验证

`scripts/smoke_test_hyper_obs_importance.py` 不启动 SC2、不连接 W&B。
检查 5m6m/8m9m 真实布局与生产网络下的纯诊断 Q 与原模型 Q 相等、精确 Jacobian、
终止/死亡/padding mask、参数/缓存/梯度/随机状态隔离，并用一个只依赖敌军血量的
人工模型验证字段定位、动作变化和三个 PNG 的生成。
所有示例图都标记 **SYNTHETIC QA - NOT EXPERIMENTAL DATA**，不是已训练模型的结果。
真实 SC2 数据及集群端 W&B 上传仍需新运行验证。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/smoke_test_hyper_obs_importance.py
```

使用现有已安装的 PyTorch、SMAC/PySC2、NumPy、PyYAML、matplotlib；不添加归因工具依赖。
