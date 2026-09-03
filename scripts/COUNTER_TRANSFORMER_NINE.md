# Counter 单 Transformer 九模型（2026-09-02）

所有模型使用相同单 Transformer 分支、相同超网络 Q-head 和 QMIX mixer。
不是删除超网络的 `qmix_minimal`。统一 seed=1、目标 10.05M steps、24G、28 CPU、48 小时；
48 小时不保证达到目标步数，每 1M 步保存 checkpoint。保留已有日志，不清理数据。

W&B 名字统一为 `grf_counter_trans9_<label>_10m_s1`。

| label | 主路径 | 辅助 TD |
|---|---|---|
| baseline | 无门控，仅 TD | 无 |
| relation | obs 门控 + 1×relation | 无 |
| relation_temporal | obs 门控 + 1×relation + 1×temporal | 无 |
| relation_random50 | 同 relation | 1×TD，原门控乘独立 Bernoulli(keep=.5) |
| relation_temporal_random50 | 同 relation_temporal | 同上 |
| kl80 | obs 门控 + KL(Bernoulli(p)‖Bernoulli(.8)) | 无 |
| relation_kl80aux | 同 relation | 1×TD，原门控乘独立 obs-conditioned KL80 随机门控 |
| relation_temporal_kl80aux | 同 relation_temporal | 同上 |
| kl80_test_open | 训练同 kl80，测试实际 mask 强制全 1 | 无 |

## 精确定义

- relation 使用原始 fixed 配对 `(t-1,t)` 和不重复的 `(floor(t/2),t)`，不是 episode_random。
  比较 sigmoid 概率，不是采样值；只计实际使用的 attention 分支。
- relation 的参数距离目标 detach；relation / temporal 辅助梯度只更新主门控。
  主 TD 和额外随机 mask TD 正常更新主网络。
- 所有启用的 relation、temporal、随机辅助 TD 系数都是固定 1。
- KL80 沿用旧 KL80 的 Bernoulli KL 正则与 Binary Concrete 连续采样（温度 .5），
  不是把每个 slot 固定为 .8，也不是硬 Bernoulli(.8)。KL 系数单独沿用原
  TD/辅助损失 EMA 比例调节（目标 .1）；这里的 .1 是损失量值比例，不是梯度比例。
- 初始概率 .95，250K warmup，warmup 后启用主门控和辅助训练。
- KL80 辅助门控独立于主门控，读取原始 obs，逐 timestep 采样，只用于辅助 TD。
  其 KL 正则只约束自身概率；主 TD、采样交互和测试不应用这个辅助 mask。
- 普通测试沿用 .5 阈值确定性主门控；第 9 个测试全开，但仍记录预测概率。

## 图与日志

每次测试记录第一个测试 episode（最多 1000 步，Counter time_limit=150）。
所有九个模型均生成并写入 W&B 离线日志：

- `test_generated_parameter_pca_trajectory`：真实生成 Q-head 参数的逐时刻 PCA；baseline 也有。
- `test_dynamic_gate_trajectory`：各 slot 的 drop 概率轨迹。
- `test_mask_probability_heatmap_attention`：每个 agent 单独面板，横轴 timestep，纵轴 slot；
  keep 概率 0=白，1=红，固定色标，不跨 agent 平均。
- 两个 KL 辅助变体额外有 `test_mask_probability_heatmap_auxiliary_kl80_attention`，
  标明该辅助门控在测试时不应用。

无门控 baseline 的热力图全红并明确标为 all kept，不声称是学习概率。
第 9 个的热力图画预测概率，标题明确指出实际测试 mask 全开。
W&B 为 offline：生成日志不等于已上传，需运行现有同步脚本后在网页查看。

## 提交

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/dual-branch-benefit-drop
CANCEL_OLD=YES REPO_DIR=/home/kyang/code/gomarl-dual-branch \
bash scripts/ozstar_submit_counter_transformer_nine.sh
```

提交前先运行九模型真实 learner 更新和绘图预检，再对所有新任务做 `sbatch --test-only`。
只有这些通过、提交清单成功写盘后，才取消当前用户且 WorkDir 精确属于本仓库的旧任务。
不会取消其他项目，也不会删除日志。已存在的同名九模型任务保留，重复执行不会重复提交。
中断后的提交结果见 `ozstar_logs/transformer_nine_*.json`。
`DRY_RUN=YES` 只打印配置，无 Slurm 调用、无取消或提交。

预检使用合成 padded episodes，不替代真实 GRF 环境训练、集群内存测试或多 seed 效果验证。

## 追加对照：仅去掉 relation

W&B：`grf_counter_trans9_obs_gate_kl80aux_10m_s1`。
对应 `relation_kl80aux`，仅关闭 relation（系数 1→0），保留同一网络结构、obs 主门控、
独立 KL80 辅助 gate、权重 1 的辅助 TD、原有 KL 正则及其系数规则、预热、测试方式和全部诊断图。
主 gate 仍受 TD 训练，不新增主路径 KL。配置中通用 Bernoulli keep=.5 字段不用于 KL80 辅助；
该辅助由独立门控的概率和 .8 先验决定，与原版相同。

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/dual-branch-benefit-drop
bash scripts/ozstar_submit_counter_kl80aux_no_relation.sh
```

此脚本只追加一个任务，不取消、重启或删除现有任务/日志；相同名称且同一仓库的活跃任务会复用。
原九模型脚本仍默认只选原九个，不要用它来提交此追加对照。

## 追加对照：固定 random80 辅助

W&B：`grf_counter_trans9_relation_random80_10m_s1`。
与 `relation_kl80aux` 比较：保留 obs 主门控、1×relation、1×辅助 TD、乘法位置、
逐 timestep 采样、250K 预热及全部训练/测试设置；仅把额外辅助概率固定为 p=.8，
取消辅助概率学习及辅助 KL 正则。这里 80 指保留概率，不是丢弃概率。

为避免混入硬/软采样差异，使用和 KL80 完全相同的 Binary Concrete 采样（温度 .5），
不是 Bernoulli 0/1 遮蔽。为了匹配其余网络初始化，保留辅助模块的构造顺序，
将其输出层固定为 logit(.8)，冻结整个辅助模块；它的概率不再依赖 obs。
主 gate 仍正常训练。辅助 mask 不应用到行为采集或测试；辅助概率图单独命名为
`test_mask_probability_heatmap_auxiliary_fixed80_attention`，不标为 learned KL80。

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/dual-branch-benefit-drop
bash scripts/ozstar_submit_counter_random80_aux.sh
```

只追加此一个对照，不停止其他任务；原九模型默认提交列表不变。
这个实验比较“可学习且 KL 约束的辅助扰动”与“固定概率辅助扰动”，
不能单独分离 KL 正则与概率可学习性各自的贡献。
