# Counter 单 Transformer 九模型（2026-09-02）

所有模型使用相同单 Transformer 分支、相同超网络 Q-head 和 QMIX mixer。
不是删除超网络的 `qmix_minimal`。统一 seed=1、目标 10.05M steps、24G、28 CPU、48 小时；
48 小时不保证达到目标步数，每 1M 步保存 checkpoint。保留已有日志，不清理数据。

W&B 名字统一为 `grf_counter_trans9_<label>_10m_s1`。

测试间隔统一为 **10,000 环境步**（2026-09-03 从 50,000 调整），与导入的
`src/config/envs/academy_counterattack_easy.yaml` 一致；仍为每次 32 局、8 个并行环境。
原九模型及追加的 no-relation / random80 对照共用这个设置。
只提高评估频率，不修改测试门控策略、测试局数或训练损失；训练诊断图仍每 50K 步记录。
该修改仅对新启动进程生效，不会自动重启现有任务；更密集的评估也会增加运行耗时。

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

## KL-drop 训练诊断

trans9 的主路径 KL80、KL80 辅助（包括无 relation 对照）、fixed80 辅助现在记录
**learner 从 replay 抽取的训练 batch**，不是测试 probe，也不是行为采集时的 mask。
诊断复用真实前向的张量，不额外采样，不改变损失、梯度、温度或测试策略。

- `train_gate/main_attention_probability/*`：主门控预测保留概率。
- `train_gate/main_attention_mask/*`：主 TD 前向实际应用的 mask（预热时可能全 1）。
- `train_gate/aux_kl80_probability/*`：辅助门控预测概率 p。
- `train_gate/aux_kl80_mask/*`：该辅助前向实际采样的连续 mask。
- `train_gate/aux_kl80_main_mask/*`：同一次辅助前向的主 mask。
- `train_gate/aux_kl80_combined_mask/*`：上面两个 mask 的实际乘积。
- fixed80 对照使用 `aux_fixed80` 前缀，不混称 KL80。

各项包含 mean/std/min/max、低于 .1/.5 和高于 .9 的比例、10 个等宽分布区间的比例、
有效 slot 总数。`bin_0.0_0.1_fraction` 等是 [0,.1) 区间占比，最后一格包含 1。
标量按 learner_log_interval 记录该次完整 batch 的有效 TD 状态，排除 padding 和末尾 bootstrap 状态；
不是两次日志之间所有更新的滑动平均。连续 mask 的 `<.5` 比例不是精确置零比例；
概率 p=.8 也不代表 Concrete 样本均值必须为 .8。

`train_gate_heatmap/<上述名称>` 按 agent 分面，横轴 replay episode timestep，纵轴 slot，
0 白、1 红。概率图和实际连续权重图分别标注。默认每 50K 环境步，在下一次标量日志时，
选该 batch 有效长度最长的一个 episode，最多画前 200 个有效时刻；不是整个训练分布。
辅助预热结束前没有辅助图。以下配置可控制开销：

```yaml
clean_train_gate_diagnostics: True
clean_train_gate_image_interval: 50000
clean_train_gate_image_max_steps: 200
```

已有 `loss_kl80_random_auxiliary`、`weighted_loss_kl80_random_auxiliary` 保留，
`kl80_random_auxiliary_coef` 现在也通过 W&B 精简日志过滤器。
新诊断需新启动的训练进程加载代码；git pull 不会让已运行的 Python 自动更新。
W&B offline 文件仍需正常同步才能在网页看到。没有从旧测试图反推或补造训练数据。

预检：`python scripts/smoke_test_train_gate_diagnostics.py`，检查 padding、实际乘积、
热力图渲染/上传缓冲、图片间隔，以及启用/关闭诊断时优化结果和 Torch RNG 完全相同。

## 追加 KL50 / KL30 辅助先验对照

- `grf_counter_trans9_relation_kl50aux_10m_s1`
- `grf_counter_trans9_relation_kl30aux_10m_s1`

以 `relation_kl80aux` 为对照，仅将辅助门控的 KL 目标先验由 Bernoulli(.8)
改为 Bernoulli(.5) / Bernoulli(.3)，配置键为 `clean_kl_auxiliary_prior`。
数字指**保留概率先验**，不是固定保留率，更不是丢弃率。
仍由 obs 预测 p，用 Binary Concrete 温度 .5 逐 timestep 采样；初始 p=.95、
250K 预热、relation 权重 1、辅助 TD 权重 1、原自适应 KL 系数规则不变。
主门控没有新增 KL；辅助 mask 仍只乘在辅助 TD 路径，行为采集和测试不应用它。
沿用每 10K 步测试 32 局及全部参数图、轨迹图、训练诊断。

新日志分别使用 `loss_kl50_random_auxiliary` / `loss_kl30_random_auxiliary`、
`train_gate/aux_kl50_*` / `train_gate/aux_kl30_*`、
`test_mask_probability_heatmap_auxiliary_kl50_attention` / `...kl30_attention`；
原 KL80 日志和内部 checkpoint 参数名保持兼容。先验本身也记录在 `train_gate/aux_klXX/keep_prior`。

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/dual-branch-benefit-drop
bash scripts/ozstar_submit_counter_kl50_kl30_aux.sh
```

仅追加这两个，不取消现有任务；复用同名同目录的活跃任务，原九模型默认列表不变。
每个新增任务提交前通过合成 learner 预检和 Slurm test-only；逐个记录提交清单。
`DRY_RUN=YES` 只打印这两个配置，不提交、不取消、不写运行日志。

## 追加单分支 linear + relation + KL80 辅助

W&B 名字：`grf_counter_trans9_linear_relation_kl80aux_10m_s1`。
其中 trans9 仅表示实验系列；此模型实际是 linear_only，不执行 Transformer 或双分支融合。
使用现有单 linear 分支定义：遮蔽后的 30 维 obs 经 `nn.Linear(30, relation_dim)` 生成条件，
保留超网络 Q-head 和 QMIX，并非 fixed-Q-head/no-hyper 版本，也不是额外新增多层 MLP。

相对 `relation_kl80aux`，仅切换编码分支；保留 obs 主门控、权重 1 的 param–mask relation、
固定 relation 配对、KL 保留先验 .8、连续随机采样温度 .5、250K 预热、乘法辅助 TD 权重 1、
自适应 KL 系数和测试主门控阈值 .5。辅助 mask 只在辅助 TD 更新中应用，不用于采集/测试。
relation 距离和辅助 KL 只计算 linear 对应的有效门控，避免无效 attention 槽位影响损失。

测试仍每 10K 环境步、32 局，CPU/内存/总训练步数等复用同系列设置。
主训练日志为 `train_gate/main_linear_*`，辅助统计仍为 `train_gate/aux_kl80_*`；
测试热力图使用 `test_mask_probability_heatmap_linear` 和
`test_mask_probability_heatmap_auxiliary_kl80_linear`，保留生成参数 PCA 和 gate trajectory。

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/dual-branch-benefit-drop
bash scripts/ozstar_submit_counter_linear_kl80aux.sh
```

仅追加此一个任务，不停止已有任务；同名活跃任务不会重复提交。
`ozstar_keep_kl_aux_three.py` 已保护此变体，加上之前的旧三项和新四项，共八个变体。
预检 `smoke_test_counter_linear_kl80aux.py` 检查 linear 实际输入等于 obs×主 mask×辅助 mask、
不执行 attention、辅助测试隔离、linear 分支梯度非零而未用分支梯度为零，以及实际更新和绘图。
