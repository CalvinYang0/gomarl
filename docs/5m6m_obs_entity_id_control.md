# 5m6m：Obs + 固定实体 ID

新增 `linear_obs_entity_id_baseline`，只用于 SMAC 的无 gate 单分支 Linear Obs 路径。
不是把 Obs 替换成 ID，也不是给主网络增加 ID。主网络本来就收到原始局部 Obs、
上一动作和观察者 agent-ID，维持不变；mixer 仍使用原来的全局 state。

## 实体身份与实际输入

5m_vs_6m 使用 11 维固定 one-hot 词表：友军（包括自己）0..4，敌军 5..10。
敌军 ID 对应原来的敌军槽/攻击动作目标；友军槽使用 SMAC 的绝对 agent 编号，
按递增编号排列并排除观察者。不是给每个观察者的 ally_0 都写 ID 0。
例如 agent 2 的友军槽对应 ID [0,1,3,4]，自己对应 ID 2。
这里的身份是环境固定 roster 编号，不是 SC2 随运行变化的 unit tag。

超网络输入实际顺序：

```
self movement 原字段
enemy_0 原字段, enemy_0 one-hot ID
...
enemy_5 原字段, enemy_5 one-hot ID
ally_0 原字段, ally_0 absolute one-hot ID
...
ally_3 原字段, ally_3 absolute one-hot ID
self health 原字段, self one-hot ID
```

55 维原始 Obs 完整保留，加 11 个实体各 11 维 ID，总计 176 维。
只扩大原有一个 `dual_linear_encoder` 的输入，不增加非线性、attention 或第二个头。
保留原始字段、主 GRU、生成器等模块的同种子初始化；新增 ID 列独立初始化。
零掉 ID 输入列的权重后，同种子模型的 Q 必须回到原始 Obs baseline，预检对此做数值验证。

ID 在所有 timestep、在线/target 网络、训练/测试、reset 后都保持同一映射。
不可见/死亡的实体仍保留其固定 ID，其原始状态字段保持环境给出的值；
不填充隐藏血量/位置、不引用其它 agent 的 Obs、不额外使用全局 state。

## 损失与实验控制

只用原 TD loss，无 gate、KL、额外 TD、关系或稳定性辅助项。
匹配现有 Linear Obs 的 optimizer、Double-Q、TD-lambda、replay 和测试设置。
三种子 1/2/3，10.05M 停止阈值（10M 图），每 10K 测试 32 局，
28 CPU/24 GiB/48h；已有 value diagnostics 每 100K，战斗视频每 1M 十局。
旧的 raw-Obs-only 字段重要性诊断明确跳过这一新模型，避免把 ID 列误标为 Obs 字段。

注意：输入编码器增加 121×64=7744 个参数，不能称为完全等参数对照。
此外整个生成器仍是仿射映射，Obs 与固定 ID 在 condition 中是加性结合，
没有新增实体交互或置换不变机制；加 ID 不保证模型学会战术语义。
所有 ID 向量都是观察者 agent 编号的固定函数，敌军 ID 部分更是所有观察者共享的常量。
因此在这个纯线性编码器下，可写成 condition(o,i)=A o+b_i；新增 ID 提供的是
固定身份偏置，并没有显式建立“某个属性属于某个 ID”的乘性交互。
这是保留原架构的输入消融，不是新的 entity-aware/attention 模型。
若它弱于 ID-only，只能支持“在该实现/训练设置下，加入 Obs 未带来收益或造成负面影响”，
不能单独排除优化、初始化、参数量等因素并证明“拟合噪声”。
ID 比较使用 `smac_5m6m_linear_id_baseline_10m_s{seed}_valuediag` 的 10M 版本，
不再选旧 5M idkl80fix。KL80 仍是 5M；与 KL80 比较时只比较共同训练区间，
不能把不同训练长度的最终值作为唯一结论。

## 组数、命令与绘图

输入对照六组：Obs、ID-only、global state、全 1、episode timestep、Obs+实体 ID。
加 direct KL80 正则对照，共七组；不重新提交已有 Obs 或其它已运行作业。

新运行名称：`smac_5m6m_linear_obs_entity_id_10m_s{1,2,3}_entityidcond`。
提交器保留相同名字的 active/completed jobs，不取消任何任务；
真实地图、配置、网络预检和所有缺失作业的调度预检通过后才提交。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/ozstar_submit_5m6m_obs_entity_id_10m_3seeds.py
```

不带 `SUBMIT=YES` 只打印计划，不提交。
当前会话自动 SSH 认证失败：本地验证/推送不代表已在 OzSTAR 启动作业。

现有绘图入口都已加入新组，Obs 使用新的 recheck 可视化记录，ID 使用 10M 版本，
不混用历史 attention-ID 或旧 5M 线性 ID：

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/plot_5m6m_head_condition_3seeds.py
# 或一次性更新六张地图；5m6m 七组，其他地图的组保持原样
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/plot_smac_six_maps_3seeds.py
```

新组无数据时显示 awaiting，不伪造曲线。原始曲线、均值/样本标准差、
seed inventory、每组实际训练终点都保留。

模拟器外验证：`scripts/smoke_test_5m6m_obs_entity_id.py` 检查真实 5m6m/8m9m 维度、
ID 对齐、固定性、dtype、主网络隔离、同初始化、梯度、target、实际 TD 更新及 value diagnostics。
这里只提交 5m6m 三种子，不自动增加 8m9m 作业。
