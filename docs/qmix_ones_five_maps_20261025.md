# QMIX 对比 all-ones：GRF 三地图 + MMM2 + 3s5z_vs_3s6z

## 最新范围调整：先不跑 GRF

用户随后决定 GRF 暂不跑。CLI 默认改为 `--domain smac`，仅提交 MMM2 和
3s5z_vs_3s6z：两模型 × 三种子 × 两地图 = 12 jobs，不进行 GRF reset/step 预检。
模型、10M 步、24h 和配额门槛保持不变；不取消现有任务，也不自动清理数据。
专用批次为 `qmix_vs_ones_mmm2_3s5z_10m_12jobs_20261025`。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_qmix_ones_five_maps_30jobs.py --domain smac
```

文件名保留兼容性，不代表当前会提交 30 个。下方为原五图方案存档；
只有显式传入 `--domain all` 才加入三张 GRF 地图。代码的 `build_plans(repo)` 默认全图供历史测试使用，
CLI 会显式传入实际选择，不把全图默认函数直接交给提交器。

## 范围与口径

用户要求在三个既有 GRF 场景、MMM2 和 3s5z_vs_3s6z 上比较普通 QMIX 与 all-ones。
沿用历史两层 ELU 行动头，不切换为官方 PyMARL2 的单线性行动头。
默认每组 seeds 1/2/3、10M 环境步（t_max=10,050,000），五张地图 × 两模型 × 三种子 = 30 个全新 job。

| 地图 | 场景键 | 模型/种子 |
| --- | --- | --- |
| academy_counterattack_easy | grf_counter | QMIX / all-ones，各 1/2/3 |
| academy_pass_and_shoot_with_keeper | grf_pass | 同上 |
| academy_3_vs_1_with_keeper | grf_3v1 | 同上 |
| MMM2 | smac_mmm2 | 同上 |
| 3s5z_vs_3s6z | smac_3s5z | 同上 |

两组共同使用 `clean_learner`、`qmix_minimal` 和同一个 QMixer：
普通组 `qmix_condition_input=state`；all-ones 组为 `ones`。
该开关在 QMixer forward 中、进入四条 state 路径之前替换输入，
因此 W1、W2、b1、b2/value 全部读取同维度全 1；不把权重设为 1，仍是非线性单调 mixer，不是 VDN。
没有门控、mask 特征、分组、one-fill、辅助损失或刷新频率 curriculum。

actor 的实时局部 obs、上一动作、agent ID、GRU 和两层行动头都不变。
训练与 target 共用同一输入规则；参数数目、参数初始化、优化器和 TD 逻辑完全相同。
旧 `MixerInputLearner` 的 ones forward 与新开关的代数相同，但本轮不沿用其独立训练器，
避免两组同时比较不同的 learner/有效 mask 实现。不能声称本轮与旧 ones job 逐位复现。
默认 `state` 保留旧 QMixer 的行为、参数键和随机数消耗，不覆盖历史实验。

## 训练与记录

同一地图内只改条件输入开关及 run 身份。
Adam lr=0.001，gamma=0.99，TD(lambda)=0.6，Double-Q；
8 个并行采集环境，batch=128，buffer=5000；每 10k 环境步做 32 局 greedy 测试。
GRF 的既有局部/全局特征、奖励整形、150 步截断和评估规则保持原样；
SMAC 的奖励、状态字段与时间限制也不改。
GRF 使用 `test_game_win_mean`，SMAC 使用 `test_battle_won_mean`，不能混用作图。

每 job 28 CPU、32 GiB 内存、24 小时时限，CPU 训练。
保留 value diagnostics 每 100k 步；SMAC 保留每 1M 步 10 局测试视频。
GRF 沿用不录制视频、不生成 full/goal dumps 的节省空间设置。
不写模型 checkpoint，保留 Sacred、W&B 离线标量和适用的诊断/媒体；
如需中断恢复，应另外讨论，不把没有 checkpoint 的任务称为可续训。

## 提交

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_qmix_ones_five_maps_30jobs.py --domain all
```

不设 SUBMIT=YES 时只预览计划。提交前检查个人 /home 至少 5 GiB + 50,000 空余 inode，
这是启动门槛，不是三十个完整任务的容量保证；不会自动删除、迁移或降低门槛。
检查 SMAC 地图定义、地图文件和 SC2 binary；逐个子进程对 GRF 三地图做短 reset/step 检查
（不训练、不录视频、不留 dump，单场景超时 60 秒）。这可提前暴露 gfootball 导入、地图、维度或运行库问题。
随后通过模型合成测试及全部缺失 job 的 sbatch --test-only 才开始提交。
不取消旧 job；同名已在跑/已完成的 job 保留，回执逐次落盘，防止重复提交。

批次：`qmix_vs_ones_grf3_mmm2_3s5z_10m_30jobs_20261025`，run 后缀 `_stateones20261025`。
当前仅提供训练/预检入口，不声称已提交或已配置新批次长期 tmux 上传。
后续补传必须按此回执精确选择，不与历史同 label 数据混合。

## 验证

`scripts/smoke_test_qmix_ones_five_maps.py` 不启动真实模拟器：
核对 30 份计划、五图生产输入尺寸、两层行动头、配对初始化、实时 actor、
all-ones 的四个 W/b 路径与 target、旧 ones 前向代数、Adam/RMSprop 反向更新、
默认路径不变、配额门槛、Slurm 全部预检后再提交及重复调用保留。
合成测试不等于真实 GRF/SC2 学习结果；真实 GRF reset/step 预检只在服务器提交前执行。
