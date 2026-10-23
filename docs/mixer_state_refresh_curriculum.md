# Mixer state 刷新频率消融（2026-10-23）

## 本轮设置

先跑 5m_vs_6m。阶段长度 S 分别为 500,000、1,000,000、2,000,000 个训练环境步，
各 seeds 1/2/3，共 9 个全新训练 job，预算沿用 10M（t_max=10,050,000）。
资源申请沿用历史 CPU/内存配置，只把 wall time 从两天改为一天。
本轮不额外提交 live-state baseline；需要重新配对的对照时应单独确认。

模型沿用历史 `qmix_minimal`：原 GRU + 两层 ELU 行动头，普通 QMixer。
不改 actor obs、上一动作、agent ID、奖励、动作空间、优化器、探索、TD loss，
不引入 gate、one-fill、辅助损失或修正后的官方行动头。

## 按最大 episode 长度分段

令 L 为环境报告的 episode_limit，T 为本次 episode/并行采集批次开始时的全局训练环境步：

`K = min(L, 1 + floor(T / S))`

更新时刻为 `floor(j * L / K)`，其中 j=0,...,K-1。
第一阶段 K=1，整个 episode 使用初始全局 state；第二阶段 K=2，初始和中点各读取一次；
第三阶段 K=3，初始及两个三等分点读取。非整除时向下取整。
例如 L=120：K=2 对应 t=0/60，K=3 对应 t=0/40/80。
提前结束不把更新点压缩到实际长度；t=L 的最终 next-state 保持最后一段缓存，不额外刷新。

每次 episode reset 固定 K，episode 中途不因跨训练阶段而改变。
并行采集使用该批次 reset 前的总环境步；多 worker 的环境步照历史逻辑合计。
评估使用当前训练 T 对应的 K，不增加训练步数。没有强制最终切换到逐 timestep 更新。
在 T 恰好为 10M 时，三个设置的 K 分别为 21、11、6（实际采集以 reset 时刻为准）。

## 回放与 target 的一致性

本实验将 EpisodeBatch 的 `state` 存为 mixer 实际用到的 sample-and-hold state；
同时保留 `mixer_state_raw`、来源时间 `mixer_state_source_t`，以及 episode 级别的
`mixer_state_segments` / `mixer_state_collection_t_env`。因此 online mixer、target mixer、
价值诊断读取的都是当时真实缓存的 state，旧 replay 不会按当前阶段重新解释。
普通 actor 始终读取实时 obs；该功能仅允许历史 qmix_minimal，避免其他模型依赖 context state。

learner 每批验证缓存值和来源边界，并记录 `mixer_input/refresh_*` 指标：
当前 K、replay K 的均值/最小/最大、缓存年龄均值。
回放要求从 episode reset 开始，只允许右侧截断/填充，沿用现有完整 episode 采样。
保存模型时额外记录刷新协议，加载时要求一致；现有 checkpoint 不保存完整 replay/RNG，
因此不能把普通 checkpoint 加载称为精确断点续训。

禁用该功能时不增加 replay 字段，不改历史采集数据、随机数、参数或训练逻辑。

## 验证与服务器运行

`scripts/smoke_test_mixer_state_refresh.py` 使用合成环境检查真实两种 runner、提前终止、
reset、train/eval、回放跨阶段、online/target 输入、行动头输出、Adam/RMSprop 参数更新一致性、
诊断和 checkpoint 协议。输入尺寸覆盖 5m6m/8m9m；不代表已经运行真实 SC2 训练。
`scripts/smoke_test_submit_mixer_state_refresh.py` 检查九份计划与历史配置差异和配额门槛。

在 OzSTAR 登录节点执行：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_mixer_state_refresh_9jobs.py
```

不设置 SUBMIT=YES 时仅预览。提交前检查个人 /home 原始配额：至少 5 GiB 和 50,000 个空余 inode，
失败则不提交；这不是九个完整训练的容量保证，也不自动清理数据或关闭已有视频。
map、配置、合成测试和 sbatch test-only 均通过后才提交；相同名字 active/completed job 保留，
不取消现有任务。回执保存在 runtime 的 ozstar_logs，批次名为
`5m6m_mixer_state_refresh_10m_9jobs_20261023`。
W&B 离线写入沿用历史设置；本改动不声称已经提交 job 或配置了新批次自动上传/三种子聚合。
