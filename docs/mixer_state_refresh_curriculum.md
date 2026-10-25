# Mixer state 刷新频率消融（2026-10-23）

## 2026-10-26：补充 5M / 10M 阶段长度

只增加 5m_vs_6m 的 S=5,000,000、10,000,000 两组，各 seeds 1/2/3，共六个新 job。
严格复用旧九任务的模型、两层 ELU 行动头、TD、采集、测试、视频和诊断配置；
仍为约 10M 总训练（t_max=10,050,000）、24h，不重新提交 0.5M/1M/2M。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_mixer_state_refresh_slow_6jobs.py
```

批次为 `5m6m_mixer_state_refresh_slow_10m_6jobs_20261026`，名称分别为
`smac_5m6m_qmix_state_refresh_5000k_10m_s{1,2,3}_refresh20261026` 和
`smac_5m6m_qmix_state_refresh_10000k_10m_s{1,2,3}_refresh20261026`。
新批次与旧九任务隔离，不拼接旧 seed 或断点续训；得到实际 job ID 才能确认提交。
保留原 5 GiB / 50,000 inode 门槛、地图/模型检查、先全部 Slurm 预检及逐任务落盘回执。
同名活动/成功任务保留；不清理空间、不降低门槛、不取消其他任务。

协议没有改成永久 K=1：S=5M 时，T<5M 为 K=1、5M<=T<10M 为 K=2，
T>=10M 为 K=3；S=10M 时，T<10M 为 K=1、T>=10M 为 K=2。
由于沿用最后 50k 步预算余量，10M 阶段在余量部分可能采到 K=2 的 episode，
最终评估也按当时 T 使用 K=2；不能将其描述为整个运行严格固定 K=1。
每局 K 仍在 reset 固定，只冻结 mixer state；actor 实时 obs 不变。
旧 `ozstar_upload_mixer_state_refresh_results.py` 仍仅处理原九任务，本次不更改或执行上传。

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
W&B 离线写入沿用历史设置；这不是新批次的长期自动上传循环。

## 完成后更新（2026-10-25）

用户提供的 sacct 显示 19086819–19086827 共九个 job 全部 COMPLETED / 0:0。
调度完成不等同于每条测试曲线已覆盖 10M；更新器另行检查实际数据。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_mixer_state_refresh_results.py
```

仅本批：补传/验证九个原始 run，生成三组单 seed 面板及 mean ± sample std 曲线，
上传三条可单独勾选、未平滑的 mean3 scalar run。只创建 5m6m section；不混入旧基线。
按 manifest 精确 job ID 核对 accounting，非 COMPLETED / 0:0 不进入上传。
不取消、不重跑、不删除、不更改 tmux 范围。`--no-upload` 仅生成本地结果，
不查询调度器或云端，也不代表数据已经上传。输出与审计报告在 runtime 的
`figures/5m6m_mixer_state_refresh_20261023_3seeds/`。
缺失/不足 10M、原 run 同步失败或 workspace 保存失败均报告 PARTIAL/INCOMPLETE，
不以已生成图片冒充完整上传；本地数据缺失时应检查云端来源，不猜测补齐。
