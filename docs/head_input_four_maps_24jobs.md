# 新补跑：四地图、八组、24 个任务

全部单分支线性、无 gate、无 KL/其他辅助损失；种子 1/2/3；每任务名义 10M，
`t_max=10050000` 与现有 10M 组一致，留出最后一次 10M 记录的步数余量。

| 地图 | 本批模型 | 任务数 |
| --- | --- | --- |
| `5m_vs_6m` | ID | 3 |
| `3s_vs_5z` | Obs、ID、all-ones | 9 |
| `MMM2` | Obs、ID、all-ones | 9 |
| `8m_vs_9m` | all-ones | 3 |

3s5z 沿用此前的 `3s_vs_5z`，不是混合兵种的 `3s5z` 地图。
all-ones 仅把超网络 Obs 填成全 1；主网络真实局部 Obs、上一动作、agent ID 和 GRU
不变，mixer 仍读真实全局状态。ID 条件是修正后的线性 agent-ID encoder，
不是历史 attention-ID，也不把 GRU 输出换成线性编码。
训练 TD、Double-Q、优化器及采样配置由同一 baseline profile/提交模板生成。
每任务 28 CPU、24G、48h，测试每 10K 步、32 局。

全部明确开启每 1M 的前 10 局战斗视频、policy 参数/输入敏感度及 value 标量诊断。
Obs 另开启超网络专属图；ID/ones 不支持 hyper-only 图，但有公共 policy 图。
依赖缺失会在启动训练前失败，不能静默关闭媒体。原始数据与上传的区别见
[后续可视化规则](future_smac_visualizations.md)。

所有新名称以 `_vizcoverage` 结尾，与旧 5M/10M/已停任务分开。
同一脚本重跑会保留这些 exact-name 正在跑、等待或已成功完成的任务，只提交缺的。
不取消旧任务，不拿旧数据冒充这批新运行。提交前验证四个实际地图文件、
config keys、真实模型的合成数据 TD 更新、可视化编码，以及所有缺失任务的 Slurm
资源参数；全部通过才开始提交。每成功提交一个 job 立即落盘清单，记录 commit、
完整参数、job ID，避免中断后丢失已提交列表。

本地已验证配置/模型/模拟调度/PNG/MP4；本地环境没有集群 SSH 密码，
**尚未真实提交或取得新 job ID**。实际提交用集群环境运行下列命令。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
/home/kyang/.conda/envs/marl_cpu/bin/python -m pip install 'imageio>=2.9' imageio-ffmpeg Pillow 'matplotlib>=3.5' &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_head_input_24jobs.py
```

不带 `SUBMIT=YES` 只打印 24 个计划，不提交。需要至少 5 GiB /home 配额余量；
这是启动门槛，不是 24 个录像任务的总容量保证。现有 running-job tmux 同步器按
实际运行的 repository job 动态发现，无需为新名称另写白名单；它不负责图的更新。

三种子总览已注册本批全部 24 个新名称，与旧 cohort 独立统计，不混合种子或架构。
新增 MMM2 后共有七张地图，原五/六地图命令兼容同一新入口：

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_smac_seven_maps_3seeds.py
```

上传分析 run 名称 `smac_seven_maps_obs_head_comparison_3seeds_figures`。
没有新数据时会在 inventory/panel 标为待测，不自动拿旧数据替代。
5m6m 专用 head-condition 图也新增这批 ID 的独立 cohort。
