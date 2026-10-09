# 8m_vs_9m 的 VDN / QMIX 对照

新增两组，各种子 1、2、3，共六个作业；不取消、重启或补提交其他模型。
明确使用 `8m_vs_9m`（8 Marines 对 9 Marines），不是 `8m`（8 对 8）。
提交前检查已安装地图的单位数量和实际地图文件。

| 设置 | 两组共用 |
| --- | --- |
| 训练预算 | 10M，t_max=10,050,000，跨过最终测试节点 |
| 主网络输入 | 局部 obs + 上一步动作 + one-hot agent ID |
| 行动价值网络 | 共享 GRU-64 + 固定 64→64→15 ELU 行动头 |
| 优化器 / rollout | clean_hyper 原有 Adam、lr=0.001；batch_run=8、batch=128、buffer=5000 |
| 学习目标 | 原有 Double-Q 动作选择 + TD(λ)，λ=0.6，γ=0.99 |
| 损失 | 有效转移上的主 TD 平方误差，系数 1；无 gate、KL、辅助 TD 或其他辅助目标 |
| 测试 | test_greedy=True，每 10K steps 测 32 局 |
| 诊断 | 每 100K steps 的测试价值诊断和训练标量日志 |
| 视频 | 每 1M steps 正常测试的前 10 局，不另采额外测试局 |
| 资源 | 每作业 28 CPU、24G、48 小时 |

唯一组间模型差异是 `mixer=vdn` / `mixer=qmix`：VDN 求和，QMIX 用全局 state
产生正权重混合。主网络都不是超网络生成的行动头，不添加新的热身训练。
这沿用本仓库现有固定头 baseline（两层 ELU 头）；不是官方 PyMARL 单层读出头的逐项复现。
因此结论应表述为本仓库、同一训练设置下的 VDN / QMIX 比较，不与论文表格混算。

运行名称：

- `smac_8m9m_vdn_baseline_10m_s{1,2,3}_valuediag`
- `smac_8m9m_qmix_baseline_10m_s{1,2,3}_valuediag`

重复提交保留同名 active/completed 作业；不使用旧 attention-ID 结果替代新的线性 ID。
现有按 RUNNING 作业及 WorkDir 自动发现的 W&B 同步流程无需添加名称白名单。
只记录配置不能确认服务器已提交；需要实际输出六个 job ID。

## 集群提交

先在登录节点执行；使用 FETCH_HEAD，不依赖 origin 分支的 tracking/refspec 配置。
不使用登录 shell 的 `set -e`，任何失败会停止本组命令但不会断开 SSH。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
SUBMIT=YES /home/kyang/.conda/envs/marl_cpu/bin/python \
  scripts/ozstar_submit_8m9m_vdn_qmix_10m_3seeds.py
```

如视频依赖尚未安装，先按 `docs/test_battle_videos.md` 安装 imageio、imageio-ffmpeg、Pillow。
提交脚本会先检查地图、配额、参数声明、真实地图维度的主网络训练、mixer 及视频路径，
然后对全部缺失作业作 Slurm test-only 检查，再开始正式提交。
本地网络测试不启动 SC2，视频预检使用明确标记的合成数据，不上传为实验结果。

结果比较请使用 `test_battle_won_mean`，保留单种子曲线和未经平滑的数据。
使用同一环境步区间对比，不把某组单种子最好 checkpoint 与另一组末期三种子均值比较。
能否“VDN 更好”和能否“后期退化”是两个需要分开统计的问题。
