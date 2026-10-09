# 以后 SMAC 实验默认记录测试战斗视频

适用于更新代码后新启动的所有 SMAC 模型，不限于 ID/Obs 对照，也不启动、重启或取消任何作业。
默认配置在 `src/config/default.yaml`：每 1,000,000 环境训练步，
取跨过该节点后的第一轮正常测试中的前 10 局。并行 batch=8 时为第一批 8 局、
第二批前 2 局；按 worker 编号选取，不按完成先后或输赢筛选。
实际测试时间例如 1,003,210 会保留在标题和 W&B 步数中；目录按 1,000,000 节点标记。
正常测试总局数、采样策略、训练数据和模型前向次数不变。
`test_nepisode` 必须至少为 10；小型调试运行可显式关闭视频。
不支持 GRF 视频，也不额外记录 force-open 测试。

## 画面和数据

视频是实际模拟器坐标重建的俯视图，不是 StarCraft 原生录像画面。
每个决策步一帧，默认 6 fps，最后加上终局状态。
展示敌我单位 ID、位置、当前血量/血条；右侧列出每个 agent 的血量、护盾和选定动作。
攻击/治疗连线表示选定目标，不表示已经造成伤害。
集火表显示每个敌人被哪些 agent 选为攻击目标，例如 `E0 <- A0,A1,A2,A3`。
这是同一决策步的联合动作，不把并行决策伪装成先后决策。
每帧是动作执行前状态，下一帧是执行后的真实状态；死去的单位保留灰色标记。

本地保存 MP4、逐步原始单位/动作 JSON 和节点清单：
`<local_results_path>/test_battle_videos/<unique_token>/seed_1_<run-hash>/step_001000000/episode_01.mp4`。
种子和运行名称指纹隔离目录，避免同一秒启动的不同条件覆盖彼此录像。
视频逐帧编码，不积累 RGB 帧数组或 PNG 目录；绘制/编码会增加节点测试的墙钟时间和存储。
如果快照缺失或编码失败，该局明确记错，不使用重复旧状态凑视频，不终止长训练。

## W&B 在哪里看

每个实验 run 的 Media/Workspace：`test_battle_video/episode_01` 至 `episode_10`。
同一个键随 1M/2M/3M 节点更新，可选择对应步数查看历史。
十局使用不同键，不会在同一步覆盖成一段视频。
离线运行的媒体由现有 tmux W&B 同步流程上传，不另建 analysis run。
状态指标是 `test_battle_video/available`、`collected`、`rendered`、`failed`。
启动时检查编码依赖；依赖缺失会明确警告、available=0，并禁用该进程的视频。
因此启动正式实验前先安装并执行以下检查。

## 集群更新和依赖检查（不提交作业）

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git fetch origin refs/heads/codex/linear-id-baselines &&
git merge --ff-only FETCH_HEAD &&
/home/kyang/.conda/envs/marl_cpu/bin/python -m pip install \
  'imageio>=2.9,<3' 'imageio-ffmpeg>=0.4,<0.7' Pillow &&
/home/kyang/.conda/envs/marl_cpu/bin/python scripts/smoke_test_periodic_battle_videos.py
```

检查使用明确标记的合成数据，不启动 SC2、不连接 W&B、不提交实验。
验证单/并行 runner、不同结束时间、前 10/32 局选取、终局对齐、重试、
胜/负局保留、编码失败不中断、十个上传键和真实 MP4 编码/解码。
合成数据不作为实验结果上传。真实集群 SC2 端到端仍需第一批新实验确认。

可调整 `test_battle_video_interval`、`test_battle_video_episodes`、
`test_battle_video_fps`、`test_battle_video_dir`；关闭用 `test_battle_videos=False`。
旧 `save_battle_trace=False` 不会关闭这项独立的新功能。
旧进程已加载的代码不会自动更新，也无法补录过去的战斗。
