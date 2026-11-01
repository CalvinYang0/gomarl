# 视频与已上传数据留存策略（2026-11-01）

用户要求后续版本不上传视频，并删除已经上传的数据。

新进程默认 `disable_video_media=True`：配置合并完成后禁用测试战斗视频采集、
轨迹视频生成和上传。即使旧提交器显式设置 `test_battle_videos=True` 也会覆盖。
Logger 再阻止视频写入 W&B；胜率、损失、热力图、其他静态诊断与测试局数不变。
不改变模型、门控、refresh、actor、TD 或训练预算。已经启动的进程不会热更新。
只有用户以后明确要求视频时才设置 `disable_video_media=False` 并启用相应视频选项。

已上传的本地副本可删除，无需逐次重新申请授权，但仅限已结束、归属明确且
云端备份完整的内容。不删除云端，不用 mean3 或 summary 代替原 run 的完整备份。
当前已验证清理入口只覆盖本地 W&B offline-run 副本：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_uploaded_completed_runs.py --apply --only-already-uploaded
quota -sv
```

它不重新上传文件；按原 run 身份核对全部本地 history 与所保存文件/media 的云端副本，
使用共享锁，重查 Slurm 状态和文件变动，写审计后删除。失败、活动或未知项保留。
这不是轻量 summary 检查，可能较慢；不可用删锁、直接清空目录来加速。

Sacred、外部视频/诊断源文件、checkpoint、figures 并非必然都包含于 W&B 副本，
当前入口不删这些；后续只有逐项证明已上传时才能按本授权清理，不能笼统删除 results。
作业日志与提交/上传/清理回执保留用于溯源。W&B 本地 SDK 缓冲删除后不能原样恢复，
但已验证的上传数据仍在云端；本地原始曲线保留可用于后续重画 mean3。

本改动不安装自动清理服务，不启动训练，不改变正在运行任务，不证明服务器已清理。
