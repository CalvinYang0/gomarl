# 仅清已结束旧 job 的战斗视频

用户已明确授权删除旧 job 本地数据、不检查云端、释放空间后提交新 soft/refresh 18 任务。
用户实测队列为空，`results/test_battle_videos` 约3.1G /13,973 inode；此前只删 W&B 副本后
quota 仍只有3.765GiB /41,700 inode。本次只进一步删除旧战斗视频及其 episode JSON。

`ozstar_clean_ended_battle_videos.py` 从提交回执的精确 group/run名/seed/job ID，结合录像
目录 `group__timestamp/seed_N_sha256(run_name)[:12]` 识别归属。只处理 fresh sacct 证明终态且
WorkDir 匹配的旧 job。保护新 `soft_onefill_refresh_three_maps_10m_18jobs_20261028` 批次；
未知或混淆归属、无回执、无终态、符号链接目录保持不动，不猜对应关系。

复用已有安全 helper，但不运行“清空全项目”的入口：要求队列为空，并检查两个登录节点
无相关 writer；持有原有同步锁，每个删除目标前重查安全状态、Slurm终态和文件快照。
只删 seed 级叶目录，先写并 fsync 审计，再删除；不跟随内部符号链接，不删除类别根目录。
三百秒内有变化的目录保留。权限/SSH/调度查询失败立即停止，不绕过。

不做云端API、补传或备份；未备份的视频/轨迹删除后可能无法恢复。
保留 Sacred、门控热力图/JSON、figures、Slurm日志/manifest、代码、Conda和所有锁inode。
估算删除量不等于 quota 实际回收，以脚本最后 quota 和提交器再次检查为准。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_ended_battle_videos.py --apply &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_soft_onefill_refresh_three_maps_18jobs.py
```

不带 `--apply` 只做预览。清理/安全检查失败或容量不足不会提交；不降低5GiB/50,000 inode门槛。
清理报告 `ozstar_logs/cleanup_ended_battle_videos_*.jsonl`。
本地测试合成数据不代表服务器已删除或已提交。
