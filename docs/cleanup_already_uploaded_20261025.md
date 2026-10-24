# 仅清理已上传的本地任务副本

用户授权删除之前已上传任务的本地数据以释放空间/inode。
本次采用保守子集：只删除 runtime/wandb 下可验证结束且云端完整匹配的离线 run 目录，
不删 results（包括 Sacred、视频/诊断原文件）、figures、代码、Conda 环境、Slurm 日志或提交回执。
云端 W&B 记录保持不变。没有把“上传了 mean3”当成原始 run 备份证明。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_uploaded_completed_runs.py --apply --only-already-uploaded
quota -sv
```

新选项不执行最终补传，也不写云端；缺数据或云端无法核验则保留。
`--apply` 是实际本地删除；不带 apply 只预览，本地候选检查不代表已通过云端全量核验。
不能与跳过云端核验的 `--delete-ended-local` 混用。

执行时从 Slurm 日志确认 run 与 job 的归属，检查所有相关 job 终态和当前 live/requeue 状态；
使用与同步工具相同的 `.gomarl-sync-once.lock`，不删除锁本身、不停止同步服务。
逐个核对原 run ID、项目、名称、关闭记录，流式比对完整 history 与云端保存文件的大小/MD5，
重查文件未变和 job 未恢复运行，先落盘审计再删除；身份不明、近期修改、符号链接异常和任何核验失败都保留。
仅允许 SDK debug-core.log 的链接本身，不跟随或删除其共享目标。

这是删除本地 W&B 副本，不是清空所有任务数据；如仅此不足以达到门槛，另行报告，不自动扩大删除范围。
已核验的训练 history/保存文件可从原云端 run 获取，未上传的本地 SDK debug 日志不保证可恢复。
清理后以 quota 实测余量为准，不把候选文件体积当成已释放配额。
脚本遇到单项核验失败会保留该项并继续，最后返回非零；即使如此也应执行 quota 检查已完成的部分清理。
不自动启动新的训练或重新提交 12 个 SMAC 任务。

## 人工确认备份后，不再查询云端

用户已明确表示人工检查完成、不要继续云端核验。可以停止旧核验进程（当前终端 Ctrl+C），
使用既有的 `--delete-ended-local` 模式，只清已证明结束且不再变化的本地 W&B run。
此模式不校验 SDK 关闭记录或云端备份；是否可恢复取决于用户人工确认的备份。
SDK 保存的文件/目录符号链接只删除链接本身，不跟随、不删除其指向的原文件。
活动/未知任务、Sacred、作业日志、诊断原文件及云端记录仍保留。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_uploaded_completed_runs.py --apply --delete-ended-local
quota -sv
```
