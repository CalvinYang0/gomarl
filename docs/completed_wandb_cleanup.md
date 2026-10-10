# 已上传且已结束任务的 W&B 本地副本清理

入口 `scripts/ozstar_clean_uploaded_completed_runs.py`，默认仅预览。
不要求整个队列为空：可保留其他运行中任务，只逐项处理已结束的 job。
不会修改训练或 tmux，不取消作业。

删除条件全部满足才清理：

1. Slurm 输出日志明确提及这个精确 offline-run 目录，关联 job 的 WorkDir 为本仓库。
2. `squeue` 中没有该 job，`sacct` 确认为终态；未知、排队、运行、重排任务保留。
3. 本地 W&B 记录有原始 run 身份和 exit，且至少 5 分钟未修改。
   W&B 0.18.7 core 正常关闭没有旧 SDK 的 final 记录，因此不要求它。
4. 再做最终 sync，不能用“之前传过一次”或旧 `.synced` 标记代替。
5. 原目标云端同名 run 为终态；用完整 `scan_history` 逐 step 核对所有本地记录的键和值。
6. `files/` 中媒体与自定义文件的云端尺寸、MD5 一致。SDK 配置/摘要/console 文件不是字节级备份，
   不对它们作字节一致性承诺；Sacred 原始配置与数值记录保留。
7. 同步与核验后再次检查 job 和目录快照：同步本身允许 SDK 重写配置/摘要等已声明文件；
   其他 payload 改变或核验期间任何文件增加/变化都阻止清理，symlink 一律不接受。

仅永久删除准确的 `<runtime>/wandb/offline-run-...` 目录。云端保留已核验的历史及媒体，
本地 `.wandb` 二进制与 SDK debug 日志删除后不能原样恢复。
**不删除 Sacred、Slurm 日志、共享缓存、模型或其他目录。**
不清理缺少关闭记录的崩溃任务，也不会为释放空间跳过验证；云端暂不可读就保留并报错。
在 W&B 根目录保存 `completed_cleanup_audit.jsonl`，先落盘核验凭据才执行删除。
与当前运行任务同步器共享 `.gomarl-sync-once.lock`，避免同时同步同一目录。

预览（不上传、不删除）：

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_clean_uploaded_completed_runs.py
```

执行最终同步、核验及清理：

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_clean_uploaded_completed_runs.py --apply
```

默认为 `/home/kyang/gomarl-runtime/gomarl-dual-branch`，只允许明确的非 symlink 实验目录位于
`/home/kyang` 下，不接受用户主目录、仓库根或系统根作为运行目录。
不同旧运行目录需用 `--runtime-root` 明确指定，不递归扫描或清空整个 `/home`。
默认仅核验 `hjh331-sjtu/gomarl` 的原始记录；发现不同云端目标就保留。

本地测试仅使用临时合成 W&B 记录、模拟 Slurm 和模拟云端，不实际删除用户实验或上传数据：
`python scripts/smoke_test_clean_uploaded_completed_runs.py`。

参考：[W&B 文件元数据（尺寸与 MD5）](https://docs.wandb.ai/models/ref/python/public-api/file)。
