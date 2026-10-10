# 已上传且已结束任务的 W&B 本地副本清理

入口 `scripts/ozstar_clean_uploaded_completed_runs.py`，默认仅预览。
不要求整个队列为空：可保留其他运行中任务，只逐项处理已结束的 job。
不会修改训练或 tmux，不取消作业。

## 直接删除已结束任务的本地目录（明确跳过云端核验）

`--delete-ended-local` 是独立的显式选项；默认模式的完整上传核验不变。
加 `--apply` 后，不同步、不调用 W&B API，也不要求 SDK exit 记录，
直接删除有精确日志归属、Slurm 确认已结束的本地 `offline-run-*` 目录。
包括已失败/取消/超时的任务。仍检查运行/排队/重排状态、目录边界、近期修改和删除前快照；
无法确认归属或终态的目录保留。与同步器共用锁，Sacred、作业日志、共享缓存均不动。
**本地二进制、媒体和 debug 日志永久删除；未完整上传的部分可能无法恢复。**

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_clean_uploaded_completed_runs.py --delete-ended-local --apply
```

去掉 `--apply` 仅预览，不上传、不删除。审计记录明确标注 `cloud_verified=false`。

## 默认完整上传核验模式

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
   其他 payload 改变或核验期间任何文件增加/变化都阻止清理。
   仅允许 SDK 的 `logs/debug-core.log` 软链接：用 `lstat/readlink` 核对链接本身，
   不遍历、不读取、不删除它指向的共享日志。其他软链接（包括媒体/数据/目录）仍拒绝。

仅永久删除准确的 `<runtime>/wandb/offline-run-...` 目录。云端保留已核验的历史及媒体，
本地 `.wandb` 二进制与 SDK debug 日志删除后不能原样恢复。
**不删除 Sacred、Slurm 日志、共享缓存、模型或其他目录。**
不清理缺少关闭记录的崩溃任务，也不会为释放空间跳过验证；云端暂不可读就保留并报错。
在 W&B 根目录保存 `completed_cleanup_audit.jsonl`，先落盘核验凭据才执行删除。
与当前运行任务同步器共享 `.gomarl-sync-once.lock`，避免同时同步同一目录。
会输出日志索引、等待同步锁、最终上传和云端核验阶段，方便区分缓慢操作和实际失败。

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

本地测试使用临时合成记录及真实 SDK 离线 run，模拟 Slurm 和云端；
覆盖正常 debug-core 软链接、悬空链接、共享日志保留、链接被替换及非日志链接拒绝。
不实际删除用户实验或上传数据：
`python scripts/smoke_test_clean_uploaded_completed_runs.py`。

参考：[W&B 文件元数据（尺寸与 MD5）](https://docs.wandb.ai/models/ref/python/public-api/file)。
