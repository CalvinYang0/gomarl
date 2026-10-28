# 10 月 28 日：检查近期 21 个 job 并补传

范围仅包括尚需更新的三个明确批次，不扫描 W&B 全项目或重新提交训练：

- 5m6m mixer-state refresh：阶段 5M / 10M，seeds 1/2/3，6 个 job。
- Qtot-sum-p10：8m9m / Corridor / MMM2 / 3s5z_vs_3s6z，seeds 1/2/3，12 个 job。
- Corridor 单分支 all-ones：seeds 1/2/3，3 个 job。

从对应提交回执读取精确 job ID，并用 sacct 验证状态和 WorkDir；不存在、重复
attempt 回执或未知状态显式报告，不猜 job ID。正常运行/排队的 job 不上传、不取消。
中断/失败的 terminal job 只补已有原始 W&B 记录，不伪装成跑满的三种子。

原始记录逐原始 run ID 检查云端；云端已达到本地 scalar snapshot 的跳过，缺少/落后
的用已有 verified sync helper append。共享上传/清理锁保留。核验仅证明标量进度，
不声称逐项媒体完整。曾上传的 refresh-5M seed1 自动云端核验后跳过。

只有同地图/模型三个 job 都 COMPLETED/0:0，且实际测试历史都有完整10M共同区间，
才生成可单独勾选的未平滑 mean3、sample std 和三条 seed 指标。mean3 采用内容寻址 ID，
先核对精确云端 ID 的 source digest、截止步数、seed_count；相同已上传数据不重复发布。
不足三种子的模型只列等待，不生成假 mean3。局部失败不阻止其他完整模型补传。

本地保存单 seed 图、mean/std 图、inventory 和原始/聚合 CSV；本入口只上传原始训练记录
与 mean3 scalar runs，不额外创建图片 analysis run 或改变 workspace 布局。

上批 16 个原始记录和5个 mean3 已由用户终端输出确认补传成功，不再重复运行旧入口。
本地缺失不会被解读为云端也缺失；会在报告标记需要单独查云端/恢复来源，不合成训练日志。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_recent_21jobs_20261028.py
```

只检查状态、不写文件、不联系W&B时，末尾加 `--status-only`。

总报告：runtime 下 `figures/recent_21jobs_backfill_20261028/upload_report.json`。
没执行服务器命令前，代码或测试通过不代表已获取实时状态或已上传。
