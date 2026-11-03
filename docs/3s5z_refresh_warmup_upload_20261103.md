# 3s5z 六版本18个任务：检查与补传

只处理20261101两个cohort的精确18个名称：三个refresh-live版本，以及严格初值1、
5M硬冻结、5M硬冻结+refresh2000k。默认项目hjh331-sjtu/gomarl。

此前19764908–19764916和19765022–19765030均启动失败。此入口不会把它们认作
修复后的完成任务；实际重提ID必须来自服务器回执，不能猜。

按回执候选ID限定sacct查询，核对WorkDir、精确JobName和Submit时间，选最新一次提交，
不是选最长曲线或最好结果。同名旧attempt保留在报告中但不进入补传或mean3。
任何候选缺记账/时间或身份冲突时标unknown，不偷偷回退旧成功任务。

原始run按选中的最新ID补传：已结束才同步，已核验标量进度的跳过，未同步/落后才append；
这不是完整媒体备份证明。最新尝试失败可补传已有原始记录，但不能成为完整mean3。
运行/排队/未知不重跑、不取消。

每模型仅三个latest job均COMPLETED/0:0且实际测试历史有共同10M才发布mean3，最多6个。
以选中job日志证明W&B ID/Sacred路径，拒绝名字相同但不属于该job的旧曲线或CSV。
本地已清理时，仅用日志里已证明的W&B原run ID查询云端，不全项目扫描；身份或数据
不足就明确报告。未平滑mean3、sample std和各seed可在W&B自行后处理。
本地保留PNG/PDF、inventory、原始曲线和聚合CSV；不创建图片workspace，不生成视频。
不删除任何数据，不提交训练，不启动tmux。清理请求应在上传结果核对后单独处理。

服务器执行：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_3s5z_refresh_warmup_results.py
```

只看状态加`--status-only`：只读回执和sacct，不联系W&B、不写图或报告。
默认输出：runtime下`figures/3s5z_refresh_warmup_backfill_20261103/upload_report.json`。

只提供命令或本地mock测试不证明服务器已执行，完成情况以实际输出/报告为准。

`smoke_test_upload_3s5z_refresh_warmup.py`已通过：重复回执按Submit选择、latest失败不回退、
WorkDir/name核验、最新job所属Sacred曲线绘图/完整mean3、本地清理后的精确云端ID回退、
status-only无上传/写图以及scope恢复（外部服务均mock）。原18任务上传入口回归也通过。
