# 最新 soft one-fill / refresh 2M 的检查与补传

只检查 `soft_onefill_refresh_three_maps_10m_18jobs_20261028` 提交回执中的精确18个job：
3s5z_vs_3s6z、MMM2、Corridor × 原 soft one-fill / 加 refresh 2000k × seeds1/2/3。
不猜缺失ID，不用 squeue 消失认定完成，不拼入旧5m6m或硬门控 qsum 数据。

复用21-job已验证补传底层，但隔离本进程的cohort/label/output，并在结束时恢复。
先打印每job精确ID、状态、exit code、elapsed，以及状态计数；sacct WorkDir需匹配checkout。
未提交/回执冲突/未知状态显式报告；RUNNING/PENDING跳过，不重跑或取消。

已结束的原始run按原ID核对云端标量进度，本地新增/云端落后的才append；已同步的跳过。
失败/超时任务已有记录可补传，但绝不成为完整mean3。共享上传/清理锁仍生效，忙时本次退出。

每地图/模型必须三个job均COMPLETED/0:0，且真实测试历史都有10M共同区间，才上传
可勾选的未平滑mean3、sample std、逐seed指标。使用原有内容寻址ID，重复运行时核验
source digest/末步/seed_count后跳过已有相同结果，上传后再核验一次。
最多6个完整mean3；不足三种子的只报告等待，不用历史attempt补齐或延长短曲线。

本地保存每模型单seed图、聚合图、inventory/CSV；此入口不新增W&B图片workspace。
原始run标量核验不等于完整媒体备份；本地缺源不是“没跑过/云端必缺”的证明。
本次不清理数据，不重启tmux，不提交训练，不升级SDK。

在服务器执行：

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_soft_onefill_refresh_results.py
```

只看状态时末尾加 `--status-only`，不写文件、不联系W&B。
报告：runtime下 `figures/soft_onefill_refresh_three_maps_backfill_20261029/upload_report.json`。
用户本地终端SSH已断开时需自行重新连接，读取旧终端不代表实时状态或已执行补传。
