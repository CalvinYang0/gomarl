# 2026-10-26：16 个已完成任务补传

用户已授权上传刚核对的 16 个已完成任务，不包含其他 running/pending 作业。

| 内容 | Job IDs | 原始 run | mean3 |
| --- | --- | --- | --- |
| MMM2 QMIX / all-ones | 19185607–19185612 | 6 | 2 |
| 3s5z_vs_3s6z QMIX / all-ones | 19185613–19185618 | 6 | 2 |
| Corridor 普通 QMIX | 19217303–19217305 | 3 | 1 |
| 5m6m stage=5M、seed=1 | 19246004 | 1 | 不生成 |

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_completed_16jobs_20261026.py
```

只回填精确原始 W&B ID，不创建冒充训练记录的替代 run；读取对应提交回执、
重新检查 sacct 的 exact ID / WorkDir / COMPLETED / 0:0，任何不一致拒绝猜测。
使用现有 `.gomarl-sync-once.lock` 与上传器/清理互斥；锁忙时报告，不绕锁。
沿用已验证的追加同步和云端 scalar snapshot 进度核验，不伪称全量媒体逐字节核验。
保留本地原始数据；不删除、不取消、不提交训练，不改 tmux 或长期同步范围。

前 15 个任务先补传原始 run，再生成三地图总览和五张独立单模型三种子图。
每模型必须有 seeds 1/2/3；共同 10k env-step 网格计算未平滑 mean/sample std，
保存逐 seed CSV/inventory，不把不同地图/旧尝试或旧 5M 基线混合。
不足 10M 明确报告 PARTIAL，不补造尾段；mean3 是分析记录，不替代原始 run。
图表使用独立 analysis run，Media、mmm2、corridor、3s5z 是并列 section，每组大图一列。
3s5z section 此处明确代表 `3s5z_vs_3s6z`。

第 16 个任务只上传 `19246004` 原始记录，即使运行命令时其他 seed 已完成也不扩大范围，
不生成 1/3 冒充 mean3。五组基线出错不会阻止尝试这个独立 seed，最后总报告仍返回非零。

输出目录：runtime 下 `figures/completed_qmix_baselines_20261026_3seeds/`。
`upload_report.json` 为 15 任务/五 mean3 的详细结果；
`completed16_upload_report.json` 为含额外 seed 的总结果；
`model_panel_upload.json` 保存实际 analysis run / workspace 链接。
布局失败但图已上传时，用已有 helper 仅重试布局，不重新上传全部图片。
缺 wandb-workspaces 依赖时按报错检查当前环境，不自动升级 W&B SDK。

加 `--no-upload` 只生成本地五组三种子图/CSV，额外 seed 仅列计划，不访问云端/Slurm。
服务器由用户现有登录终端执行；代码推送、离线合成测试成功不等于真实上传完成。
