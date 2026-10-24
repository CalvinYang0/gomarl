# 2026-10-25：重跑 18 个 + 刷新实验 9 个的更新

本次用户提供的 sacct 输出确认：

| 批次 | 调度状态 | 这次处理 |
| --- | --- | --- |
| 10/21 原始 18 个（18944777–18944791、18944951–18944953） | 全部 FAILED / 1:0；此前因配额不足中断 | 不混入全程结果；已有 PARTIAL mean3 保持独立，不覆盖或删除 |
| 10/22 从头重跑 18 个（19013233–19013252，编号有空缺） | 18/18 COMPLETED / 0:0 | 验证/补传原 run，生成 6 组三种子结果 |
| 10/23 刷新实验 9 个（19086819–19086827） | 9/9 COMPLETED / 0:0 | 验证/补传原 run，生成 3 组三种子结果 |

调度完成尚不保证所有测试记录都已保存到 10M；上传器检查真实测试终点。
这份记录不是 W&B 上传成功确认。

## 模型清单

重跑批次 `_rerun20261022`：

- 5m6m：`qmix_qtot_sum_bothones`、`qmix_qtot_sum_p10`、`qmix_qtot_alpha10_p10`、`qmix_soft_onefill_p10`。
- MMM2、corridor：各自的 `qmix_onefill_p10`。

刷新批次 `_refresh20261023`：

- 5m6m：`qmix_state_refresh_500k`、`qmix_state_refresh_1000k`、`qmix_state_refresh_2000k`。

每个地图/模型组合使用种子 1/2/3，因此共 27 个原始 run、9 个 mean3。
MMM2 和 corridor 虽然模型 label 相同，也必须是两组 mean3，不能按唯一 label 计数。
不拼接旧失败 attempt 和从头重跑的轨迹，不借旧长曲线填补新 attempt。

## 服务器命令

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_recent_completed_27jobs.py
```

可加 `--no-upload` 只核对本地库存并出图，不访问调度器/云端，不上传。
只处理重跑批次时用 `scripts/ozstar_upload_qtot_soft_rerun_results.py`；
只处理刷新批次时仍用 `scripts/ozstar_upload_mixer_state_refresh_results.py`。
与原命令一样支持 `--runtime-root`、`--project`；不启动 tmux，不提交训练、不清理数据。

每批先用其独立 manifest 的精确 job 名/ID 核对 accounting，必须 COMPLETED / 0:0。
原始 W&B 日志同步与图表 run、mean3 scalar run 分开记录。
mean3 不预平滑，按三种子共有区间对齐、保存样本标准差；图默认 centered window=10 test points。

W&B workspace 支持 Media、5m6m、mmm2、corridor 同级 section，每个模型独立大图；
不存在的数据分区不创建，未知地图不静默漏画。
两批独立执行，某批失败仍尝试另一批，最终返回明确的 PARTIAL/INCOMPLETE。

输出报告分别位于 runtime：

- `figures/qtot_soft_rerun_20261022_3seeds/upload_report.json`
- `figures/5m6m_mixer_state_refresh_20261023_3seeds/upload_report.json`

缺少本地来源不证明从未上传；需另查对应精确 run 的云端记录。
若图片已上传但 workspace 失败，用同目录 `model_panel_upload.json` 重试布局，
不要重新上传全部图片。原始同步目前验证 scalar 进度，不应宣称逐项媒体全量审计已通过。

## 本地验证

`scripts/smoke_test_upload_recent_27jobs.py` 使用合成 Sacred、回执和 W&B/scheduler mock，
覆盖 18+9 原 run 范围、6+3 mean3、旧 attempt 排除、三地图图像、三地图 section 路由和批次失败隔离。
合成测试不是服务器训练结果或真实上传证明。
