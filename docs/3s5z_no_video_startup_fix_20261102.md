# 2026-11-02：18个job启动失败，视频预检与禁用策略冲突

用户提供的记账：refresh组19764908–19764916、warmup组19765022–19765030，
全部FAILED/1:0，运行约9–16秒。两组代表性stderr均明确：

`run_sequential -> preflight_visualizations -> ValueError: Required SMAC visualization disabled: test_battle_videos`

这是已确认的共同启动错误，不是模型不收敛。其余16个未逐一读取stderr，不虚构
其他错误已被排除。早先PASS是合成测试，不是实际训练；Slurm test-only也不会运行Python。

此前视频策略关闭了采集/上传，且修改了后面的BattleVideoSession检查，但漏掉
run_sequential最前面的visualization_requirements强制开关。该遗漏导致视频关闭的
配置在环境runner创建前退出。修复保留test_visualizations_required=True：

- disable_video_media=True：不强制视频开关、不检查编码器或视频采样数量。
- 静态policy importance开关、matplotlib依赖、interval和测试局数覆盖检查仍执行。
- 显式disable_video_media=False的旧视频模式仍保留编码器/视频数量检查。
- 不修改门控、刷新公式、初始值、5M阈值冻结、TD、行动头或训练预算。

新增`smoke_test_no_video_startup.py`：按提交计划和真实YAML合并18个配置，经
args_sanity_check调用实际run_sequential，在预检之后的runner构造点截停，
证明故障路径已通过，不启动SC2、作业或W&B。编码器检查被设为一旦调用就报错；
另测静态诊断缺失仍拒绝、编码器opt-in仍强制，以及无视频输出inventory可写。
两组SMOKE_SCRIPTS均加入此检查。旧policy importance回归显式选择视频opt-in，
避免测试期望与新的无视频默认策略冲突。

服务器拉取后可用原提交命令重新提交失败任务；已有活动/成功同名任务保留。
每组仍独立预检和记录回执，不取消、不覆盖旧日志，不自动删除失败run。

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_3s5z_refresh_and_warmup_18jobs.py
```

本地通过只能确认本次启动错误及相关回归，不保证服务器后续SC2启动、资源或完整10M训练。

已通过的检查：新no-video启动路径、video_media_policy、policy_importance图像/依赖检查、
三种实时刷新调度、三种初值/冻结门控（Adam和RMSProp）。没有修改训练模型或loss。
