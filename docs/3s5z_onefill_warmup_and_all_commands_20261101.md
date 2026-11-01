# 3s5z：严格初值1、5M热身后二值冻结，与前一组刷新调度

新增三个模型，各seeds1/2/3，独立新cohort：
`3s5z_onefill_initial1_freeze5m_10m_9jobs_20261101`。

|CLI variant|设置|
|---|---|
|initial1|初始门控严格为1，全程可训练；直接参数p，每次optimizer.step后投影回[0,1]|
|freeze5m|保留原sigmoid初值0.1；前5M训练，到5M将p>0.9置1，其余（含等于）置0并永久冻结|
|freeze5m_refresh|上一版再加原refresh2000k，按最大episode长度分段保持raw state|

不把严格1偷换成0.99或无限logit。第一版不用sigmoid，也不在forward中clamp，
因此边界p=1仍然可以接收向内更新的梯度。每次更新后、target复制和诊断之前投影。
这改变了参数化，不是相对p0=.1原版的纯初值单因素消融，需如此标注。

两个热身版不改变原sigmoid参数化。冻结发生在第一次t_env>=5,000,000的
learner.train更新之前（采集有步长，因此实际记录时刻可能略大于5M）。
由online当前p生成一张mask，同时安装到online/target；不提前复制其他target权重。
门控梯度设None、requires_grad关闭、移除其optimizer状态，其他参数继续原TD训练。
后续target更新不会重新开启门控；checkpoint记录mask/冻结标志/实际冻结时刻，
恢复后仍保持冻结。阈值0.9是5M时一次性决策，不是测试阈值。

关闭字段的输入仍为1，不是0：x=mask*state+(1-mask)。refresh版先保持raw state，
再通过门控；不保持门控参数。Replay沿用原采集state快照，不按当前阶段重构。
三版测试时mixer state均实时；测试动作不经过mixer。保留旧GRU+两层ELU行动头、
单完整QMIX、原joint soft TD/optimizer/target频率，p1版以外不改参数化，不增加loss。
不生成/上传视频；指标、门控图、JSON保留，门控图在冻结后显示实际二值mask。

地图3s5z_vs_3s6z，每job28CPU/32G/24h，t_max=10,050,000（名义10M），不迁移旧权重。

## 一次提交全部六种模型（18个job）

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_3s5z_refresh_and_warmup_18jobs.py
```

包含上述新9个，以及前一组refresh_live的9个：
旧refresh2000k仅测试实时、8M到实时的faster_stages、8M到实时的larger_jumps。
前一组沿用原精确job名/GROUP，已运行/已成功任务由已有幂等逻辑保留，不改名再提交。
两组分别做配额/地图/模型/Slurm test-only预检并分别保存回执；不是全18个原子提交。
第二组失败不表示第一组也没提交，须看两份回执。

只提交新增三种模型（9个）：

```bash
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_3s5z_refresh_and_warmup_18jobs.py --suite warmup
```

只提交前一组刷新模型（9个）：末尾改`--suite refresh`。
只提交一个新模型的三种子：运行`ozstar_submit_3s5z_onefill_warmup_9jobs.py`，
加`--variant initial1`、`--variant freeze5m`或`--variant freeze5m_refresh`。
去掉`SUBMIT=YES`只预览。保留字节/inode门槛，不自动清理、取消作业、建立后台服务。

注意：若0.1起步的门在5M时全都未超过0.9，就会全部关闭，后半程mixer条件变成all-ones；
这是这次规则允许且应记录的结果，不会偷偷保留top-k或放宽阈值。

## 已验证（不代表真实训练已完成）

新增smoke测试在真实3s5z维度的合成batch上通过Adam/RMSProp更新：初值严格1且
可向内学习、投影先于target更新、5M前与原soft TD逐参数一致、阈值等号处理、
冻结后无梯度/动量、其余网络继续更新、online/target同mask、全关/全开边界、
checkpoint前后恢复、refresh replay校验及实际0/1热力图JSON。
合并入口身份/分组和配额拦截使用mock验证；没有启动真实作业或上传数据。
旧soft onefill、soft+refresh、上轮实时刷新、视频禁用及硬onefill/KL/MC回归通过。
