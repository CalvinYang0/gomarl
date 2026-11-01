# 3s5z soft one-fill 刷新调度消融

以 20261028 批次的 `qmix_soft_onefill_p10_refresh_2000k` 为代码基线；用户提供
mean3 标识 `f4fb0c`。不是从 mean3 恢复权重（聚合 run 不是 checkpoint）。
当前只实现/测试并提供提交入口，未在服务器启动训练。

共同设置：3s5z_vs_3s6z，seeds1/2/3，10.05M 环境步预算，每job32G/28CPU/24h。
保留旧 GRU+两层 ELU 行动头、单完整QMIX、逐state字段静态sigmoid p0=.1、
`x=p*held_state+(1-p)`、原soft learner joint TD/target/optimizer，无新loss。
不生成/上传视频，静态图和指标保留。agent局部obs始终实时。

## 三个版本

- `eval_live`：训练仍是每2M增加一个分段；只把测试mixer state改成每步更新。
- `faster_stages`：每次阶段切换仍然只加一个分段，但阶段间隔逐渐缩短。
- `larger_jumps`：仍每2M切换一次阶段，但每次分段数按几何速度增长。

三个版本测试均实时，包括t=L的终止/截断后状态。后两种加速调度暂取H=8M到达实时，
这是本次实现的可配置默认，不是用户已确认的最终超参。阶段长度S=2M。

设地图最大episode长度L。`linear`保留旧公式K=min(L,1+floor(t/S))。
`faster_stages`使用K=1+floor((L-1)*(min(t,H)/H)^2)：第j次+1转换发生在
ceil(H*sqrt(j/(L-1)))；阶段等待逐渐缩短，不是每2M才更新一次K。
`larger_jumps`使用K=ceil(L^(min(floor(t/S),H/S)/(H/S)))，要求H为S的整数倍。

按本地官方地图定义的L=170示例（服务器仍实际读取安装地图的episode_limit）：

|训练进度|旧linear|faster_stages|larger_jumps|
|---|---:|---:|---:|
|0M|1|1|1|
|2M|2|11|4|
|4M|3|43|14|
|6M|4|96|48|
|8M|5|170/实时|170/实时|

两种是不同调度曲线，不是相同累计state刷新预算的严格配对；不能把差异只归因于
“阶段数”。共同H便于比较到实时的时间，但中间暴露的信息量不同。

## 执行语义和兼容

K在每局/并行rollout reset固定，段边界仍floor(jL/K)，短episode不按实际长度重分。
先缓存raw state，再走当前online/target gate；不冻结门控或mixer参数。
Replay仍记录采集时的K、t_env、source_t和raw state，训练校验原采集调度，
不会用当前阶段重解释旧样本。新实时阶段t=L也更新；旧linear默认保留原终端保持行为。
新checkpoint合同记录schedule/full_steps/eval_live，避免用旧checkpoint假装新实验。
默认schedule=linear、eval_live=False，所以历史入口行为不变。

测试选动作不经过mixer：仅eval_live不会直接改变同一checkpoint的胜率；不必为了
这个评估口径单独重训。入口仍提供它的独立三种子fresh配置，供用户选择，绝不自动重跑。

## 命令（用户决定提交时）

预览九个任务：

```bash
REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_3s5z_soft_refresh_live_9jobs.py
```

确认8M调度后提交（含eval_live对照共9个）：

```bash
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_3s5z_soft_refresh_live_9jobs.py
```

可用`--variant eval_live`、`--variant faster_stages`、`--variant larger_jumps`只选一组三种子。
如需修改H或公式，必须用新label/cohort，不能复用已提交名称覆盖实验身份。
本入口保留字节/inode门槛、精确名称幂等、sbatch test-only与提交回执，不清理数据，
不取消任务、不设置后台同步，也不在上传前删除尚未备份的数据。

## 已做验证

`smoke_test_refresh_live_schedules.py`通过：两种真实runner路径（合成环境），
测试实时含t=L，训练单调调度/8M终点，混合阶段replay/损坏检测，原soft TD在
Adam与RMSProp下逐参数更新相同，旧ELU行动头不变，checkpoint合同与9个精确计划。
旧`smoke_test_mixer_state_refresh.py`、`smoke_test_soft_onefill_refresh.py`和
`smoke_test_video_media_policy.py`回归通过；不代表服务器训练或真实地图效果已验证。
