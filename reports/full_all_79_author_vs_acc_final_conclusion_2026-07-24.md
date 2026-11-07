# 79 个完整文件：作者原始算法 vs 纯计算加速版最终测试报告

完整结论与逐条结果位于：

`imu-uwb-inekf-acc/results/full_all_79_parallel/FINAL_CONCLUSION.md`

## 核心结果

- 全部 79 个 CSV 完整回放，79/79 成功、失败 0；
- 13,548,996 个事件、9,984,576 次预测、3,564,341 次 UWB 更新；
- 单文件加速比中位数 38.9794x，范围 31.0226x--47.9746x；
- 作者/加速中位耗时之和 2,798.184/73.485 s，比值 38.0781x；
- 两边均拒绝 17,178 条 UWB，拒绝决策变化总数 0；
- 最大旋转/速度/位置/偏置/NIS 差为
  `3.33e-14/4.84e-14/3.80e-14/9.85e-18/8.27e-9`；
- 最大 RMSE 绝对差不超过 `6.05e-15`；
- 端到端墙钟 11.41 分钟，其中包含一次前台监控连接中断和 `--resume` 续跑；
- 79 个 case 的指标、两类图和日志全部存在，缺失 0、错误日志 0。

## 解释

纯计算加速版已经在全部 79 个文件上证明：它没有改变作者算法的效果，只消除了
Python 热循环开销。可以报告为“全数据集约 31--48 倍，中位数约 39 倍”，但
不能把它表述为定位精度提升。

原作者算法在 const4 部分文件上的位置/yaw 表现明显弱于其它 constellation：
最大位置 RMSE 0.8882 m，最大 yaw RMSE 10.6238 deg。对比图显示两种实现逐点
重合，因此这是原算法/数据组合的既有现象，不是加速版退化。

并行计时用于评估全量吞吐；论文中的单核绝对速度仍应在空闲机器上以
`--jobs 1` 对代表文件复测。

## 输出

- `imu-uwb-inekf-acc/results/full_all_79_parallel/summary.csv`
- `imu-uwb-inekf-acc/results/full_all_79_parallel/summary.json`
- `imu-uwb-inekf-acc/results/full_all_79_parallel/summary.png`
- `imu-uwb-inekf-acc/results/full_all_79_parallel/FINAL_CONCLUSION.md`
- 79 个 case 子目录中的 `metrics.json`、`comparison.png`、
  `nis_and_rejections.png` 与运行日志。

