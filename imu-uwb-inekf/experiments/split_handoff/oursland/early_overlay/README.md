# 原作者后端的冻结早期 XY 归因

`protocol.json` 在读取本组任何真值分数前落盘。原四录制 × 0/180° 为 primary；另四个互补几何/协议录制 × 0/180° 为 extension，两组各 8 对，单列。均为 trial4、45 s、10–20 s 断距，起始 X/偏置来自传感器、P 为原 main 的数值默认。航向注入发生在共同初始 0.6 s，和本目录上级原生主组一致；这并非 `early_tdoa` 原实验的 10 s 注入流程。

原缓存只有初始 P，因此没有从轨迹行反推断点。每条件从相同初值重跑原作者原类，到恢复前复制整个对象，保留 X、P、X_hat、P_hat、bias 及事件队列的前一 IMU 时间。分支期间使用未修改的原作者 predict/correct、expm、噪声与原生队列。原四录制的 8 条正常基线与上级已有 NPZ 的时间和状态数组完全一致，最大差异均为 0。

唯一移植是冻结 `early_tdoa.horizontal_candidate` 的水平可用性及 `handoff_xy`。原阈值、候选 .14/.22 方差和两秒滚动缓存不改；原生队列同时间戳只收第一包，候选只看已到达事件。预留分支从 20 s 到触发不做无线标量校正。第一次接受时只改 XY、XY 物理协方差及其交叉块；R/v/z/bias 与全部非 XY 物理协方差保留。仅在这个外部写回边界同时设置 X_hat/P_hat，确保随后的原作者 radio-only 事件不会用旧 prior 覆盖 XY；作者类本身未改。之后只融合时间戳严格大于触发的原生包。

`early_reserved` 是真实预留分支。`early_switch` 仅在输出上于触发前显示正常作者轨、触发后显示预留分支；没有把正常轨的已融合无线信息写入预留分支。若没有接受，后者全程显示原作者轨。两种输出都保存、都评分，避免隐藏预留无线等待的代价。

16 个条件均触发，触发时刻 20.0502–20.7504 s；零数值失败；三个评分窗覆盖率均为 100%。全部写回的 R/v/z/bias 差异为 0，非 XY 物理协方差最大保留误差 9.58e-16；所有首个后续标量包均严格晚于触发。作者源码及冻结前端源码哈希前后不变。

下表为逐录制 RMSE 中位数，XY m / 航向 °。评分为原点后 [20,22)、[26,31)、[26,45) 的共同 20 Hz 网格，无对齐；最后一窗从 26.00 s 起，与上级主表从 26.05 s 起略有不同。

| 组别与方法 | [20,22) | [26,31) | [26,45) |
| --- | --- | --- | --- |
| primary 作者正常 | .9969 / 6.99 | .1070 / 5.72 | .1116 / 5.93 |
| primary 真实预留分支 | 1.3798 / 7.58 | .0747 / 6.78 | .0868 / 6.80 |
| primary 输出切换 | 1.0016 / 7.50 | .0747 / 6.78 | .0868 / 6.80 |
| extension 作者正常 | .9241 / 12.38 | .0620 / 9.54 | .0705 / 7.22 |
| extension 真实预留分支 | 1.2509 / 7.93 | .0624 / 6.90 | .0709 / 6.22 |
| extension 输出切换 | .9245 / 9.40 | .0624 / 6.90 | .0709 / 6.22 |

primary 输出切换的 XY 逐对改善数依次为 6/8、4/8、3/8，真实预留分支为 3/8、4/8、3/8。extension 输出切换为 5/8、6/8、4/8。**收益混合，不能宣称普遍更优**：primary 后续 XY 中位数下降，但多数配对未改善且航向中位数较差；extension 的 XY 中位数近乎不变；两组早期真实预留分支都有等待代价。若只看 few large XY gains，容易掩盖未改善的录制。

在项目根 PowerShell 重跑全部冻结条件：

```powershell
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\early_overlay\run.py --workers 16
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\early_overlay\analyze.py
```

复用 `cpu_replay.replay_pool`，16 个独立作业、每 worker 单 BLAS 线程，优先 P 物理核、独立 E 核、P 超线程，Windows 为可迁移调度偏好。传感器阶段墙钟 242.29 s（含进程收尾，最后一个任务返回约 240.0 s），约 3.96 条条件/分钟。缓存命中时不重算已完成条件；清空本实验自己的 `results` 后才是全量重新生成。

`results/paired.csv` 是全部 48 条方法/条件行，带 XY/yaw 绝对值、相对作者差值和覆盖；`overview.json`/`summary.csv` 为分组汇总；`availability_attempts.csv` 保留全部拒绝/接受理由；`integrity.json` 与 `baseline_validation.json` 给出保留及精确原轨检查；`paired_overlay.png` 可视化全部配对。每条件 NPZ 包含三条输出、同初始 X/P/bias、完整恢复分叉 X/P/X_hat/P_hat/bias 和前次 IMU 时间，可核查或继续接续，不依赖丢失 P 的轨迹重建。
