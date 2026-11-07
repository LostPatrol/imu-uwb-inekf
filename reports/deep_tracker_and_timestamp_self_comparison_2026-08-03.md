# 联合 Tracker 定义与 Deep 时间戳严格自对照报告

日期：2026-08-03

## 1. 本次问题与结论

本次核清了论文表 2--4 中两个含糊名称，并补做了当前 deep 算法自身的
时间戳未校正对照。

1. “联合 tracker”是所提 `DeepCoupledTwoFilter` 内部的原始 TWR
   15 维联合 InEKF 跟踪支路自身，不是普通 InEKF 基线，也不是最终
   自适应输出。
2. “同行即时生效版本”原意是零延时假设：CSV 同一行的 IMU/TWR
   仍配对，但令采集时刻等于日志时刻，即 `sensor_delay_s=0`。该名称
   容易被误解为计算即时更新，现已从论文中删除。
3. 原表 4 引用的 `results/deep_coupled_full` 不是严格时间戳消融：它还
   缺少当前版本的低健康输出保护和 decisive-range override。因此旧表
   同时混入算法版本变化，不能把差异全部归因于 0.20 s 时间校正。
4. 已使用当前完全相同的代码和配置重新进行 12 日志 × 2 初值 = 24 次
   零延时回放，并与当前 0.20 s 正式结果逐运行匹配。表 2 和表 4 均已
   改为这个严格自对照。

## 2. “联合 tracker”的准确含义

`DeepCoupledTwoFilter` 同时维护两个支路：

- `capture`：Task06 的 `LiftedTWRInEKF` 松耦合增广捕获器；
- `tracker`：`TWRInEKF` 原始距离联合跟踪器。

tracker 的状态误差为

\[
\delta x_{\rm tr}
= [\delta\theta^\top,\delta v^\top,\delta p^\top,
   \delta b_g^\top,\delta b_a^\top]^\top\in\mathbb R^{15}.
\]

它在同一个 15×15 协方差中保留姿态、速度、位置及两个 IMU 偏置之间的
交叉协方差。原始 TWR 距离创新首先直接约束位置/姿态相关量，再通过这些
交叉项修正速度和偏置。因此这里“联合”的含义是单个滤波器内部对
IMU 状态和原始 TWR 的联合更新，不是把 capture 与 tracker 的输出当作
两份独立信息做协方差融合。

表 3 的 tracker 曲线直接记录

\[
\hat p_{\rm tr}=X_{\rm tr}[1:3,5],\qquad
\hat v_{\rm tr}=X_{\rm tr}[1:3,4].
\]

它和普通 InEKF 使用相同的 `TWRInEKF` 核心，但工作条件不同：普通
InEKF 从初始状态独立运行、立即启用 NIS 门限、没有捕获重置；联合
tracker 使用 128 次 gate 预热，并可由 capture 状态和膨胀协方差离散
重置。最终“自适应输出”才是

\[
\hat p_{\rm deep}=(1-w)\hat p_{\rm cap}+w\hat p_{\rm tr},\qquad
\hat v_{\rm deep}=(1-w)\hat v_{\rm cap}+w\hat v_{\rm tr},
\]

其中健康监督器控制 `w` 在 0、0.3 和 1 之间平滑变化。

## 3. “同行即时生效版本”究竟是什么

该旧称指的是以下时间模型：

\[
t_k^{\rm acquisition}=t_k^{\rm log},\qquad \tau_s=0.
\]

含义仅是估计和评价时不把同一行 IMU/TWR 的共同采集时刻向前校正
0.20 s。它不是：

- 另一套 Kalman 滤波公式；
- 只平移 UWB 而不平移 IMU；
- 声称真实硬件没有通信或计算延迟；
- 在线 OOSM 缓冲与重传播实现。

正确校正版本使用

\[
t_k^{\rm acquisition}=t_k^{\rm log}-0.20\ \mathrm{s},
\]

IMU 与 TWR 共同平移，二者相对事件顺序不变，并在采集时刻插值
OptiTrack 真值。

## 4. 严格自对照方法

零延时批测输出：

`imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_zero_delay_current/real`

0.20 s 正式批测输出：

`imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_delay_full/real`

两套 `paired_metrics.csv`：

- 都有 24 个唯一的 `(sequence, offset_name)`；
- 列模式完全相同；
- 当前低健康保护和 decisive-range override 字段均存在；
- 16 项静态监督器配置逐行完全相同；
- 唯一主动改变的实验量是共同传感器延时 0 或 0.20 s。

新增脚本
`scripts/estimation/compare_deep_time_alignment.py` 会再次校验上述条件，
生成逐运行配对 CSV、汇总 CSV/JSON 和四指标差值图。

## 5. 表 2 新增的自身未校正对照

| 指标 | 松耦合增广，0.20 s | deep，0 s | deep，0.20 s | 已校正 deep - 松耦合配对中位差 | 已校正 deep - 未校正 deep 配对中位差 |
|---|---:|---:|---:|---:|---:|
| 全程位置 RMSE (m) | 0.386090 | 0.368936 | 0.338625 | -0.020030 | -0.028444 |
| 末 5 s 位置 RMSE (m) | 0.312664 | 0.237738 | 0.211746 | -0.092643 | -0.023142 |
| 全程速度 RMSE (m/s) | 0.518496 | 0.506414 | 0.490124 | -0.038338 | -0.036507 |
| 末 5 s 速度 RMSE (m/s) | 0.521877 | 0.482974 | 0.379749 | -0.075905 | -0.127923 |

注意，后三列的“方法值”是 24 次值的合并中位数，而两列差值是先逐运行
相减再取中位数。因此“中位数之差”不必等于“差值的中位数”。

## 6. 修正后的表 4 结论

| 指标 | deep 0 s 中位数 | deep 0.20 s 中位数 | 中位数相对变化 | 逐运行优/平/差 |
|---|---:|---:|---:|---:|
| 全程位置 RMSE | 0.368936 m | 0.338625 m | -8.2% | 23/0/1 |
| 末 5 s 位置 RMSE | 0.237738 m | 0.211746 m | -10.9% | 22/0/2 |
| 全程速度 RMSE | 0.506414 m/s | 0.490124 m/s | -3.2% | 20/0/4 |
| 末 5 s 速度 RMSE | 0.482974 m/s | 0.379749 m/s | -21.4% | 20/0/4 |

四项总体中位数均改善，但校正后仍分别有 1、2、4、4 次回放变差。
这不否定时间模型修正：物理对齐更正确并不等价于一个带门限、重置和
状态机的非线性估计器在每条有限样本轨迹上 RMSE 必然单调下降。

## 7. 检查与产物

- 当前代码零延时 24 次真实回放墙钟约 194.6 s，运行速度正常；
- 代表性零延时轨迹连续，误差有界，监督器权重变化合理；
- `paired_deltas.png` 保留全部 24 次差值，包含少量正柱，没有裁掉退化项；
- 新比较脚本 `compileall` 通过，仓库正式入口 28/28 项测试通过；
- 论文 TeX/PDF 已同步更新，表 2 加入自身未校正 deep，对表 3 tracker
  给出定义，表 4 改为同算法严格时间戳自对照。
- XeLaTeX 两遍编译得到 12 页 A4 PDF，无布局、引用或待重跑警告；
  12 页全部以 PNG 渲染检查，无裁切、重叠、坏字或不可读表格。
