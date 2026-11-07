# task01 执行情况报告（低开销 IMU 融合修订）

日期：2026-07-22  
代码仓库：`E:\0syncthing\Code\InEKF\imu-uwb-inekf`

## 1. 结论

已按追加要求完成修订：

1. 保留论文/代码不匹配的理论修正：统一误差与名义偏置顺序 `[theta,v,p,b_g,b_a]`，修复 SO(3) 小角左雅可比符号，保留稳定 Exp/Log、测地姿态评价、SLERP 对齐和 O(N) 查找。
2. 默认 `proposed` 撤销中点积分、逐 UWB 事件严格子步传播和 Van Loan 过程噪声离散化；这些仅保留在 `verified` 研究模式。
3. 新增算法层面的 IMU 融合：首次持续静止段检测、陀螺/加计偏置可观测性分离、零角速度伪测量（ZARU）。
4. 同一次 60 s 配对回放 Legacy/Proposed 为 17.824/17.677 s，处于约 1% 的计时波动内，可视为没有明显变慢；全部主要误差指标相对 Legacy 均改善。

上一版“约 25 s 增至 73.3 s”的三倍开销已从默认路径移除。该高开销方案只获得约 0.4%–3.5% 的关键 RMSE 改善，不值得作为生产默认值。

## 2. 文献检索与取舍

检索并阅读了以下直接相关的高水平期刊/会议工作：

- Li, Cao, Hanebeck, “Continuous-Time Ultra-Wideband-Inertial Fusion,” IEEE RA-L 2023, DOI: https://doi.org/10.1109/LRA.2023.3281932 。该工作以连续时间样条和滑窗融合处理异步 UWB/IMU，并在 UTIL 上验证实时性；迁移到当前递推 InEKF 需要架构重写，不符合本轮低开销约束。
- Goudar, Schoellig, “Online Spatio-temporal Calibration of Tightly-coupled Ultrawideband-aided Inertial Localization,” IEEE/RSJ IROS 2021, https://arxiv.org/abs/2108.00133 。其核心启示是只有满足可观测/可辨识条件的参数才应在线估计。
- Li et al., “Accurate 3D Localization for MAV Swarms by UWB and IMU Fusion,” IEEE ICCA 2018, DOI: https://doi.org/10.1109/ICCA.2018.8444329 。强调低延迟递推融合适合 MAV。
- Kilic et al., “Evaluation of the Benefits of Zero Velocity Update in Decentralized EKF-Based Cooperative Localization Algorithms…,” 2023, https://arxiv.org/abs/2306.17703 。提供 IMU/UWB/里程计框架中零状态约束的实证依据。

最终没有把连续时间样条、滑窗优化或在线时空标定硬塞入当前代码，而是选择与递推 InEKF 兼容、计算量可忽略的静止角速度信息融合。

## 3. 实现内容

### 3.1 保留的理论修正

- `scripts/estimation/inekf.py` 统一 `BG=9:12`、`BA=12:15` 和名义偏置 `[b_g,b_a]`。
- SO(3) 左雅可比小角分支改为 `I + 1/2 phi^ + 1/6(phi^)^2`，与一般分支连续。
- 保留近 π 稳定 Log、旋转正交投影、Joseph 协方差更新。
- 完整右不变 TDoA 杠杆臂 Jacobian 和 reset Jacobian 继续留在 `verified`，因局部组合曾使大 yaw 收敛退化，不冒充成熟默认改进。

### 3.2 撤销的高开销默认项

`FilterConfig.proposed()` 当前为：

- `midpoint_integration=False`
- `van_loan_discretization=False`
- `direct_lever_arm_jacobian=False`
- `covariance_reset=False`
- `ground_clamp=False`

`pipeline._fast_replay()` 每个 IMU 区间只传播一次；UWB 使用最近 IMU 状态，最大时间滞后约一个 IMU 周期（UTIL 约 1 ms）。`verified` 仍可复现逐异步事件严格传播和 Van Loan 离散化，便于后续研究。

### 3.3 算法层面的 IMU 融合

静止检测要求连续 0.2 s 内每个样本满足：

```text
abs(norm(acc)-1g) < 0.03g
norm(gyro) < 2 deg/s
```

只允许开机后的首次连续静止段；离开后永久关闭，避免把后续悬停误认为零速度。更新限频 10 Hz。

初始化不再把陀螺和加计标定捆绑拒绝：零角速度直接观测 `b_g`，所以静止时始终保留 `mean(gyro)`；加计候选偏置仍需通过 0.25 m/s² 一致性门控。const4 的候选加计偏置模长为 0.417 m/s²，故只拒绝加计偏置，不丢弃有效陀螺信息。

ZARU 使用 `z = omega_m - b_g_hat`、`H[:,BG] = -I` 和 0.5 deg/s 测量标准差，通过 3×3 创新协方差更新。

曾试验零速度行，但在 90° 初始 yaw 下，右不变速度/姿态耦合把静止残差错误注入 yaw，15 s 姿态 RMSE 从 0.398 恶化到 0.692 rad，已如实撤销。

## 4. 自动测试

命令：

```powershell
D:\miniconda\Scripts\conda.exe run --no-capture-output `
  -p D:\conda-envs\imu-uwb-inekf python tests\test_inekf.py
```

结果：9/9 通过。新增测试确认 ZARU 降低陀螺偏置误差且不改变速度；原有 SO(3)、SE₂(3)、UWB Jacobian、reset Jacobian、Van Loan 参考一致性、协方差 PSD 和偏置顺序测试继续通过。

## 5. 冻结数据结果

### const1-trial1-tdoa2，前 30 s，yaw 0°

| 方法 | 位置 RMSE (m) | 速度 RMSE (m/s) | 姿态 RMSE (rad) | yaw RMSE (rad) | 用时 (s) | 静止更新 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 0.110765 | 0.074837 | 0.054874 | 0.053124 | 9.184 | 0 |
| Proposed | **0.107131** | **0.072354** | **0.053332** | **0.052867** | **9.053** | 11 |

两者 UWB 拒绝均为 0/14289。

### const4-trial7-tdoa2-manual1，前 60 s，yaw 90°

| 方法 | 位置 (m) | 速度 (m/s) | 姿态 (rad) | yaw (rad) | 10° (s) | 5° (s) | 用时 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 0.356135 | 0.192817 | 0.211825 | 0.207027 | 14.389 | 16.367 | 17.824 |
| Proposed | **0.354669** | **0.186921** | **0.206193** | **0.200991** | **14.156** | **15.996** | **17.677** |

两者 UWB 拒绝均为 149/25915。

关闭 ZARU 后，const4 位置/速度/姿态/yaw 为 0.354698/0.187392/0.206626/0.201303。ZARU 对 RMSE 的增益较小但可测；主要收益来自两个 Proposed 变体共享的低开销基础（偏置顺序理论修正、快速调度与可观测性分离初始化）。ZARU 对 10°/5° 收敛时刻比关闭自身时慢约 0.06/0.07 s，但仍明显优于 Legacy，未掩盖该细小权衡。

## 6. 图片检查与产物

已查看两张 comparison 图和两张 summary 图：轨迹连续，无发散或新异常跳变；const4 的主要误差峰值在两方法中对应相同运动段；Proposed 没有放宽 UWB 门限。

结果目录：

- `imu-uwb-inekf/results/task01_final_fast_const1_30s`
- `imu-uwb-inekf/results/task01_final_fast_const4_60s`

数学说明：`tex/task01_verified_inekf.tex` 与同名 PDF。

## 7. 边界与后续

- UTIL 无可靠 IMU 偏置真值，本次不伪造偏置 RMSE 或完整 NEES。
- 当前快速路径给 UWB 引入不超过约 1 ms 的保持误差；若以后需要连续时间精确异步处理，应采用结构化预积分或连续时间估计，并单独证明实时性。
- 完整右不变 TDoA Jacobian 仍需传播、测量、注入和 reset 的端到端统一推导后再进入默认模式。
- 连续时间样条与在线时空标定是合理后续方向，但不应在没有时延/外参可辨识证据时盲目增加状态和计算量。

## 8. 2026-07-23：恢复交互式完整对比绘图

`scripts/estimation/main.py` 新增 `--show-comparison`。该模式在同一数据、初始 yaw 和时长下顺序运行 `legacy` 与 `--variant`（默认 `proposed`），然后弹出九个独立窗口：

1. 二维和三维轨迹；
2. x/y/z 位置估计与 Vicon；
3. x/y/z 速度估计与 Vicon；
4. x/y/z 位置误差与误差范数；
5. x/y/z 速度误差与误差范数；
6. SO(3) 相对旋转向量、测地姿态误差与 yaw 误差；
7. 按实际名义传播顺序规范化后的陀螺/加计偏置；
8. UWB NIS、门限和累计拒绝数；
9. 位置、速度、姿态、yaw RMSE、运行时间和拒绝率汇总。

绘图曲线默认最多 6000 点，只影响显示、不影响滤波或指标计算。每次滤波结束后先复制抽稀后的绘图数据并释放完整协方差历史，避免完整数据同时保留两套约 20 万事件结果造成过高内存占用。

验证命令使用 const1 前 2 s，实际生成九张 PNG；逐张检查未发现曲线错位、裁切或错误的偏置顺序。`MPLBACKEND=Agg` 下完成无人值守烟雾测试；目标 Conda 环境默认后端为 `qtagg`，实际 PowerShell 运行会弹出独立窗口。9/9 数值测试和 `py_compile` 继续通过。
