# IMU–UWB InEKF 最小修正与偏置参数实验工程

本目录从论文作者公开提交
`f3b664e0e8c86b62c6ed2ea2a112261810228e1c` 出发，最初用于只修复能够由论文公式
和原代码直接证明的不一致；随后在保持原始滤波结构不变的前提下，加入了更符合
BMI088/数据残余量级的偏置先验和随机游走参数。

只读原作者版本位于 `../imu-uwb-inekf - backup`。本目录不是完整生产版本：
右不变协方差/NIS 坐标修复、开机静止更新、快速事件管线和 Numba 编译核位于
`../imu-uwb-inekf`。

## 相对原作者版本改了什么

实际算法代码只修改：

- `scripts/estimation/inekf.py`
- `scripts/estimation/main.py`

### 1. 统一偏置状态的物理顺序

论文误差状态是：

\[
\delta x=[\theta,v,p,b_g,b_a].
\]

原作者代码同时存在三种不一致含义：名义传播把前 3 维偏置当 \(b_a\)，传播矩阵把
第一个偏置块当 \(b_g\)，初始协方差和过程噪声又按另一顺序赋值。

本目录统一为：

```text
bias[0:3] = b_g
bias[3:6] = b_a
P/Q block 4 = gyro bias
P/Q block 5 = accelerometer bias
```

这样名义传播、误差状态、初始协方差、过程噪声和更新后的六维偏置具有唯一物理含义。

### 2. 修复 SO(3) 左雅可比的零角符号

原代码一般分支和论文式 (11) 的一阶项为正号，但 `np.isclose(angle,0)` 分支返回
\(I-\frac12\phi^\wedge\)。本目录改为
\(I+\frac12\phi^\wedge\)，消除分支处的符号不连续。

### 3. 放宽原作者近乎冻结的偏置参数

| 参数 | 原作者版本 | 本目录 |
|---|---:|---:|
| 陀螺偏置先验标准差 | \(10^{-6}\,\mathrm{rad/s}\) | \(10^{-3}\,\mathrm{rad/s}\) |
| 加计偏置先验标准差 | \(10^{-4}\,\mathrm{m/s^2}\) | \(0.1\,\mathrm{m/s^2}\) |
| 陀螺偏置随机游走幅值 | \(10^{-6}\) | \(3\times10^{-4}\) |
| 加计偏置随机游走幅值 | \(10^{-4}\) | \(3\times10^{-2}\) |

const1 起始约 1 s 的残余量级约为 \(6.5\times10^{-4}\,\mathrm{rad/s}\) 和
0.01–0.12 \(\mathrm{m/s^2}\)。原参数小约三个数量级，使偏置状态在实际回放中基本
冻结；新值是基于 BMI088、Crazyflie 开机陀螺校准和数据残余量级的一次性工程先验，
不是看完最终 RMSE 后反向扫描出来的最优值。

本目录仍沿用作者的 \(Q_b=\operatorname{diag}(\sigma_{rw}^2)\Delta t^2\) 语义，
没有偷换过程噪声离散模型。

## 明确没有改什么

本目录没有加入：

- 右不变初始协方差的 Adjoint 映射与坐标一致 NIS；
- 全圆未知航向的 2 rad 先验；
- 静止检测、ZARU、静态偏置初始化；
- 中点积分、严格异步子步、Van Loan 离散化；
- 新 UWB 杠杆臂 Jacobian、covariance reset；
- 快速事件调度、结构化矩阵核或 Numba 编译；
- 新的批量实验 CLI。

原始 O(N²) 浮点时间查找、地面截断、UWB 模型、事件语义和主要滤波计算规模都保留。

## 效果

### 只做论文一致性修正时

完整 const1 回放的四项 RMSE 相对原作者仅变化 +0.002% 至 +0.159%，曲线几乎完全
重合且没有新增发散。这一阶段的价值是**状态物理含义正确**，不是精度提升；修正没有
新增矩阵运算，也没有可辨识的固定运行开销。

### 加入偏置参数修订后

完整 `const1-trial1-tdoa2`：

| 指标 | 原作者版本 | 本目录 | 相对变化 |
|---|---:|---:|---:|
| 位置 RMSE | 0.106714 m | 0.097732 m | -8.416% |
| 速度 RMSE | 0.059415 m/s | 0.056660 m/s | -4.636% |
| 姿态测地 RMSE | 0.061043 rad | 0.058476 rad | -4.204% |
| yaw RMSE | 0.059456 rad | 0.057880 rad | -2.651% |
| UWB 拒绝 | 0/67,205 | 0/67,205 | 不变 |

修订版最终估计偏置约为：

```text
b_g = [ 0.000890,  0.000187, -0.000232] rad/s
b_a = [ 0.017339,  0.060315,  0.018835] m/s²
```

曲线平滑有限；原作者陀螺偏置最大范数仅
\(5.25\times10^{-10}\,\mathrm{rad/s}\)，说明原偏置状态几乎被冻结。

但在 const4 前 60 s、初始 yaw 误差 180° 的压力测试中：

- 位置 RMSE 变差 1.005%；
- 速度 RMSE 变差 4.992%；
- 姿态 RMSE 改善 0.602%；
- yaw RMSE 变差 0.777%；
- 持续进入 5° 的时间延后 0.540 s。

该场景没有发散，但加计 \(b_y\) 后半程持续上升，说明大姿态误差下偏置可能吸收姿态
或未建模误差。因此不能把 const1 的收益外推为“所有场景都优于原作者”。若要形成
论文级通用参数结论，还需要多轨迹冻结验证、静止长记录或温变实验以及 Allan
方差/偏置稳定性标定。

## 环境与运行

- Python：`D:\conda-envs\imu-uwb-inekf\python.exe`
- 数据：`E:\datasets\dataset`

运行本目录的原式入口：

```powershell
$env:MPLBACKEND = "Agg"
Set-Location "E:\0syncthing\Code\InEKF\imu-uwb-inekf-fix"
& "D:\conda-envs\imu-uwb-inekf\python.exe" scripts\estimation\main.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv"
```

本目录没有内置新的公平比较器；已有原作者/修正版配对结果由仓库外的统一比较脚本生成，
以避免为了测试而继续改动该最小工程。机器可读结果和已检查图片位于：

- `results/paper_alignment_const1_full`
- `results/bias_parameters_const1_full`
- `results/bias_parameters_const4_60s_yaw180`

图片检查结论：const1 轨迹连续，后半程姿态/yaw 误差降低；const4 无异常跳变或发散，
但加计偏置存在持续爬升，速度误差略差。

## 何时使用这个目录

- 需要审查“论文公式与作者代码的最小不一致”时使用本目录；
- 需要研究偏置先验/随机游走对原始结构的单独影响时使用本目录；
- 需要当前完整算法与编译加速时使用 `../imu-uwb-inekf`；
- 需要严格保持作者算法、只比较计算速度时使用 `../imu-uwb-inekf-acc`；
- `../imu-uwb-inekf - backup` 永远只读。

## 详细报告

- `../reports/paper_code_alignment_fix_report_2026-07-23.md`
- `../reports/bias_parameter_revision_report_2026-07-23.md`
- `../reports/bias_order_causal_ablation_report_2026-07-23.md`
- `../tex/paper_code_alignment_fix.tex`
- `../tex/paper_code_alignment_fix.pdf`
- `../tex/bias_parameter_revision.tex`
- `../tex/bias_parameter_revision.pdf`
