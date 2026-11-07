# Task 04：位姿传播公式改进

本目录由作者原始提交 `f3b664e0e8c86b62c6ed2ea2a112261810228e1c`
独立导出，未覆盖原项目或只读备份。

## 改动范围

作者代码在一个 IMU 采样间隔内冻结步首姿态：

\[
v_{k+1}=v_k+(R_k a_k+g)\Delta t,
\qquad
p_{k+1}=p_k+v_k\Delta t+\frac12(R_k a_k+g)\Delta t^2.
\]

本版本仅把其中的加速度旋转积分替换为 `eq_ob.pdf` 的精确分段常值形式：

\[
\begin{aligned}
\phi_k &= \omega_k\Delta t,\\
R_{k+1} &= R_k\exp(\phi_k^\wedge),\\
v_{k+1} &= v_k+g\Delta t
          +R_kJ_l(\phi_k)a_k\Delta t,\\
p_{k+1} &= p_k+v_k\Delta t+\frac12g\Delta t^2
          +R_kQ_l(\phi_k)a_k\Delta t^2.
\end{aligned}
\]

旋转传播、协方差传播、UWB 更新、门限、偏置、噪声和地面约束均保持作者代码原样。
作者提交中的其他已知问题也没有在本任务中顺带修改。

## 推荐入口

在 Windows PowerShell 中进入本目录：

```powershell
Set-Location 'E:\0syncthing\Code\InEKF\imu-uwb-inekf-update-improve'
```

只运行公式、单步运动和改动范围测试：

```powershell
.\run_task04.ps1 -Mode test
```

运行测试并回放默认 const1 前 30 秒，生成前后对比图和指标：

```powershell
.\run_task04.ps1 -Mode quick
```

回放默认 CSV 的完整时长：

```powershell
.\run_task04.ps1 -Mode full
```

运行局部公式微基准和两组 8 轮 AB/BA 完整回放开销测试：

```powershell
.\run_task04.ps1 -Mode overhead
```

该测试约需 1--2 分钟，输出到 `results/task04_compute_overhead`。

指定其他数据和时长：

```powershell
.\run_task04.ps1 `
  -Mode quick `
  -AnchorPath 'E:\datasets\dataset\flight-dataset\survey-results\anchor_const4.npz' `
  -CsvPath 'E:\datasets\dataset\flight-dataset\csv-data\const4\const4-trial7-tdoa2-manual1.csv' `
  -DurationSeconds 60 `
  -OutputDir 'results\task04_const4_manual1_60s'
```

限制系统总 TDoA 更新率为 50 Hz（选择均匀时间网格最近的原始测量，不插值）：

```powershell
.\run_task04.ps1 -Mode quick -DurationSeconds 30 -UwbRateHz 50 `
  -OutputDir 'results\task04_const1_30s_uwb50hz'
```

每次对比会生成：

- `comparison.png`：轨迹、位置/速度/姿态误差、公式导致的状态差和汇总表；
- `metrics.json`：输入、源码哈希、事件数、RMSE、拒绝数和状态差；
- `summary.csv`：便于表格查看的简明指标。

## 已完成结果

- `results/task04_const1_30s`
- `results/task04_const4_manual1_60s`
- `results/task04_compute_overhead`
- `results/task04_const1_30s_uwb50hz`
- `results/task04_const4_manual1_60s_uwb50hz`

两组回放中改进前后的 IMU/UWB 调用数完全一致，UWB 拒绝数也完全一致。
由于 IMU 频率约为 1 kHz，高阶旋转积分带来的变化本来就应很小；对比图中的
“Magnified state difference”面板专门放大了这部分差异。单次 Python 墙钟时间
只作记录，不构成性能基准。

## 总 UWB 50 Hz 结论

“50 Hz”按系统合计 50 次 TDoA 更新/秒解释。const1 30 s 和 const4 manual1
60 s 的实际频率分别为 50.0053 Hz、50.0023 Hz。改进式相对作者式的四项 RMSE
变化仍只有 `+0.0074%` 到 `-0.0365%`，而且 const1 的位置/速度略差、const4
略好，不构成稳定显著收益。const4 最大位置状态差从全频 0.497 mm 增至
1.113 mm，但位置 RMSE 只改善 0.0041%。

原因是 UWB 降频不会把约 1 ms 的 IMU 传播步长改成 20 ms；改进式漏项的首项分别
按 `dt^2`（速度）和 `dt^3`（位置）缩放。UWB 降频主要恶化观测约束，而不是专门
放大传播离散化修正。若“每个测距对 50 Hz”，8 对合计约 400 Hz，则更接近已有
全频结果。详细解释和两组绝对精度变化见
`..\reports\task04_uwb50hz_assessment_2026-07-27.md`。

## 计算开销验证

开销基准固定 BLAS 单线程，先执行 3 秒交替预热，再对一般角度和小角分支各做
9 轮交替微基准；完整滤波使用两个 5 秒数据片段，各做 8 轮 AB/BA：

| 范围 | 作者公式 | 改进公式 | 开销 |
|---|---:|---:|---:|
| 一般角度名义位姿公式 | 41.581 us/次 | 57.333 us/次 | +37.884% |
| 小角名义位姿公式 | 41.009 us/次 | 57.181 us/次 | +39.436% |
| const1 完整 InEKF 回放 | 1.6072 s | 1.6938 s | +5.389% |
| const4 manual1 完整 InEKF 回放 | 1.5308 s | 1.6178 s | +5.682% |

局部公式约增加 16 us/次，但协方差传播、UWB 更新和历史记录仍占完整滤波的大部分，
所以端到端开销约为 5.4%--5.7%。两组片段均有 5,044 次预测；实测每 5 秒数据增加
约 0.087 秒桌面 CPU 时间，即约 17 ms/秒数据。没有增加持久状态内存。

完整数据、原始计时和图片：

- `results/task04_compute_overhead/benchmark.json`
- `results/task04_compute_overhead/raw_timings.csv`
- `results/task04_compute_overhead/overhead.png`
