# Task04 位姿传播公式改进执行报告

日期：2026-07-27  
任务来源：`tasks/04位姿更新公式改进.md`

## 1. 执行结论

任务已按指定范围完成：

- 从论文作者原始提交
  `f3b664e0e8c86b62c6ed2ea2a112261810228e1c` 导出 71 个跟踪文件到独立目录
  `imu-uwb-inekf-update-improve`；
- 作者备份 `imu-uwb-inekf - backup` 没有被修改；
- 唯一被修改的作者跟踪文件是
  `scripts/estimation/inekf.py`；
- 只把名义速度/位置传播中的冻结姿态积分替换为
  `J_l(phi)`/`Q_l(phi)` 精确积分；
- 旋转传播、协方差、UWB、NIS 门限、偏置、噪声、状态注入和地面约束均保持作者
  实现原样；
- 新增了自动测试、数据集前后对比、PowerShell 一键入口、数学说明和可视化结果。

两组代表数据的四项 RMSE 都小幅改善，UWB 拒绝数保持不变；图片目视检查未发现
新增跳变、发散、轨迹断裂或绘图错误。

## 2. 论文核对与公式

### 2.1 作者论文及代码

Oursland 与 Mehrabian 原论文第 5 个 PDF 页面、式 (39) 以及作者
`InEKF.predict` 均使用

```text
R_{k+1} = R_k Exp(omega_k dt)
v_{k+1} = v_k + (R_k a_k + g) dt
p_{k+1} = p_k + v_k dt + 1/2 (R_k a_k + g) dt^2
```

速度和位置积分在整个采样间隔冻结 `R_k`。

### 2.2 `eq_ob.pdf`

已提取并渲染核对：

- PDF 第 6 页定义第二阶左雅可比 `Q_l`；
- PDF 第 10 页的式 (31)--(33) 给出分段常值 IMU 的精确单样本增量；
- 任务截图中的式 (35)--(37)与该页公式一致。

实现采用

```text
phi_k = omega_k dt
v_{k+1} = v_k + g dt + R_k J_l(phi_k) a_k dt
p_{k+1} = p_k + v_k dt + 1/2 g dt^2
            + R_k Q_l(phi_k) a_k dt^2
```

其中

```text
J_l(phi) = I
           + (1-cos(theta))/theta^2 Phi
           + (theta-sin(theta))/theta^3 Phi^2

Q_l(phi) = 1/2 I
           + (theta-sin(theta))/theta^3 Phi
           + (theta^2+2cos(theta)-2)/(2 theta^4) Phi^2
```

`Phi = phi^`，`theta = ||phi||`。当角速度为零时，
`J_l = I`、`Q_l = 1/2 I`，严格退化为作者公式。

## 3. 实现细节与范围控制

在目标目录的 `scripts/estimation/inekf.py` 中新增
`SO3_motion_jacobians(phi)`：

- 一次共享计算 `Phi`、`Phi^2` 和 `theta^2`；
- 一般角度使用闭式三角系数；
- `||phi|| < 1e-4` 时使用同一函数的泰勒系数，避免
  `1-cos(theta)`、`theta-sin(theta)` 的浮点消减；
- 没有修改作者已有的 `SO3_left_jacobian`，从而不顺带改变状态注入等其他路径。

导出时逐文件用 Git 对象哈希验证，71/71 与作者提交一致。最终自动测试还会检查：

1. 作者备份仍位于指定提交；
2. 作者跟踪工作树没有修改；
3. 目标目录中除 `scripts/estimation/inekf.py` 外，全部作者原跟踪文件与备份字节一致。

新增支持文件不属于作者原跟踪文件：

- `experiments/compare_pose_update.py`
- `tests/test_pose_update.py`
- `run_task04.ps1`
- `TASK04_README.md`

## 4. 自动验证

命令：

```powershell
Set-Location 'E:\0syncthing\Code\InEKF\imu-uwb-inekf-update-improve'
.\run_task04.ps1 -Mode test
```

结果：`4/4` 通过。

| 测试 | 结果 |
|---|---|
| `J_l/Q_l` 对增广 `9x9` 矩阵指数 | 五组零角、小角、一般角和大角均在 `2e-12` 容差内一致 |
| 恒定 IMU 单步闭式真值 | 改进状态达到机器精度；作者冻结姿态式存在可检测截断误差 |
| 零角速度退化 | 改进/作者状态在 `1e-14` 绝对容差内一致 |
| 原作者代码范围保护 | 通过 |

`compileall` 对核心文件、测试入口和比较入口通过。

## 5. 数据集回放

环境：

```text
D:\miniconda\Scripts\conda.exe run --no-capture-output
  -p D:\conda-envs\imu-uwb-inekf python ...
Python 3.11.15
NumPy 1.26.4
SciPy 1.17.1
Pandas 3.0.3
Matplotlib 3.11.1
```

比较入口对两种公式提供相同：

- 初值和作者原始协方差；
- IMU/UWB 时间戳及调用顺序；
- TDoA 方差 `0.13`；
- 协方差、门限和状态修正代码。

姿态指标使用 `SO(3)` 测地角，yaw 误差做了周期包裹。运行计时只包含回放循环，
但每种公式仅运行一次，不能作为性能基准。

### 5.1 const1 常规轨迹，前 30 秒

输入：

```text
anchor_const1.npz
const1-trial1-tdoa2.csv
```

共同事件：

- 30,267 次 IMU 预测；
- 14,289 次 UWB 更新；
- 前后均拒绝 0 条 UWB。

| 指标 | 作者冻结姿态 | 改进 `J_l/Q_l` | 相对变化 |
|---|---:|---:|---:|
| 位置 RMSE / m | 0.113244377 | 0.113243527 | -0.000750% |
| 速度 RMSE / (m/s) | 0.067806094 | 0.067795289 | -0.015936% |
| 姿态测地 RMSE / rad | 0.053281483 | 0.053245049 | -0.068381% |
| yaw RMSE / rad | 0.051465742 | 0.051427884 | -0.073561% |

最大前后状态差：

- 位置 `1.470208e-4 m`；
- 速度 `1.142179e-3 m/s`。

### 5.2 const4 手动转动，前 60 秒

输入：

```text
anchor_const4.npz
const4-trial7-tdoa2-manual1.csv
```

共同事件：

- 60,531 次 IMU 预测；
- 25,915 次 UWB 更新；
- 前后均拒绝 138 条 UWB。

| 指标 | 作者冻结姿态 | 改进 `J_l/Q_l` | 相对变化 |
|---|---:|---:|---:|
| 位置 RMSE / m | 0.355254510 | 0.355251628 | -0.000811% |
| 速度 RMSE / (m/s) | 0.187816636 | 0.187748699 | -0.036172% |
| 姿态测地 RMSE / rad | 0.103965884 | 0.103938205 | -0.026623% |
| yaw RMSE / rad | 0.093971539 | 0.093941190 | -0.032295% |

最大前后状态差：

- 位置 `4.971900e-4 m`；
- 速度 `1.548359e-3 m/s`。

### 5.3 结果解释

两组四项 RMSE 均为小幅改善，而且更强转动的 const4 具有更大的前后状态差，
与本任务只补偿“采样间隔内姿态继续旋转”的高阶项相符。

改变量很小是合理现象：数据 IMU 约 1 kHz，单步
`||omega dt||` 很小；UWB 又以高频闭环修正状态。不能为了让图看起来更明显而
降低 IMU 频率、扩大步长或改变其他滤波设置。也不能仅凭两组结果声称所有数据上
必然提高精度。

单次回放时间：

- const1：`9.478 s -> 10.578 s`；
- const4：`17.497 s -> 14.137 s`。

方向相反，说明冷启动和系统负载影响明显。本任务没有执行 AB/BA 多轮计时，因此
不宣称加速，也不宣称固定变慢。

## 6. 图片与 PDF 检查

逐图目视检查：

- const1 轨迹为连续闭合环，位置/速度/姿态误差无异常阶跃或发散；
- const4 手动轨迹连续，误差峰值与机动段一致，前后主曲线近乎重合；
- 差异放大图在 `1e-12` 显示下限以上连续变化，无数值爆炸；
- 两组 UWB 拒绝数未变化，改善不是由放宽门限造成。

LaTeX：

- `tex/task04_pose_update_improvement.tex`
- `tex/task04_pose_update_improvement.pdf`
- 5 页；
- XeLaTeX 编译无 overfull、underfull、未定义引用警告；
- 全部 5 页渲染为 PNG 后逐页检查，无裁切、重叠、乱码或不可读公式。

## 7. 用户运行入口

快速测试加 30 秒对比：

```powershell
Set-Location 'E:\0syncthing\Code\InEKF\imu-uwb-inekf-update-improve'
.\run_task04.ps1 -Mode quick
```

完整默认 CSV：

```powershell
.\run_task04.ps1 -Mode full
```

自定义输入示例和输出说明见：

```text
imu-uwb-inekf-update-improve\TASK04_README.md
```

## 8. 交付物

核心：

- `imu-uwb-inekf-update-improve/scripts/estimation/inekf.py`
- `imu-uwb-inekf-update-improve/run_task04.ps1`
- `imu-uwb-inekf-update-improve/TASK04_README.md`

测试与比较：

- `imu-uwb-inekf-update-improve/tests/test_pose_update.py`
- `imu-uwb-inekf-update-improve/experiments/compare_pose_update.py`

结果：

- `imu-uwb-inekf-update-improve/results/task04_const1_30s`
- `imu-uwb-inekf-update-improve/results/task04_const4_manual1_60s`

数学说明：

- `tex/task04_pose_update_improvement.tex`
- `tex/task04_pose_update_improvement.pdf`

## 9. 已知边界

- 改进只对分段常值 IMU 假设下的确定性名义位姿积分精确；
- 没有同步改变协方差离散化或 IMU 噪声积分，因为任务明确禁止修改其他部分；
- 作者原始偏置布局、左雅可比小角符号、坐标一致性和时间调度等问题全部保留；
- 结果仅证明两组代表数据上的合理性，未做全部 79 个 CSV 的统计覆盖；
- 单次墙钟不能当作性能结论。

后续已补充 AB/BA 多轮开销验证，见
`reports/task04_compute_overhead_validation_report_2026-07-27.md`；最终测得当前
Python 实现的局部公式开销约 `+38%--39%`，完整 InEKF 回放开销约
`+5.39%--5.68%`。
