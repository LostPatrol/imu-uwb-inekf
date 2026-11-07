# Task06 状态增广与 TWR 执行报告

> **2026-07-29 补充说明：** 本文记录首轮两方法实现；其中“48 个真实运行、
> 40 行蒙特卡洛、12 项测试”及相应表格已被后续三方法复核取代。普通
> TWR--InEKF 已补入全部真实批测和大初值蒙特卡洛，末 5 s 劣化也已完成
> 因果消融。最新结论与 72/60/84 行结果见
> `reports/task06_tail_diagnosis_and_inekf_baseline_report_2026-07-29.md`，
> 最新数学说明见 `tex/task06_state_augmentation_twr.tex/.pdf`。

- 日期：2026-07-28
- 任务文件：`tasks/06状态增广与TWR.md`
- 实现目录：`imu-uwb-inekf-state-augment`
- 基线目录：`imu-uwb-inekf - backup`
- 结论状态：**核心目标已完成，但理论和真实数据效果均按实际结果限定，不宣称完整系统全局收敛**

## 1. 交付摘要

本任务完成了以下工作：

1. 从论文作者备份 `f3b664e0e8c86b62c6ed2ea2a112261810228e1c` 独立建立
   `imu-uwb-inekf-state-augment`，没有直接修改备份代码。
2. 只修正两项与原论文直接冲突的低级问题：
   - 15 维误差状态和协方差中的偏置顺序统一为
     `[姿态, 速度, 位置, 陀螺偏置, 加速度偏置]`；
   - `SO(3)` 左雅可比小角度分支由错误的
     `I - 0.5*phi^` 修为 `I + 0.5*phi^`，与通用分支和左雅可比定义一致。
3. 将作者 TDoA 量测改为 TWR，并提供普通 TWR--InEKF 参考实现。
4. 实现推荐的平方距离状态增广：
   \[
   q=\frac12p^\top p,\qquad s=p^\top v,\qquad h=v^\top v,\qquad
   \chi=[q,s,h,p^\top,v^\top,1]^\top .
   \]
5. 实现分段常值世界系加速度下的精确闭式状态转移、常数 TWR 观测行、
   平方距离噪声去偏、Joseph 协方差更新、NIS 门限和增广流形残差诊断。
6. 下载并校验 TU Delft 的真实 TWR 数据集，在全部 12 条 TWR 日志上执行
   两种初始条件、两种方法的 48 个批量运行。
7. 执行 20 次持续激励、单锚点、大初值蒙特卡洛，单独检验状态增广的
   大范围数值收敛特性。
8. 生成四张对比图、CSV/JSON/NPZ 原始结果、12 条自动测试、PowerShell
   一键入口、中文 LaTeX/PDF 数学说明及本报告。

总体结果不是“增广法全面优于普通 EKF”：

- 真实数据大初始偏差下，增广法的全程 RMSE 和最大瞬态更稳健；
- 真实数据零初始偏差和末 5 s 稳态指标没有优势；
- 理想化持续激励单锚点试验中，增广法的大初值成功率为 20/20，
  普通直接 TWR EKF 为 0/20；
- 该合成结果不能外推为含姿态、偏置、NLOS 的完整 IMU--UWB 系统已经
  获得 UGES、AGAS 或全局收敛证明。

## 2. 指南与参考文献审计

### 2.1 原论文 TDoA 的结构

Oursland--Mehrabian 论文式 (33)--(38) 的 TDoA 为

\[
h_{BA}(p)=\|p-a_B\|-\|p-a_A\|,
\]

其位置梯度为

\[
\frac{\partial h_{BA}}{\partial p}
=\frac{(p-a_B)^\top}{\|p-a_B\|}
-\frac{(p-a_A)^\top}{\|p-a_A\|}.
\]

论文随后在估计点构造观测雅可比，并以当前状态估计的伴随矩阵变换。
因此量测方向仍显式依赖当前位置估计。把 TDoA 直接换成
\(\rho_i=\|p-a_i\|\) 后，普通 TWR 雅可比

\[
H_i^{\rm dir}
=\left[\frac{(p-a_i)^\top}{\|p-a_i\|}\quad 0_{1\times3}\right]
\]

同样依赖估计点，大初值时可能长期使用错误的线性化方向。

### 2.2 “平方距离加状态增广”的合理部分

对世界系平移动力学

\[
\dot p=v,\qquad \dot v=u
\]

定义 \(q,s,h\) 后有

\[
\dot q=s,\qquad
\dot s=h+p^\top u,\qquad
\dot h=2v^\top u.
\]

因此对已知输入 \(u\)，齐次增广状态满足线性时变系统

\[
\dot\chi=A(u)\chi.
\]

对锚点 \(a_i\)：

\[
\frac12\|p-a_i\|^2
=q-a_i^\top p+\frac12a_i^\top a_i
=H_i\chi,
\]

\[
H_i=
\begin{bmatrix}
1&0&0&-a_i^\top&0_{1\times3}&\frac12a_i^\top a_i
\end{bmatrix}.
\]

这里 \(H_i\) 与估计状态无关，确实消除了普通距离 EKF 的位置相关雅可比。
这一结构与 `2006.02688v2.pdf` 中“线性系统加二次输出”的状态增广例子一致，
也是任务指南中最值得实施的部分。

### 2.3 没有照搬的理论外推

`eq_ob.pdf` 对一般距离观测的分类并没有给出完整有偏惯导系统的全局不变
观测结论；Theodosis 等人的结果也要求输入及其导数满足一致可观测/持续
激励条件。真实 IMU 中用于增广系统的

\[
\hat u=\hat R(a_m-\hat b_a)+g
\]

依赖有误差的姿态和加速度偏置。故本实现采用分阶段结构：

```text
IMU -> InEKF 姿态/偏置核心 -> 世界系加速度 -> 增广 TWR 线性滤波器
```

UWB 修正增广状态中的位置和速度并同步名义平移状态，但不反向注入姿态、
偏置或它们的交叉协方差。这样可以先验证核心线性结构，却不是统一的
群仿射 InEKF。报告和 PDF 均没有把子系统定理偷换成完整系统定理。

## 3. 代码实现

### 3.1 主要文件

| 文件 | 作用 |
|---|---|
| `scripts/estimation/lifted_twr.py` | 增广 TWR KF、普通直接 TWR EKF、普通 TWR--InEKF、分阶段 `LiftedTWRInEKF` |
| `scripts/estimation/twr_experiment.py` | TU Delft 日志读取、IMU 初始化、真实批测、合成试验、蒙特卡洛、指标和绘图 |
| `scripts/estimation/inekf.py` | 仅修复左雅可比小角度符号和偏置顺序 |
| `scripts/estimation/main.py` | 仅同步论文偏置协方差顺序 |
| `tests/test_lifted_twr.py` | 12 条自动测试 |
| `download_dataset.ps1` | 官方数据下载、MD5 检查和解压 |
| `run_task06.ps1` | `download/test/quick/synthetic/full` 一键入口 |
| `README.md` | 任务说明、运行方法、输出和理论限制 |

### 3.2 精确闭式传播

状态顺序为 `[q,s,h,p(3),v(3),1]`。对一个采样间隔内的常值 \(u\)，实现
直接构造 \(\Phi=\exp(A(u)\Delta t)\) 的闭式，不用 Euler 近似。关键项为：

\[
\begin{aligned}
q_{k+1}={}&q_k+s_k\Delta t+\frac12h_k\Delta t^2
+\frac12p_k^\top u\Delta t^2
+\frac12v_k^\top u\Delta t^3
+\frac18\|u\|^2\Delta t^4,\\
s_{k+1}={}&s_k+h_k\Delta t+p_k^\top u\Delta t
+\frac32v_k^\top u\Delta t^2+\frac12\|u\|^2\Delta t^3,\\
h_{k+1}={}&h_k+2v_k^\top u\Delta t+\|u\|^2\Delta t^2,\\
p_{k+1}={}&p_k+v_k\Delta t+\frac12u\Delta t^2,\\
v_{k+1}={}&v_k+u\Delta t.
\end{aligned}
\]

测试将完整 10×10 闭式逐元素与 `scipy.linalg.expm` 对比。

### 3.3 平方距离噪声处理

固定距离偏置修正后的量测记为

\[
r_i=\rho_i^m-b_\rho=\rho_i+n_i,\qquad
n_i\sim\mathcal N(0,\sigma_\rho^2).
\]

直接平方存在 \(\frac12\sigma_\rho^2\) 均值偏置，因此采用

\[
y_i=\frac12(r_i^2-\sigma_\rho^2),\qquad
R_i'\approx r_i^2\sigma_\rho^2+\frac12\sigma_\rho^4.
\]

自动测试用蒙特卡洛检查去偏后的样本均值。这一模型只覆盖高斯加性距离
噪声，不会自动解决 NLOS、锚点相关偏置或错误关联。

### 3.4 数值保护

- 协方差更新使用 Joseph 形式并显式对称化；
- 齐次坐标始终固定为 1，其协方差行列固定为 0；
- 前 64 个更新作为 NIS 门限预热，随后使用阈值 25；
- 无效、拒绝和接受的量测分别计数；
- 输出
  \(q-\frac12p^\top p\)、\(s-p^\top v\)、\(h-v^\top v\)
  三个约束残差；
- 没有用强制投影掩盖增广辅助状态和物理状态的不一致。

代表性真实运行中增广协方差最小特征值为 0，是齐次坐标零协方差带来的
结构性零值，不是负定或 NaN。

## 4. 数据集与实验协议

### 4.1 数据来源

- 数据集：TU Delft, `UWB dataset (Version 1)`
- DOI：<https://doi.org/10.4121/14827680.v1>
- 下载目录：`E:\datasets\tu-delft-uwb-mhe`
- ZIP：`E:\datasets\tu-delft-uwb-mhe\UWB dataset.zip`
- MD5：`7f9f2f81e1b028db8f03ccc4ab0da346`
- 解压目录：`E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset`
- TWR 日志数：12
- 数据内容：8 个锚点、IMU、各锚点 TWR 保持值、OptiTrack XYZ 位置

日志以高频记录保持最近一次各锚点距离值。实现先在原始记录上进行变化
检测，只在值变化时计为一次新的 TWR 事务，避免把保持值反复当作新量测。

### 4.2 初始化与固定参数

- 前最多 1.5 s 的低角速度样本用于重力方向对齐和 IMU 偏置初始化；
- 名义运行的位置初值取第一帧 OptiTrack 位置；
- 大初差运行在上述位置上加 `[6,-6,3]` m；
- 初速度为 0；
- 距离偏置固定为 `-0.18 m`；
- 距离噪声标准差固定为 `0.20 m`；
- 有效世界系加速度过程标准差固定为 `10 m/s²`；
- 普通直接 TWR EKF 与增广 TWR KF 使用完全相同的世界系加速度输入、
  TWR 事件、距离参数和门限。

`10 m/s²` 是较大的有效过程噪声，反映当前简单姿态/偏置初始化下世界系
加速度输入仍很粗糙。该值在开发阶段查看过三角轨迹第一条日志和随机轨迹
第二条日志，二者后来仍被纳入 12 条批测。因此批测是固定参数下的完整
描述性结果，不是严格独立的留出集或盲测。对此不作“泛化已验证”的表述。

真实数据实验还存在以下边界：

- 只有位置真值，不能计算姿态和偏置误差；
- OptiTrack 参考点与 UWB 天线杆臂按零处理；
- 没有逐锚点偏置、遮挡标签或 NLOS 模型；
- OptiTrack 速度由平滑位置数值微分得到，只作辅助指标。

## 5. 自动测试

最终命令：

```powershell
cd E:\0syncthing\Code\InEKF\imu-uwb-inekf-state-augment
.\run_task06.ps1 -Mode test
```

结果：

```text
Ran 12 tests in 0.148s
OK
```

覆盖项：

1. `SO(3)` 左雅可比小角度正号；
2. 陀螺/加速度偏置在预测中的论文顺序；
3. 初始协方差的论文偏置顺序；
4. 闭式状态转移与矩阵指数一致；
5. 无噪声传播保持 \(q,s,h\) 流形；
6. 常数观测行精确恢复半平方距离；
7. 平方高斯距离噪声去偏；
8. 八锚点无噪声位置恢复；
9. 零加速度分阶段传播；
10. 单 TWR 更新的修正方向；
11. 单锚点持续激励下的大初差对比；
12. 真实日志 TWR 事件读取。

此外：

```text
python -m compileall -q scripts tests
```

在指定 Conda 环境中通过。48 行真实批测数值字段全部有限；40 行蒙特卡洛
记录除 20 个预期未收敛的普通 EKF 收敛时间为空外，其余必需字段全部有限；
4 个 JSON 文件均可正常解析。

## 6. 真实数据结果

### 6.1 全部 12 条日志

两种初值 × 两种方法共 48 个运行，全部保持有限并达到所用 0.5 m 末段
收敛判据。

| 初值 | 方法 | 全程位置 RMSE 中位 (m) | 全程位置 RMSE 均值 (m) | 末 5 s RMSE 中位 (m) | 收敛数 | 拒绝量测总数 |
|---|---|---:|---:|---:|---:|---:|
| 真值位置 | 普通 TWR EKF | 0.387111 | 0.396931 | 0.192065 | 12/12 | 66 |
| 真值位置 | 增广 TWR KF | 0.404446 | 0.401514 | 0.334650 | 12/12 | 69 |
| `[6,-6,3]` m 偏移 | 普通 TWR EKF | 0.438736 | 0.477435 | 0.192065 | 12/12 | 218 |
| `[6,-6,3]` m 偏移 | 增广 TWR KF | 0.407519 | 0.408992 | 0.334650 | 12/12 | 69 |

配对比较：

- 真值位置：增广法较低 5/12，普通法较低 7/12；
  `增广 - 普通` 全程 RMSE 中位差 `+0.008780 m`，均值差 `+0.004583 m`。
- 大偏移：增广法较低 8/12，普通法较低 4/12；
  `增广 - 普通` 全程 RMSE 中位差 `-0.016618 m`，均值差 `-0.068442 m`。

合理解释：

- 大初差时，增广法减少了少数日志的很大初始瞬态，因此均值改善较明显；
- 零偏移时没有总体优势；
- 普通直接 EKF 的末 5 s 指标明显更好；
- 拒绝数差异说明普通直接 EKF 在大初差时更易出现高 NIS，但拒绝数本身
  不是定位精度的替代指标。

### 6.2 代表性随机轨迹

日志：
`2021-02-12_09_56_09_twr_random.csv`，时长 65.2 s，
6150 个 IMU 样本，15724 次 TWR 事务，大初值 `[6,-6,3]` m。

| 方法 | 全程位置 RMSE (m) | 末 5 s RMSE (m) | 最大位置误差 (m) | 速度 RMSE (m/s) | 首次进入 0.5 m (s) | 拒绝数 |
|---|---:|---:|---:|---:|---:|---:|
| 普通 TWR EKF | 0.437097 | 0.166681 | 3.967703 | 0.802646 | 0.59 | 8 |
| 增广 TWR KF | 0.417739 | 0.366533 | 1.896865 | 0.603442 | 0.98 | 8 |
| 普通 TWR--InEKF 参考 | 0.387324 | 0.187830 | 3.179399 | 0.425470 | 0.02 | 9 |

代表性运行用时 6.55 s。三条轨迹均连续且有限。增广法降低了全程位置
RMSE、最大位置误差和速度 RMSE，但末 5 s 更差，图上能看到增广法
\(z\) 方向的较大稳态波动。

增广约束：

- RMS：
  `[0.4417 m², 2.0174 m²/s, 1.2536 m²/s²]`
- 末端绝对值：
  `[1.2746 m², 1.4623 m²/s, 1.6619 m²/s²]`

这些残差说明把 \(q,s,h\) 当作独立线性状态后会偏离物理二次流形。
当前实现将它们如实输出，没有靠后处理投影美化结果。

## 7. 持续激励大初值合成结果

使用单锚点原点、20 Hz 距离、0.1 m 噪声、20 s 轨迹：

\[
p(t)=
\begin{bmatrix}
20\cos t-20\\
10\sin(2t)+20\\
-4\cos(4t)
\end{bmatrix},
\]

并由解析式提供准确 \(v(t)\) 和带小过程噪声的世界系 \(u(t)\)。这一实验
故意隔离出 Theodosis 类“线性系统加二次输出、持续激励”的理想设置，
不是对完整有偏 IMU 的仿真替代。

### 7.1 代表性大初值

初始位置 `[35,-25,20]` m、初速度 0：

| 方法 | 全程位置 RMSE (m) | 末 5 s RMSE (m) | 最大误差 (m) | 进入 0.5 m 时间 |
|---|---:|---:|---:|---:|
| 普通 TWR EKF | 147.801 | 204.418 | 225.947 | 未进入 |
| 增广 TWR KF | 19.067 | 0.292 | 69.363 | 12.0 s |

普通 EKF 数值保持有限，但轨迹持续偏离；增广法在约 12 s 后收敛。增广法
19.067 m 的全程 RMSE 没有被改成只报末段，它真实保留了长初始瞬态。

### 7.2 20 次蒙特卡洛

- 随机种子：20260728
- 初始位置：各维独立 `Uniform[-50,50] m`
- 初始速度：各维独立 `Uniform[-10,10] m/s`
- 成功定义：末 5 s 位置 RMSE `< 1 m`

| 方法 | 成功率 | 末 5 s RMSE 中位 (m) | 末 5 s RMSE P95 (m) | 全程 RMSE 中位 (m) |
|---|---:|---:|---:|---:|
| 普通 TWR EKF | 0/20 | 276.887 | 409.731 | 190.810 |
| 增广 TWR KF | 20/20 | 0.381 | 0.423 | 13.493 |

该结果提供了“在已知输入和持续激励条件下，状态增广比估计点线性化具有
更大的数值吸引域”的强数值证据，但不能证明真实级联系统全局收敛。

## 8. 调试过程和失败记录

开发初期真实日志出现了明显的竖直发散。原因不是滤波理论，而是初始化
辅助函数对 `np.asarray` 返回的数组执行了原地归一化，意外把调用者持有的
平均加速度向量也归一化，随后加速度偏置估计失真。修复方式是进入向量
对齐函数时显式复制输入，再原地归一化。修复后重新执行测试和完整数据批测，
此前发散结果没有纳入最终数据。

另一个客观现象是：为了覆盖简单姿态/偏置初始化造成的世界系加速度误差，
真实实验需要 `10 m/s²` 的有效过程噪声。这不是理想参数，也说明当前分阶段
架构尚未解决姿态/偏置与平移的完整一致性。

没有发现程序卡死、无限循环、NaN 传播或运行速度异常。完整批测在合理的
分钟级内完成；代表性 65.2 s 日志为 6.55 s，但该值未做严格硬件基准控制。

## 9. 图像和 PDF 人工检查

已用图像理解逐张检查四张最终结果图：

1. `real_batch_comparison.png`
   - 两种初值的 12 条配对曲线可读；
   - 零偏移时普通法略优、大偏移时增广法降低尖峰的结论与 CSV 一致；
   - 没有裁切、空图或错误图例。
2. `real_representative_large/comparison.png`
   - XY 轨迹连续，无跳到无穷或 NaN；
   - 初始误差、三维误差、速度误差和增广约束均显示；
   - 增广法 \(z\) 稳态较噪，未被隐藏。
3. `synthetic_representative/comparison.png`
   - 普通 EKF 持续发散，增广法约 12 s 后收敛；
   - 对数误差轴和约束残差与数值指标一致；
   - 初始瞬态未裁掉。
4. `synthetic_monte_carlo/monte_carlo.png`
   - 0% 对 100% 成功率和末段对数尺度分布清晰；
   - 1 m 成功门限标出。

LaTeX 最终编译为 7 页 A4 PDF。使用 Poppler 以 135 DPI 逐页渲染，
全部 7 页均人工检查：公式矩阵、表格、中文、路径、引用和四张图均完整，
没有裁切；编译日志没有实际 LaTeX 警告、未定义引用或 overfull/underfull。

## 10. 目录完整性审计

- `imu-uwb-inekf - backup`：
  - HEAD 仍为 `f3b664e0e8c86b62c6ed2ea2a112261810228e1c`；
  - 所有 Git 跟踪文件无修改；
  - 仅有原有未跟踪 `.vscode/`，本任务没有向备份写入源码。
- `imu-uwb-inekf`：
  - 本任务开始前已有若干修改和未跟踪文件；
  - 本任务只做只读状态审计，没有向该目录写入。
- 任务源码修改均限制在 `imu-uwb-inekf-state-augment`；
- 额外写入仅为任务要求的 `E:\datasets` 数据、
  `tex` 文档/PDF、`reports` 报告和根目录 `MEMORY.md`。

## 11. 复现命令

```powershell
cd E:\0syncthing\Code\InEKF\imu-uwb-inekf-state-augment

# 下载并校验真实数据
.\run_task06.ps1 -Mode download

# 12 条自动测试
.\run_task06.ps1 -Mode test

# 代表性真实日志 + 代表性合成试验
.\run_task06.ps1 -Mode quick

# 全部 12 条真实日志 × 两种初值 × 两种方法，并执行 20 次蒙特卡洛
.\run_task06.ps1 -Mode full
```

脚本使用：

```text
D:\miniconda\Scripts\conda.exe run --no-capture-output
  -p D:\conda-envs\imu-uwb-inekf
```

## 12. 交付物

### 源码与入口

- `imu-uwb-inekf-state-augment/scripts/estimation/lifted_twr.py`
- `imu-uwb-inekf-state-augment/scripts/estimation/twr_experiment.py`
- `imu-uwb-inekf-state-augment/scripts/estimation/inekf.py`
- `imu-uwb-inekf-state-augment/scripts/estimation/main.py`
- `imu-uwb-inekf-state-augment/tests/test_lifted_twr.py`
- `imu-uwb-inekf-state-augment/download_dataset.ps1`
- `imu-uwb-inekf-state-augment/run_task06.ps1`
- `imu-uwb-inekf-state-augment/README.md`

### 最终结果

- `imu-uwb-inekf-state-augment/results/task06_full/real_batch_metrics.csv`
- `imu-uwb-inekf-state-augment/results/task06_full/real_batch_summary.json`
- `imu-uwb-inekf-state-augment/results/task06_full/real_batch_comparison.png`
- `imu-uwb-inekf-state-augment/results/task06_full/real_representative_large/`
- `imu-uwb-inekf-state-augment/results/task06_full/synthetic_representative/`
- `imu-uwb-inekf-state-augment/results/task06_full/synthetic_monte_carlo/`

### 数学文档和报告

- `tex/task06_state_augmentation_twr.tex`
- `tex/task06_state_augmentation_twr.pdf`
- `reports/task06_state_augmentation_twr_report_2026-07-28.md`

## 13. 最终评价

任务要求的“将 TDoA 换成 TWR，并通过 \(q,s,h\) 增广把平方距离量测并入
线性状态模型”已经实质完成，且有解析公式、单元测试、真实数据批测、
大初值合成验证、可视化和一键复现入口。

目前最可信的结论是：

> 状态增广在理想持续激励模型中显著扩大了大初值数值收敛域；在真实
> 多锚点数据的大初始位置偏差下改善了部分初始瞬态和全程平均误差，但
> 尚未改善稳态尾段，也尚未形成姿态、偏置、平移和 UWB 的统一全局
> 收敛 InEKF。

后续若继续改进，应优先研究带一致性约束的 \(q,s,h\) 处理、UWB 对姿态/
偏置的交叉协方差反馈，以及逐锚点/NLOS 距离误差模型；这些均属于下一阶段，
本任务未越界实现。
