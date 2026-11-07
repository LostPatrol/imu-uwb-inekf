# IMU–UWB InEKF 改进主工程

本目录是当前的**完整改进与研究主线**。它以论文作者公开提交
`f3b664e0e8c86b62c6ed2ea2a112261810228e1c` 为基线，仍以
“不变扩展卡尔曼滤波（InEKF）融合 IMU 与 UWB TDoA 定位”为核心，没有改成其他
定位框架。

只读原作者版本位于 `../imu-uwb-inekf - backup`。另外两个子工程的定位是：

| 目录 | 用途 | 是否改变作者算法 |
|---|---|---|
| `../imu-uwb-inekf-fix` | 最小论文一致性修正与偏置参数实验 | 是，但改动范围小 |
| `../imu-uwb-inekf-acc` | 隔离 Python 计算开销的等价加速 | 否，目标是逐点等价 |
| 本目录 | 坐标一致性、IMU 静止约束、结构化优化和编译核 | 是，当前完整版本 |

原论文对应的[作者代码与引用信息](https://github.com/oursland/imu-uwb-inekf)供查阅。

## 相对原作者版本的主要改动

### 1. 修复状态与李群公式的一致性

- 将名义偏置、误差状态、初始协方差和过程噪声统一为
  \([\theta,v,p,b_g,b_a]\)，消除原代码中 `b_g`、`b_a` 块含义不一致的问题。
- 修复 `SO(3)` 左雅可比小角分支的符号，并加入近零连续级数、近 \(\pi\) 稳健
  `Log`、测地姿态误差和 SLERP 真值对齐。
- 保留完整 15 维误差状态和右不变误差定义，不以降维或冻结偏置换取速度。

### 2. 修复右不变协方差与 UWB 门限的坐标不一致

原作者代码用左坐标测量雅可比和右不变协方差先计算门限，之后才转换雅可比，导致
NIS 和拒绝决定依赖世界坐标原点。本工程改为：

\[
P_r=\operatorname{Ad}_{\hat X}P_l\operatorname{Ad}_{\hat X}^{\mathsf T},
\qquad
H_r=H_l\operatorname{Ad}_{\hat X^{-1}},
\]

\[
S=H_rP_rH_r^{\mathsf T}+R,\qquad
\mathrm{NIS}=z^2/S.
\]

也就是先把先验协方差和量测雅可比放到一致的右不变误差坐标，再执行唯一一次
NIS 门限。全圆未知航向的默认先验标准差由 1 rad 调整为 2 rad，避免对
180° 初始航向误差过度自信；如果已有更准确先验，可以显式覆盖该值。

### 3. 增加低开销的开机静止期 IMU 信息融合

- 检测开机后的首次连续静止段，离开该段后永久关闭，避免把后续悬停误判为静止。
- 将陀螺和加速度计的可观测性分开：静止角速度可直接估计 \(b_g\)，加计偏置候选
  仍经过一致性门控。
- 以 10 Hz 限频执行零角速度更新（ZARU），不加入曾导致大航向误差退化的零速度行。
- 每个 IMU 区间只传播一次；严格异步子步、中点积分和 Van Loan 离散化保留在
  `verified` 研究变体，不作为低开销默认路径。

### 4. 采用方程等价的结构化数值计算

生产路径在不放宽 UWB 门限、不丢测量的前提下加入：

- 利用冻结传播矩阵 \(A^4=0\)，精确计算
  \(\Phi=I+A\Delta t+\frac12(A\Delta t)^2+\frac16(A\Delta t)^3\)；
- 利用过程噪声的对角与零列结构，减少稠密零块乘法；
- 用标量 rank-one 公式替代等价的 UWB Joseph 更新；
- 直接展开当前成熟路径的 \(H_l\operatorname{Ad}_{\hat X^{-1}}\)；
- 用共享三角函数的 \(\Gamma_1/\Gamma_2\) 闭式公式传播常角速度、常比力 IMU。

这些改动是数学等价或更精确的结构化计算，不是协方差降频、UWB 降采样或缓存
Kalman 增益等近似。

### 5. 增加完整固定尺寸 Numba 编译核

`scripts/estimation/compiled_replay.py` 将事件循环、IMU 名义/协方差传播、UWB
门限与更新、开机静止更新以及输出缓冲区放入一个 `nopython` 固定尺寸核中。

编译路径仍使用 float64、完整 15×15 协方差、每个 IMU 预测和每条 UWB 量测。
首次 JIT 或缓存加载耗时单独报告，不混入稳定态回放。当前没有启用 float32、
协方差降频、9/12 维状态或 UWB 丢弃。

## 已验证的效果

### 坐标一致性和大初始航向

在 const3/const4 六条 manual 轨迹、0°–180°、共 60 次 30 s 配对回放中：

- 修复前 const3 在 120°–180° 时位置 RMSE 为 36.8–560.8 m，UWB 拒绝率最高
  99.49%；
- 修复后全部条件的位置 RMSE 为 0.208–0.431 m，拒绝率为 0.249%–2.17%；
- 同时平移轨迹和基站后，修复版的 NIS、拒绝决定与 RMSE 保持不变，消除了不应存在
  的世界原点依赖。

代价是：在真实初始 yaw 本来接近 0° 的场景中，2 rad 保守先验会带来更明显的初始
姿态瞬态，不能声称所有指标都优于作者窄先验。

### 低开销算法与结构化计算

代表数据结果：

| 对比 | 数据与设置 | 主要结果 |
|---|---|---|
| Legacy → Proposed | const1 前 30 s | 位置 RMSE 0.110765 → 0.107131 m；用时 9.184 → 9.053 s |
| Legacy → Proposed | const4 前 60 s，yaw 90° | 位置 0.356135 → 0.354669 m；速度 0.192817 → 0.186921 m/s；用时 17.824 → 17.677 s |
| 结构化优化前 → 后 | 三组代表数据 | 端到端加速 1.292×–1.313×；UWB 拒绝数不变；RMSE 基本不变 |
| Python Proposed → 编译核 | 三组代表数据 | 稳定态加速 16.18×–16.92×；四项 RMSE 相对差小于 \(3\times10^{-10}\%\)；拒绝决策变化 0 |

### 与原作者完整实现的直接比较

const1 前 30 s、公平关闭双方历史记录后的单滤波器计时中，原作者版中位回放
7.998 s，当前 `task03_compiled` 为 0.235 s，约 33.99×。该系统级比较同时包含
算法修复、先验变化和执行优化，因此不能把全部差异归因于某一个公式。

同一数据段中，当前版的位置/速度 RMSE 改善 7.96%/5.03%，但姿态/yaw RMSE
因更保守的全圆 yaw 先验而变差约 188%/197%；轨迹连续且约 24 s 后持续进入 5°。
所以准确结论是“更稳健地覆盖未知大航向并显著加速”，不是“所有场景、所有精度指标
全面更好”。

## 目录中新增或重点修改的文件

| 路径 | 作用 |
|---|---|
| `scripts/estimation/inekf.py` | InEKF 状态、传播、修正、结构化核与配置 |
| `scripts/estimation/pipeline.py` | 事件计划、变体选择、回放与指标接口 |
| `scripts/estimation/compiled_replay.py` | 完整 Numba 固定尺寸回放核 |
| `scripts/estimation/main.py` | Windows 友好的命令行、绘图与变体入口 |
| `experiments/` | 坐标修复、优化、作者版公平比较等冻结实验 |
| `tests/test_inekf.py` | 李群、协方差、Jacobian、数值等价和编译核回归 |
| `results/` | 指标 JSON/CSV 与已检查的轨迹、误差、NIS 图 |

## 环境与数据

- Python：先从仓库根使用 `conda env create -f imu-uwb-inekf/conda/conda-env.yaml` 建立环境，
  再执行 `conda activate imu-uwb-inekf`。
- 数据集根目录：`E:\datasets\dataset`
- Windows 下建议将 Matplotlib 后端设为 `Agg`，批处理时避免打开交互窗口。

## 运行示例

快速运行当前低开销 Python 版本：

```powershell
$env:MPLBACKEND = "Agg"
python scripts\estimation\main.py `
  -i "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
     "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --variant proposed --yaw-deg 0 --max-duration 30 `
  --output-dir results\readme_proposed_const1
```

运行当前编译版本：

```powershell
python scripts\estimation\main.py `
  -i "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
     "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --variant task03_compiled --yaw-deg 0 --max-duration 30 `
  --output-dir results\readme_compiled_const1
```

直接比较原作者版本与当前最新版本：

```powershell
python `
  experiments\benchmark_author_original_vs_latest.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --max-duration 30 --warmup-duration 10 --repeats 2 `
  --output-dir results\readme_author_vs_latest
```

运行自动测试：

```powershell
python tests\test_inekf.py
```

## TWR 恢复共同协议

`experiments/roadmap_stage1` 比较同 15 维状态的 InEKF／ESKF、捕获初始化、
中断重置和完整双滤波器。入口共享离散 IMU 模型、物理先验、逐锚标定和鲁棒门控，
保存冻结协议、完整轨迹及恢复指标。
需要同级目录 `../imu-uwb-state-augment-deep-coupled-two-filters` 的捕获与监督器源码，
以及 `E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset` 数据。

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python experiments\roadmap_stage1\run.py `
  --output-dir results\roadmap_stage1_new
python experiments\roadmap_stage1\analyze.py `
  --output-dir results\roadmap_stage1_new
```

首次检查可加 `--suite smoke --duration 8`，正式结果应使用完整日志。
默认测试矩阵含初始位置扰动、yaw 扰动和测距中断；位置初值使用受控真值偏移，
因此结果属于恢复域实验。数学定义见 `../tex/roadmap_stage1_20261104.tex`。

## 结果解释边界

- 当前速度是指定 Windows/Conda/CPU 环境下的离线稳定态结果，不是飞控 MCU 的
  WCET；首次 JIT 需要预热。
- UTIL 没有可靠 IMU 偏置真值，不能从这些结果伪造偏置 RMSE 或完整 NEES。
- const3 修复后的 yaw 仍可能约为 10°–15°，单标签 TDoA 的航向可观测性限制没有
  被编译优化解决。
- `verified` 用于研究严格异步传播、Van Loan、完整杆臂 Jacobian 和 covariance
  reset；其小幅精度收益不足以抵消约三倍回放开销，因此不是默认生产结论。

## 详细报告

- `../reports/task01_execution_report_2026-07-22.md`
- `../reports/coordinate_consistency_fix_report_2026-07-23.md`
- `../reports/task02_algorithm_optimization_report_2026-07-23.md`
- `../reports/task03_compiled_optimization_report_2026-07-24.md`
- `../reports/fair_author_original_vs_latest_report_2026-07-24.md`

对应的 LaTeX 数学说明和 PDF 位于 `../tex`。

## 数据集与引用

本工程使用 University of Toronto UTIL UWB 数据集。数据说明见
[UTIL 数据集网站](https://utiasdsl.github.io/util-uwb-dataset/)，数据集论文见
[arXiv:2203.14471](https://arxiv.org/abs/2203.14471)。使用本工程或数据集发表结果时，
请同时引用原 InEKF 论文和 UTIL 数据集论文。
