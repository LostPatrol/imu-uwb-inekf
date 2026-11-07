# IMU–UWB InEKF 原作者算法纯计算加速工程

本目录用于回答一个单独的问题：**不改变论文作者的 InEKF 算法、参数和量测决定，
仅优化 Python 执行方式，能够加速多少？**

它不是当前算法改进主线，也不包含主工程中的偏置顺序修正、右不变协方差初始化修复、
坐标一致 NIS、全圆 yaw 先验或 ZARU。需要这些算法修复时应使用
`../imu-uwb-inekf`；需要最小修正版时使用 `../imu-uwb-inekf-fix`。

## 基线与不可变约束

本目录由只读原作者备份 `../imu-uwb-inekf - backup` 复制，基线提交为：

```text
f3b664e0e8c86b62c6ed2ea2a112261810228e1c
```

作者核心文件 `scripts/estimation/inekf.py` 保持字节级不变，其 SHA-256 为：

```text
97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0
```

公平比较入口会在运行前校验该哈希；如果作者文件被改动，基准测试会拒绝继续。

加速版有意保留作者的：

- 初始状态、初始协方差、零偏置和 `[b_a,b_g]` 名义排列；
- IMU/UWB 噪声、偏置随机游走、`Q*dt²` 离散语义；
- 地面高度截断和速度清零；
- 原始 UWB Jacobian、预门限顺序、NIS=25 门限；
- Joseph 协方差更新；
- 每一次 IMU 预测、每一条 UWB 处理和每一个拒绝决定。

因此本目录也有意继承作者版本已有的坐标/门限与偏置语义问题，不能作为已修复的生产
算法版本。

## 相对原作者版本改了什么

算法方程不变，只重写了计算执行层：

1. 将浮点时间反复查找改成预计算整数事件计划；
2. 使用连续、固定尺寸的 float64 输入和输出缓冲区；
3. 把事件循环、IMU 预测、15×15 协方差传播、UWB 门限/更新编译为一个 Numba
   `nopython` 核；
4. 利用作者传播矩阵 \(A^4=0\) 的结构，用精确三次多项式计算
   \(\exp(A\Delta t)\)；
5. 直接展开作者的 \(H_l\operatorname{Ad}_{\hat X^{-1}}\)，避免反复构造中间增广矩阵；
6. 批量测试支持文件级多进程、单进程单线程、CPU 亲和性、心跳日志和断点续跑；
7. 计时模式可不保存未被指标使用的逐事件协方差历史，但滤波内部仍在每个事件传播
   完整 15×15 协方差；等价回归会另行保存并核对完整历史。

这些是方程等价的执行优化，不是定位算法创新。

## 新增文件

| 路径 | 作用 |
|---|---|
| `scripts/estimation/accelerated_replay.py` | 作者方程的固定尺寸 Numba 等价执行核 |
| `experiments/benchmark_author_original_vs_acc.py` | 单文件作者版/加速版公平比较 |
| `experiments/benchmark_all_constellations.py` | 四组代表文件或全部 79 个 CSV 的批量比较 |
| `tests/test_acceleration.py` | 状态、旋转、协方差、偏置、NIS 和拒绝决定等价回归 |
| `results/` | 指标、轨迹/误差图、NIS 图、日志和全量汇总 |

`conda/conda-env.yaml` 仅增加运行编译核所需的 Numba 依赖。

## 效果

### 全部 79 个 UTIL CSV 完整回放

全数据集实际测试结果：

| 项目 | 结果 |
|---|---:|
| 成功/失败 | 79/79，失败 0 |
| 总事件数 | 13,548,996 |
| IMU 预测数 | 9,984,576 |
| UWB 处理数 | 3,564,341 |
| 单文件加速比中位数 | 38.9794× |
| 单文件加速比范围 | 31.0226×–47.9746× |
| 作者/加速中位耗时之和 | 2798.184/73.485 s |
| 两边 UWB 拒绝数 | 均为 17,178 |
| 拒绝决定变化 | 0 |

最大旋转/速度/位置/偏置/NIS 数值差分别为
`3.33e-14 / 4.84e-14 / 3.80e-14 / 9.85e-18 / 8.27e-9`，最大 RMSE
绝对差不超过 `6.05e-15`。

结论是：**全数据集约 31–48 倍、中位约 39 倍，并且没有改变作者算法的定位效果。**
这不能表述为定位精度提升。

const4 的部分原始算法结果明显弱于其他星座配置，最大位置 RMSE 0.8882 m、最大 yaw
RMSE 10.6238°；作者版和加速版曲线逐点重合，说明这是原算法/数据组合的既有现象，
不是加速退化。

### 计时解释

全量 20 进程批处理的端到端墙钟约 11.41 min，它适合快速生成全部曲线和筛查异常，
但属于负载下的吞吐测试。论文中需要报告单核绝对时间时，应在空闲机器上使用
`--jobs 1` 复测代表文件。

Windows 下当前 MKL/LLVM OpenMP 组合曾触发原生 `0xC06D007F` 异常。入口在导入
NumPy/Numba 之前固定 `MKL_THREADING_LAYER=SEQUENTIAL`，既规避冲突，也保持双方
单核计时公平。

## 环境与快速复现

- Python：`D:\conda-envs\imu-uwb-inekf\python.exe`
- 数据集：`E:\datasets\dataset`

单文件 30 s 快速比较：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_author_original_vs_acc.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --max-duration 30 --warmup-duration 10 --repeats 4 `
  --output-dir results\readme_author_vs_acc
```

四个 constellation 的代表文件完整回放：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_all_constellations.py `
  --repeats 2 --warmup-duration 10 `
  --output-dir results\readme_all_constellations
```

全部 79 个文件；只在确实需要全量结果时运行：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_all_constellations.py `
  --all-files --jobs 20 --cpu-list 0-19 `
  --repeats 2 --warmup-duration 10 --resume `
  --output-dir results\full_all_79_parallel
```

运行等价回归：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" tests\test_acceleration.py
```

## 输出与检查

单 case 目录包含：

- `metrics.json`：计时、精度、事件计数、等价误差和源码哈希；
- `comparison.png`：轨迹及误差对比；
- `nis_and_rejections.png`：NIS 与累计拒绝数；
- 批量模式下的 `runner_stdout.log`、`runner_stderr.log`。

全量汇总位于 `results/full_all_79_parallel`：

- `summary.csv`、`summary.json`、`summary.png`；
- `FINAL_CONCLUSION.md`；
- 79 个 case 的完整指标、两类图和日志。

上述 79 组结果已检查：产物缺失 0、错误日志 0，曲线无加速版新增的跳变或发散。

## 限制

- 这里证明的是“计算实现等价”，没有修复作者算法本身的问题。
- 速度数字依赖当前 Windows、CPU、Numba、NumPy/MKL 和单线程设置，不是目标飞控
  的 WCET。
- Numba 首次运行有 JIT/缓存加载成本，正式计时把它单独记录；部署时应先预热。
- 如果修改作者核心文件或加速方程，必须重新执行完整等价回归，不能继续引用现有
  79 文件结论。

## 详细报告

- `../reports/author_original_compute_only_acceleration_report_2026-07-24.md`
- `../reports/full_all_79_author_vs_acc_final_conclusion_2026-07-24.md`
- `results/full_all_79_parallel/FINAL_CONCLUSION.md`
- `../tex/author_original_compute_only_acceleration.tex`
- `../tex/author_original_compute_only_acceleration.pdf`
