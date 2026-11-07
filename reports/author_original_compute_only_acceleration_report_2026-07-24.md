# 原作者 InEKF 与纯计算加速版公平对比

日期：2026-07-24  
原作者只读备份：`E:\0syncthing\Code\InEKF\imu-uwb-inekf - backup`  
纯加速副本：`E:\0syncthing\Code\InEKF\imu-uwb-inekf-acc`

## 1. 任务结论

已从原作者 backup 逐文件复制出独立目录 `imu-uwb-inekf-acc`，没有复制 `.git`
和 `.vscode` 元数据。复制时共校验 78 个工作文件，SHA-256 无不一致。

核心原作者文件保持字节级不变：

- backup `scripts/estimation/inekf.py`
- acc `scripts/estimation/inekf.py`
- 两者 SHA-256：
  `97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0`

新增：

- `scripts/estimation/accelerated_replay.py`：原作者方程的固定尺寸 Numba 编译回放核；
- `experiments/benchmark_author_original_vs_acc.py`：公平速度、曲线、NIS 和拒绝对比；
- `tests/test_acceleration.py`：源码哈希和逐状态/协方差等价测试。

入口启动时会再次比较 backup 和 acc 内的 `inekf.py` 哈希；若不一致，立即终止，
防止算法或调参改动混入“纯加速”实验。

## 2. 明确保留的原作者设置

两边完全相同地使用：

- 相同的 Vicon 初始位置和姿态，加相同的 `--yaw-deg`；
- yaw 初始标准差 `1.0 rad`；
- roll/pitch 初始标准差 `0.01 rad`；
- 速度初始标准差 `0.01 m/s`；
- 位置初始标准差 `0.1 m`；
- 加速度计/陀螺偏置标准差 `1e-4/1e-6`；
- 6 维偏置初值全零；
- 原作者的名义偏置布局和误差状态偏置布局；
- TDoA 方差 `0.13`；
- Mahalanobis 门限 5（NIS 门限 25）；
- 原始地面高度/速度钳位；
- 原始左坐标预门限；
- 原始右不变量测模型、Joseph 协方差更新和状态注入；
- 每个 IMU 预测和每个 UWB 更新。

特别说明：本目录有意保留原作者代码中已经识别的偏置排列和协方差/门限坐标问题。
它用于隔离计算开销，不替代当前经过修复的生产算法。

## 3. 只改变的计算执行方式

1. 将合并后的 IMU/UWB 时间轴预计算为固定事件数组和整数起止下标；
2. 使用连续 float64 固定尺寸数组，避免 Python 对象和临时列表；
3. 事件循环、IMU 预测、15×15 协方差传播、UWB 门限与 Joseph 更新进入一个
   Numba `nopython` 核；
4. 原作者当前误差动力学严格满足 \(A^4=0\)，因此

   \[
   e^{A\Delta t}
   =
   I+A\Delta t+\frac12(A\Delta t)^2+\frac16(A\Delta t)^3
   \]

   与通用 `scipy.linalg.expm` 数学等价；
5. 直接展开 `H_left @ Ad(X_hat^-1)`，不构造通用 15×15 Adjoint；
6. 正式计时关闭两边的历史记录；计时外各运行一次完整历史用于效果曲线。

没有使用降维、float32、UWB 丢弃、协方差降频、偏置冻结或缓存过期 Kalman 增益。

## 4. 公平计时边界

- 数据、初始状态、初始协方差、偏置、噪声和事件计划相同；
- 每轮均断言 30,267 次 IMU 预测和 14,289 次 UWB 更新；
- 数值库单线程；
- 排除数据加载、对象构造、JIT 编译/缓存加载、历史、指标和绘图；
- 四轮 AB/BA 交替，不挑选有利单次结果；
- 额外完整历史回放不进入正式速度统计；
- JSON 记录事件计划哈希、两份原作者源码哈希、编译核哈希和每轮原始时间。

## 5. const1 前 30 秒正式结果

命令参数：

`--max-duration 30 --warmup-duration 10 --repeats 4 --yaw-deg 0`

| 指标 | 作者原始版 | 纯计算加速版 | 结果 |
|---|---:|---:|---:|
| 回放中位数 | 6.811990 s | 0.216905 s | 31.4054x |
| MAD | 0.062050 s | 0.001329 s | 均较稳定 |
| 最小--最大 | 6.734570--6.884360 s | 0.214860--0.230287 s | 无顺序翻转 |
| 耗时下降 | — | 96.8158% | 不含 JIT |

逐轮配对加速：

`29.8051x / 31.6495x / 31.2553x / 31.3440x`

配对中位数 `31.2996x`，MAD `0.1971x`。首次无缓存编译在本轮开发中实测
约 `8.240 s`；正式运行已有缓存时的加载/预热为 `0.128 s`，两者都没有计入稳定
回放。

## 6. 数值等价性

| 对比项 | 最大绝对差 |
|---|---:|
| 旋转矩阵元素 | \(6.77\times10^{-15}\) |
| 速度 | \(5.54\times10^{-15}\) m/s |
| 位置 | \(3.33\times10^{-15}\) m |
| 偏置 | \(4.98\times10^{-19}\) |
| NIS | \(1.29\times10^{-14}\) |
| UWB 拒绝决策变化 | 0 |

四项 RMSE 在显示精度内完全相同：

| 指标 | 作者原始版 | 纯计算加速版 |
|---|---:|---:|
| 位置 RMSE | 0.1162660522 m | 0.1162660522 m |
| 速度 RMSE | 0.0767476310 m/s | 0.0767476310 m/s |
| 姿态测地 RMSE | 0.0547138522 rad | 0.0547138522 rad |
| yaw RMSE | 0.0529485209 rad | 0.0529485209 rad |
| UWB 拒绝 | 0/14,289 | 0/14,289 |

`comparison.png` 已目视检查：XY、位置、速度、姿态和 yaw 曲线逐点重合，轨迹连续，
无新增跳变或发散。`nis_and_rejections.png` 中 NIS 曲线重合，两边拒绝均为零。

## 7. VSCode PowerShell 命令

先进入新目录：

```powershell
cd "E:\0syncthing\Code\InEKF\imu-uwb-inekf-acc"
```

建议先运行 30 秒正式对比：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_author_original_vs_acc.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --max-duration 30 --warmup-duration 10 --repeats 4 `
  --output-dir results\manual_author_original_vs_acc
```

快速检查只需将重复次数改为 2。只有确实需要整段航迹时才删除
`--max-duration 30`；`--warmup-duration` 只控制预热数据，不限制正式回放长度。

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_author_original_vs_acc.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --max-duration 30 --warmup-duration 10 --repeats 2 `
  --output-dir results\manual_author_original_vs_acc_30s
```

查看结果：

```powershell
Invoke-Item .\results\manual_author_original_vs_acc\comparison.png
Invoke-Item .\results\manual_author_original_vs_acc\nis_and_rejections.png
Get-Content -Raw -Encoding UTF8 `
  .\results\manual_author_original_vs_acc\metrics.json
```

## 8. 适用边界与代价

- 31.4x 是当前 Windows/Conda/单线程、30 秒离线回放的直接结果，不是机载 WCET；
- 主要收益来自消除 Python 小矩阵热循环，若与优化 C/C++ 比较，倍数不会自动保留；
- 首次 JIT 有数秒延迟，实飞必须启动预热、使用缓存或改为提前编译；
- 编译核与原作者 Python 源码形成双实现，后续必须维持等价测试；
- 本加速版有意继承原作者算法缺陷，只用于回答“同一作者算法能加速多少”，不应取代
  当前修复后的 `imu-uwb-inekf` 生产版本。

正式结果：

`imu-uwb-inekf-acc/results/fair_author_original_vs_acc_const1_30s`

数学与执行边界说明：

- `tex/author_original_compute_only_acceleration.tex`
- `tex/author_original_compute_only_acceleration.pdf`（6 页，已逐页渲染检查）

## 9. 2026-07-24 Windows MKL 兼容修复

用户手动运行入口时，Python 在约 5 秒后无 Python traceback 地直接退出。使用
`-X faulthandler` 复现得到 Windows 原生异常 `0xC06D007F`，位置在原作者
`predict()` 的 3×3 矩阵乘法。最小 10 万次矩阵乘法循环可独立复现，说明不是
输入路径、参数或滤波数据错误。

当前环境同时包含 MKL 2026.1 与 LLVM OpenMP/Numba；MKL 的线程执行层在重复
小矩阵运算中触发 DLL/运行库异常。对比本来就是单核测试，因此入口现在会在
导入 NumPy/Numba 前设置 `MKL_THREADING_LAYER=SEQUENTIAL`。这既避免冲突，也
明确固定单核执行；没有修改作者滤波方程或加速核。入口还会立即打印 `setup`
信息，避免数据准备阶段看似“无响应”。

修复后在未手动设置任何 MKL 环境变量时完成 const1 前 30 秒、2 轮回归：

- 作者原始版回放中位数：9.477928 s；
- 纯计算加速版回放中位数：0.237608 s；
- ratio-of-medians：39.8889x；
- 两边位置、速度、姿态、yaw RMSE 一致，UWB 拒绝均为 0/14,289；
- 2/2 等价回归通过，作者副本 SHA-256 仍为
  `97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0`；
- `comparison.png` 与 `nis_and_rejections.png` 已目检，曲线重合且无发散。

该 39.89x 数值使用新的 MKL 顺序层，不应与此前 31.41x 的运行库条件混为同一
统计批次；它证明修复后的入口稳定且算法等价，不代表算法本身再次获得额外加速。

## 10. const1--const4 完整代表文件批量结果

新增入口：

`imu-uwb-inekf-acc/experiments/benchmark_all_constellations.py`

默认选择每个 constellation 的同类代表文件并进行完整回放：

- const1：`const1-trial1-tdoa2.csv`
- const2：`const2-trial1-tdoa2.csv`
- const3：`const3-trial1-tdoa2.csv`
- const4：`const4-trial1-tdoa2-traj1.csv`

这不是对所有试验文件的穷举；整个数据集共有 79 个 CSV。增加 `--all-files`
才会运行全部文件。单进程顺序执行预计约 2.77 小时；第 11 节给出并行入口与
约 12--20 分钟的实测外推。默认四文件完整回放命令为：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_all_constellations.py `
  --repeats 2 --warmup-duration 10 `
  --output-dir results\full_all_constellations
```

四组均不传 `--max-duration`，正式重复 2 轮：

| 数据 | 事件数 | 作者原始版/s | 加速版/s | 加速比 |
|---|---:|---:|---:|---:|
| const1 | 209,426 | 45.645867 | 1.121883 | 40.6868x |
| const2 | 183,197 | 39.344386 | 0.970576 | 40.5372x |
| const3 | 196,615 | 43.110833 | 1.031722 | 41.7853x |
| const4 | 186,075 | 40.473703 | 0.985142 | 41.0841x |

| 数据 | 位置 RMSE/m | 速度 RMSE/(m/s) | 姿态 RMSE/deg | yaw RMSE/deg |
|---|---:|---:|---:|---:|
| const1 | 0.108003 | 0.060901 | 3.6570 | 3.5695 |
| const2 | 0.103045 | 0.060616 | 1.6560 | 1.2562 |
| const3 | 0.219516 | 0.167438 | 2.2330 | 2.0697 |
| const4 | 0.200157 | 0.088926 | 3.2633 | 2.0724 |

上述效果指标在作者版和加速版之间显示精度内相同。四组拒绝决策变化均为 0；
const1--const3 均为 0 次拒绝，const4 两边均拒绝同一条 UWB。最大旋转、位置和
NIS 差分别不超过 `2.75e-14`、`1.16e-14` 和 `2.24e-13`。

`summary.png` 以及四组 `comparison.png` 已目检：曲线逐点重合，无新增跳变或
发散。const3 本身的位置/速度误差峰值较大，但两种实现完全一致，不能归因于
计算加速。

批量入口支持 `--resume` 复用已经成功生成的 `metrics.json`；最终输出为：

- `results/full_all_constellations/summary.csv`
- `results/full_all_constellations/summary.json`
- `results/full_all_constellations/summary.png`
- `results/full_all_constellations/const1` 至 `const4` 的逐条曲线与指标。

## 11. 20 核并行穷举入口与验证

### 11.1 并行方式和公平性

新增的是“文件之间”的进程级并行，而不是把同一条滤波时间序列拆成多个线程。
每个子进程仍按时间顺序完整执行一条 CSV，并固定
`MKL_THREADING_LAYER=SEQUENTIAL`；在当前 20 核/20 线程处理器上，每个子进程
绑定一个逻辑 CPU。某个核完成一条后立即领取下一条，因此长短文件不要求静态
平均分配。

若用 \(J\) 个工作进程，第 \(j\) 个进程领取的文件集合为
\(\mathcal A_j\)，则批量墙钟时间近似为

\[
T_{\rm batch}(J)
\approx
\max_{1\le j\le J}\sum_{i\in\mathcal A_j}T_i
+T_{\rm overhead}.
\]

这只缩短“79 条离线试验全部跑完”的总等待时间；单条飞行的滤波延迟仍由单核
回放决定，不会把并行批处理收益冒充成机载单滤波器加速。

为避免 20 个完整历史同时占用过多内存，计时外、用于曲线和指标的加速回放不再
保存未被任何输出消费的逐事件 \(15\times15\) 协方差历史。约 20 万事件时，

\[
M_{\rm covariance\ history}
=200000\times15^2\times8
\approx360\ {\rm MB/文件}.
\]

滤波器仍在每个事件传播完整的 \(15\times15\) 协方差；默认测试模式也仍保存并
逐项核对完整协方差历史。因此这是输出内存优化，不是降维、降频或算法简化。

### 11.2 完整文件并发压力测试

在 CPU 0--3 上同时完整运行四个代表文件，每条正式重复 2 轮，结果如下：

| 数据 | 子进程墙钟/s | 作者中位数/s | 加速版中位数/s | 加速比 |
|---|---:|---:|---:|---:|
| const1 | 105.545 | 31.4105 | 0.9754 | 32.20x |
| const2 | 95.102 | 28.2619 | 0.8590 | 32.90x |
| const3 | 142.348 | 42.9766 | 1.0582 | 40.61x |
| const4 | 132.765 | 39.8653 | 0.9996 | 39.88x |

四条子进程墙钟之和为 475.760 s，批量墙钟为 142.353 s，调度收益
为 3.342x，相对四核理想值的效率为 83.55%，吞吐为 101.16 条/小时。
测试中父进程加四个子进程峰值工作集约 1.06 GB；按 20 进程线性外推约
5.3 GB，低于测试时约 9.25 GB 的可用内存。

若把四核实测吞吐理想外推到 20 核，79 条约需 9.37 分钟。考虑本机包含性能不同
的核心、全核持续功耗、绘图与磁盘开销，建议按 **12--20 分钟** 预留，而不是
把 9.37 分钟当作承诺。

四条完整结果的四项 RMSE 与先前单独批次完全相同，UWB 拒绝决策变化均为 0，
最大 NIS 差不超过 `2.24e-13`。并行汇总图和 const3 对比图已目检：两种实现
曲线重合；const3 既有误差峰值未因并发或加速改变。

### 11.3 20 进程压力与首事件 UWB 边界修复

另以 20 条 CSV 的前 2 秒同时启动 20 个固定核子进程，最终 20/20 完成、失败
为 0。压力测试首次发现 `const1-trial4-tdoa3.csv` 的首个事件就是 UWB；旧公平
调度器从第二个事件循环，会少执行一次 UWB。现已在作者公平回放器和加速回放器
中显式处理“初始时刻 UWB”，但不在该时刻额外执行 IMU 预测。修复后的该文件
两边均执行 2,017 次预测和 573 次 UWB，拒绝均为 2 条，位置/速度/姿态/yaw
RMSE 相同，速度为 32.34x。合成回归数据也改为首事件含 UWB，2/2 自动测试通过。

这解释了此前某些文件“未测到失败”的现象：不是作者滤波突然变得更鲁棒，而是
批量入口此前没有覆盖该时间戳边界；该问题属于实验调度完整性，不改变滤波方程。

### 11.4 VSCode PowerShell 完整命令

在 `imu-uwb-inekf-acc` 目录执行：

```powershell
& "D:\conda-envs\imu-uwb-inekf\python.exe" `
  experiments\benchmark_all_constellations.py `
  --all-files --jobs 20 --cpu-list 0-19 `
  --repeats 2 --warmup-duration 10 --resume `
  --output-dir results\full_all_79_parallel
```

不要添加 `--max-duration` 或 `--limit-cases`，这样才是 79 条 CSV 的完整回放。
`--resume` 会复用已含有效 `metrics.json` 的成功条目；吞吐只按本次真正执行的
条目计算，避免把复用结果计成瞬时完成。每个 case 目录保留
`runner_stdout.log`、`runner_stderr.log`、曲线和 `metrics.json`，根目录生成
`summary.csv/json/png`。并行时的绝对耗时受 P/E 核和系统负载影响；若用于论文
报告绝对单核速度，应另在空闲系统用 `--jobs 1` 复测选定条目。

## 12. 全部 79 个完整文件最终结果

2026-07-24 已实际完成全部 79 个 CSV 的作者原始算法与纯计算加速版对比：

- 79/79 成功，失败 0；
- 13,548,996 个事件、9,984,576 次预测、3,564,341 次 UWB 更新；
- 单文件加速比中位数 38.9794x，范围 31.0226x--47.9746x；
- 作者/加速中位耗时之和 2,798.184/73.485 s，比值 38.0781x；
- 两边均拒绝 17,178 条 UWB，决策变化总数 0；
- 最大旋转/速度/位置/偏置/NIS 差为
  `3.33e-14/4.84e-14/3.80e-14/9.85e-18/8.27e-9`；
- 最大 RMSE 绝对差不超过 `6.05e-15`；
- 端到端完成时间 11.41 分钟，包括一次前台监控连接中断和无损续跑。

const4 的原算法误差分布明显更宽：最大位置 RMSE 0.8882 m，最大 yaw RMSE
10.6238 deg。最差条目的作者/加速曲线仍逐点重合，因此属于原算法/数据组合的
既有表现，不是加速引入。

79 个 case 均具有 `metrics.json`、`comparison.png`、
`nis_and_rejections.png` 和标准输出/错误日志；缺失 0，所有错误日志为空。
总览图、最小加速比、最大位置 RMSE、最大姿态/yaw RMSE 和最大 NIS 差条目均已
目检，没有发现加速引入的跳变、发散、门限改变或轨迹偏离。

详细最终结论：

- `reports/full_all_79_author_vs_acc_final_conclusion_2026-07-24.md`
- `imu-uwb-inekf-acc/results/full_all_79_parallel/FINAL_CONCLUSION.md`
