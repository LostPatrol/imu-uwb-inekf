# 当前 Numba、Numba-CUDA、纯 Python 与 C++ 的机载性能评估

日期：2026-07-24  
工作仓库：`E:\0syncthing\Code\InEKF\imu-uwb-inekf`

## 1. 结论先行

对于当前这一个在线 IMU-UWB InEKF 实例，不建议把 CPU Numba 核迁移到
Numba-CUDA。它大概率不会产生显著加速，若按每个 IMU/UWB 事件启动 CUDA 内核，
单事件延迟反而很可能增加。核心原因不是 GPU 算力不足，而是当前问题由严格串行的
滤波递推、15x15 小矩阵和标量 UWB 秩一更新组成，没有足够的跨时间并行度和单次
计算规模去占满 GPU。

在“单个飞行器、单个滤波器、在线逐事件处理”这一明确场景下：

1. **预计原始计算速度：优化的固定尺寸 C++ 最快；彻底静态类型化的 Cython 与
   CPU Numba 属于同一高性能等级，谁略快必须实测；Numba-CUDA 通常无优势且可能
   更慢；无 Numba 的 Python/NumPy 最慢。**
2. **实飞适用性：固定尺寸 C++ 最好；彻底类型化、AOT 构建的 Cython 在 Linux
   companion computer 上略优于 JIT Numba；CPU Numba 适合科研实飞和快速验证；
   无 Numba 的 Python 只适合原型，Numba-CUDA 不适合当前单滤波器热路径。**
3. 当前 CPU Numba 已有约 110--126 倍的离线回放实时余量（含完整历史输出），
   再追求 GPU 吞吐对定位周期没有实际收益。实飞更应该优化最坏时延、启动、
   传感器 I/O、时间同步、故障恢复和热稳定性。
4. 如果任务变成几十到几百个独立滤波器、粒子滤波、Monte Carlo、离线全数据集
   并行，CUDA 才可能通过“跨实例并行”获得显著吞吐提升；这不是当前在线定位问题。

上述 C++ 和 CUDA 排序是基于当前代码结构的工程预测，不是伪装成实测的数字。
本轮只实测了当前 CPU Numba 与无 Numba 路径。

## 2. 当前实现与硬件事实

### 2.1 当前 CPU Numba 核

`scripts/estimation/compiled_replay.py` 的 `task03_compiled` 路径将以下内容全部放入
Numba `nopython` 热循环：

- IMU/UWB 合并事件调度；
- SO(3) 闭式名义状态传播；
- 15 维协方差传播；
- TDoA 几何、NIS 门限和标量秩一更新；
- 初始静止陀螺更新、状态注入；
- 可选固定尺寸历史输出。

它仍保留 float64、15 维状态、每个 IMU 时刻的协方差传播和全部 UWB 量测。

### 2.2 当前工作站

- CPU：Intel Core Ultra 7 265KF，20 核 / 20 线程；
- GPU：NVIDIA GeForce RTX 4070 SUPER，12 GB，报告功率上限 220 W；
- Python 3.11.15，Numba 0.61.2，NumPy 1.26.4；
- 当前 Conda 环境中 `numba.cuda.is_available()` 返回 `False`，即虽然系统有
  NVIDIA GPU，但当前项目环境没有可用的 Numba CUDA 运行时。

不应为了回答可行性问题直接改变已冻结的项目环境。若后续做独立原型，应使用单独
Conda 环境；Numba 0.61 的内置 CUDA target 已弃用，新部署应使用 NVIDIA 独立维护
的 `numba-cuda` 包。更需注意，截至本报告日期，Numba-CUDA 官方主页已说明其进入
维护模式，新功能方向转向 Numba-CUDA-MLIR。

官方资料：

- [Numba 0.61 内置 CUDA target 弃用说明](https://numba.readthedocs.io/en/0.61.0rc1/reference/deprecation.html#deprecation-of-the-built-in-cuda-target)
- [Numba-CUDA 安装与 CUDA 12/13 支持](https://nvidia.github.io/numba-cuda/user/installation.html)
- [Numba-CUDA 维护状态](https://nvidia.github.io/numba-cuda/)
- [CUDA Python 支持的 Python/NumPy 子集](https://nvidia.github.io/numba-cuda/user/cudapysupported.html)

## 3. 当前数据集实测

命令在指定环境中运行，固定 MKL/OpenMP/OpenBLAS 为单线程，执行两轮 AB/BA
交替配对：

```powershell
D:\miniconda\Scripts\conda.exe run --no-capture-output `
  -p D:\conda-envs\imu-uwb-inekf `
  python experiments\run_task03_compiled_optimization.py `
  --repeats 2 `
  --output-dir results\cuda_feasibility_cpu_baseline_2026-07-24
```

计时包括当前接口的完整状态、旋转、15x15 协方差、偏置和 NIS 历史输出，因此比
真实机载“只保留当前状态”的计算核更保守。JIT 缓存加载/预热为 0.3004 s，单独
记录，不计入稳定态回放。

| 数据 | IMU / UWB 频率 | 无 Numba Python/NumPy | CPU Numba | 加速 | Numba 平均每传感器事件 | Numba / 数据时长 |
|---|---:|---:|---:|---:|---:|---:|
| const1 yaw 0°, 30 s | 1008.9 / 476.3 Hz | 4.5567 s | 0.2706 s | 16.84x | 6.07 us | 0.902% |
| const4 yaw 90°, 60 s | 1008.9 / 431.9 Hz | 9.3616 s | 0.5269 s | 17.77x | 6.09 us | 0.878% |
| const3 yaw 180°, 60 s | 1008.9 / 227.5 Hz | 7.9309 s | 0.4751 s | 16.69x | 6.41 us | 0.792% |

无 Numba 路径在当前桌面上也可满足平均实时性，计算时间为数据时长的
13.2%--15.6%，但该结果不能外推为低功耗 ARM 机载平台的最坏时延保证。CPU Numba
约有两数量级实时余量，已不是本系统端到端更新率的瓶颈。

三组对照中：

- 四项 RMSE 的相对差均处于约 `1e-10%` 以下；
- UWB 拒绝决策变化为 0；
- 最大状态差不超过 `1.08e-11`；
- 最大协方差差不超过 `2.22e-11`；
- 比较图中轨迹、位置误差和 yaw 曲线重合，连续且无新增跳变或发散。

证据：

- `imu-uwb-inekf/results/cuda_feasibility_cpu_baseline_2026-07-24/metrics.json`
- `imu-uwb-inekf/results/cuda_feasibility_cpu_baseline_2026-07-24/comparison.png`

## 4. 为什么当前 CUDA 难以显著加速

### 4.1 时间递推不能并行

当前在线滤波满足

```text
(X_k, P_k) --predict/correct at event k+1--> (X_{k+1}, P_{k+1})
```

第 `k+1` 个事件必须等待第 `k` 个事件产生的状态和协方差，不能把 60,000 个 IMU
时刻分别交给 60,000 个 CUDA 线程。这直接消除了 GPU 最擅长的跨样本并行。

### 4.2 每次计算规模太小

传播最大只是 15x15 矩阵，UWB 是一个标量观测的秩一更新。单事件实测约 6 us，
而约 1 kHz IMU 的采样间隔接近 1 ms。即使假设 GPU 把纯计算变为零，端到端周期也
不会超过传感器到达率；现有 CPU 余量已经约 100 倍。

若把每个小矩阵乘分别做成 CUDA kernel，会引入多次 kernel launch、同步和小尺寸
访存开销。若把整个事件循环放进一个 persistent kernel，又只能让一个线程或一个
小线程块串行推进时间，GPU 的大量 SM 仍然空闲。

### 4.3 不是把 `@njit` 换成 `@cuda.jit`

当前 CPU 核在设备函数内频繁使用 `np.zeros`、`np.eye`、返回新数组、矩阵乘法、
外积以及一个 3x3 `np.linalg.solve`。Numba-CUDA 官方说明设备代码不允许动态数组
分配，只支持有限 NumPy 子集，数组表达式也不会自动并行。因此 CUDA 版本需要手写
固定索引、共享内存/寄存器布局、同步和矩阵核，是一次实质性重写，不是装饰器替换。

这种重写还会带来：

- CPU/GPU 异步错误与同步点更难验证；
- float64 在嵌入式 GPU 上通常不是优势路径；
- 传感器数据和状态跨 CPU/GPU 所有权更复杂；
- GPU 被视觉/神经网络占用时会出现资源竞争和时延尾部；
- Numba-CUDA 运行时、驱动和 JetPack/CUDA 版本形成额外部署矩阵。

### 4.4 离散 GPU与 Jetson 均不改变并行度事实

离散 GPU 还需考虑 PCIe 传输，220 W 的本机 RTX 4070 SUPER 本身不属于通常的小型
机载功耗级。Jetson 统一内存能减少显式 PCIe 拷贝，但不能消除严格串行递推、
kernel launch 和低占用率。

NVIDIA 当前资料显示，Jetson Orin Nano/NX/AGX 的功耗和算力跨度很大，例如 Orin
Nano 常见 7--25 W，Orin NX 可到 40 W 新功耗模式。AI TOPS 或 CUDA 核心数量不能
直接转换为 15 维 EKF 的单实例延迟。

- [Jetson Orin 官方规格](https://www.nvidia.com/en-eu/autonomous-machines/embedded-systems/jetson-orin/)
- [JetPack 6.2 Orin Nano/NX 功耗模式](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)

## 5. 四种方案的速度与实飞适用性

| 方案 | 当前证据 / 预计速度 | 单实例在线延迟 | 部署与确定性 | 实飞建议 |
|---|---|---|---|---|
| 无 Numba Python/NumPy | 已实测最慢；桌面仍为 6.4--7.6x 实时 | 最高 | Python 调度、临时分配和运行库尾延迟较多 | 仅原型、回放、算法调试 |
| 当前 CPU Numba | 已实测比当前 Python 快 16.69--17.77x | 很低，约 6 us/传感器事件平均值 | 需 Python/Numba/LLVM 环境、启动预热和缓存管理 | 科研实飞可用；短期推荐 |
| 深度优化 Cython | 当前环境未安装、未实测；预计与 Numba 同量级，可能略快或略慢 | 很低 | AOT 原生扩展，无 JIT 预热；仍依赖 CPython/NumPy ABI | companion computer 的可选生产过渡 |
| Numba-CUDA | 未实现；对当前单滤波器预计不快于 CPU Numba，可能更慢 | launch/同步主导风险 | NVIDIA GPU、CUDA/驱动/包版本绑定，调试和资源竞争复杂 | 当前不推荐 |
| 固定尺寸 C++ CPU | 未实现；预计最快或与 Numba 同量级，可通过静态缓冲和展开小矩阵继续降低常数 | 最低且最容易做 WCET | 单二进制、静态内存、线程/优先级/异常策略可控 | 生产实飞首选 |

### 5.1 Cython 会带来什么

本报告所说的 Cython 必须分成两类：

1. **浅层 Cython**：只是把当前 `.py` 改为 `.pyx` 后编译，仍执行 `np.zeros`、
   `np.eye`、`@`、`np.outer`、切片返回和 Python/NumPy 函数调用。这类改法只减少
   少量函数分派，预计不会接近当前 Numba，更不能直接写成“C 速度”。
2. **深度 Cython**：把状态、协方差和临时量改为连续 `double` typed memoryview
   或固定 C 数组；对 3x3、15x15 运算写固定循环；热路径不创建 Python 对象；
   使用 `cdef inline`、明确异常语义，并在适当边界使用 `noexcept nogil`。这种实现
   才可能与 Numba/C++ 同量级，但它的工作量已经接近一次受控 C/C++ 移植。

Cython 官方文档明确指出，直接按普通 Python 方式使用 NumPy 会失去潜在加速；
typed memoryview 才能用 C 整数直接访问数据缓冲并消除 Python 索引开销。Cython
会在构建期先生成 C/C++，再由本机编译器生成 `.so`/`.pyd` 扩展，因此没有 Numba
首次 JIT 或缓存加载延迟。

`nogil` 不是单线程加速开关。官方说明同一份单线程 C 级代码持有或释放 GIL 的速度
相同；它的价值是允许其他 Python/传感器线程运行，或使用并行循环。当前 InEKF 的
相邻时间步仍然串行，所以 `nogil` 不改变 CUDA 并行度结论。错误的 `except *`
函数还可能在每次调用时重新获取 GIL 检查异常，反而增加热路径开销。

官方资料：

- [Cython 源码与 AOT 扩展编译](https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html)
- [Cython 的 NumPy 类型化性能路径](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html)
- [Typed memoryview 与 GIL](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)
- [Cython `nogil` 的真实作用](https://cython.readthedocs.io/en/latest/src/userguide/nogil.html)

当前 `imu-uwb-inekf` Conda 环境未安装 Cython，本轮未更改环境，也未实现或基准
Cython 核。因此 Cython 的相对位置只能给工程区间，不能给虚假倍数。

### 5.2 “谁最快”的严格回答

- **当前已经实测的两项：CPU Numba 明显快于无 Numba。**
- **当前单实例问题的工程预测：固定尺寸 C++ CPU 最快；深度 Cython 与 CPU Numba
  同属第二梯队；CUDA Numba 不会凭 GPU 自动超过它们；无 Numba 最慢。**
- 不能给 C++、Cython 或 CUDA 写出虚假的倍数。Cython 是否比 Numba 快 20% 还是
  慢 20%，取决于是否消除所有热循环 Python/NumPy 对象、目标编译器、固定尺寸展开、
  异常检查、是否保留历史以及时钟/热限制，必须在目标机载计算机上实测。

### 5.3 “谁更适合实飞”的严格回答

1. **量产/长期飞行/强实时约束：C++。** 使用固定尺寸数组、启动期一次性分配、
   禁止热循环异常和动态分配、固定线程优先级，并在目标硬件做热浸和最坏时延测试。
2. **Linux companion computer 的过渡生产方案：深度 Cython。** AOT 扩展消除了
   JIT 预热失败风险，也能在传感器线程并发时释放 GIL；但仍有 CPython、扩展 ABI
   和平台构建矩阵，不等同于独立 C++ 二进制。
3. **当前科研机载验证：保留 CPU Numba。** 它已经有巨大余量，算法迭代快，并且
   已与 Python 路径逐点回归。启动后必须先预热，若预热失败则禁止解锁/起飞。
4. **纯 Python：只作回放和参考实现。** 当前桌面平均实时不等于机载最坏时延合格。
5. **Numba-CUDA：不作为当前滤波核的实飞路线。** 若 Jetson 的 GPU 已用于视觉，
   更应把小型 InEKF 留在 CPU，减少 GPU 争用。

若“机载计算机”实际是 MCU（如无 Linux/Python 的飞控板），Numba 和 CUDA 均不成立，
C/C++ 是唯一合理路线。若是 x86 工控机或 Jetson companion computer，CPU Numba
可作为过渡，C++ 仍是最终生产路线。

## 6. 何时应重新考虑 CUDA

满足下列至少一项时才值得做独立 CUDA 原型：

- 同时运行几十到几百个互不依赖的滤波器、粒子或多假设 yaw；
- 离线 Monte Carlo / 全数据集批量验证，吞吐而不是单次控制延迟最重要；
- 状态维数和观测批量大幅增加，转为大矩阵平滑/图优化；
- 传感器前端已经常驻 GPU，能完全避免逐事件 CPU-GPU 往返；
- 目标硬件已经冻结为 Jetson，且 GPU 资源预算和热设计有明确余量。

即使如此，也应优先按“每个独立滤波实例一个 block/warp”做批处理，而不是试图并行
同一滤波器的相邻时间步。

## 7. 建议的实施路线

### 近期

- 保持当前 `task03_compiled` 作为科研实飞路径，不做 CUDA 改写；
- 机载模式关闭完整协方差历史，只保留当前状态、环形诊断缓冲和必要日志；
- 启动时固定执行 JIT 预热，并做一次短自检；
- 在目标硬件记录 p50/p95/p99/max 单事件时延、CPU 占用、内存和温度降频。

### 生产化

- 将已冻结并通过回归的 Numba 核逐函数移植为固定尺寸 C++；
- 如果希望先降低迁移风险，可先实现一个深度 Cython AOT 核作为中间里程碑，但必须
  预先设定退出条件：若仍频繁调用 NumPy、无法清除热路径 Python 对象或目标是 MCU，
  直接转 C++，不要把浅层 Cython 当作完成；
- 优先使用栈/预分配内存和显式 3x3、15x15 运算，避免运行期堆分配；
- 用当前 79 个 CSV 的逐量测拒绝决策、状态、协方差、NIS 和 RMSE 作为交叉语言
  golden regression；
- 在目标机载硬件上做长时间热浸、I/O 干扰、视觉/GPU 满载、丢包和时钟异常测试；
- 只有 C++ 版本通过数值等价与最坏时延后，才替换科研 Numba 路径。

## 8. 本轮边界与诚实性说明

- 未改动滤波算法和生产代码；
- 未安装 CUDA 工具链，未实现 CUDA 版本；
- 当前环境未安装 Cython，未实现 Cython 版本；
- 未实现 C++ 版本；
- CUDA、Cython 和 C++ 的相对排序基于已审计的数据依赖、矩阵规模、当前实测余量
  及部署特性；
- 最终机载 WCET 结论必须在用户选定的具体机载计算机、功耗模式和热条件下重测。

## 9. 交付前验证

- 现有 `tests/test_inekf.py`：17/17 通过；
- `metrics.json` 已成功解析，三组配对结果和数值差完整；
- 基准命令的 60 s 外层执行限时在全部结果、JSON 和 PNG 已写出后触发，产物时间戳、
  JSON 尾部和三组汇总均完整，不把该外层包装器退出码伪装成正常退出；
- `comparison.png` 已用图片理解逐图检查：轨迹连续，Numba/无 Numba 曲线重合，
  无新增异常跳变或发散；
- `tex/numba_cuda_cpp_onboard_evaluation.pdf` 共 4 页，已逐页渲染检查；
- 最终 LaTeX 日志无 overfull，中文、表格、公式、图和超链接均未裁切；
- `imu-uwb-inekf - backup` 未修改；本轮也没有改动工作仓库中的滤波源码。
