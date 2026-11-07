# 原作者 InEKF 与改进版 proposed 公平算法对比

日期：2026-07-24  
代码目录：`E:\0syncthing\Code\InEKF\imu-uwb-inekf`  
原作者只读备份：`E:\0syncthing\Code\InEKF\imu-uwb-inekf - backup`

## 1. 结论

在完整 `const1-trial1-tdoa2.csv` 上，先分别预热 30 s，再以
`AB/BA/AB/BA/AB/BA` 顺序完成 6 组单线程配对回放。原作者 InEKF 与
`proposed` 每轮均执行 142,220 次 IMU 预测和 67,205 次 UWB 更新；
`proposed` 还额外执行 11 次初始静止更新。

公平回放耗时中位数为：

| 指标 | 原作者 InEKF | proposed | 结果 |
|---|---:|---:|---:|
| 统一调度回放 | 44.5708 s | 18.6254 s | 配对中位加速 2.3933x |
| 预测+更新内核 | 44.1506 s | 18.2856 s | 配对中位加速 2.4147x |
| 回放耗时 MAD | 0.0661 s | 0.0297 s | 两边均稳定 |
| 回放最小--最大 | 44.4922--45.5002 s | 18.5902--18.7208 s | 无顺序翻转 |

`proposed` 的统一回放耗时下降 58.21%。这说明此前记忆中的 40--60 s
与原作者 InEKF 核心回放量级相符；当前入口十几秒不是漏跑数据，而是当前单滤波器
路径和结构化内核优化共同作用的结果。

精度并非四项全赢：位置 RMSE 改善 9.82%，速度 RMSE 改善 13.97%；
姿态测地 RMSE 增加 27.65%，yaw RMSE 增加 30.17%。后两项主要来自
`proposed` 的 2 rad 全圆未知 yaw 先验在本条“真实初始 yaw 恰好接近 0°”数据上
产生的初始瞬态；该保守先验用于避免大初始 yaw 时的过度自信。不能把位置/速度
改善偷换成“所有精度指标都改善”。

## 2. 为什么原作者入口不能直接和当前命令计时相除

原作者 `scripts/estimation/main.py` 一次运行包含：

1. 原作者 InEKF；
2. 一套 ESKF；
3. 基于 `t in array` 和 `np.where` 的逐事件全数组扫描；
4. 误差统计、大量绘图和交互式 `plt.show()`。

当前 `python scripts\estimation\main.py --variant proposed ...` 只运行指定的一套
滤波器，而且使用预索引事件调度。两条入口的算法数量、调度复杂度和绘图行为不同，
直接比较墙钟时间既不能说明 InEKF 内核速度，也不能回答是否完整处理了所有测量。

本次因此只读导入原作者 `inekf.py`，让两套滤波器进入同一个预计算事件计划。
原作者备份源码没有被修改，版本与哈希为：

- Git commit：`f3b664e0e8c86b62c6ed2ea2a112261810228e1c`
- `inekf.py` SHA-256：
  `97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0`

## 3. 公平计时约束

- 数据、锚点、初始时刻和事件顺序相同。
- 事件计划只构建一次，构建时间不计入。
- 两边逐轮断言预测和 UWB 调用数完全相同。
- 数据读取、滤波器构造、历史记录、RMSE、绘图均不计入正式回放时间。
- 正式计时关闭 Python GC；两边相同。
- `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`，避免自动并行随桌面负载变化。
- 先用同一数据的前 30 s 分别预热，再跑完整数据。
- 6 轮奇偶交替执行顺序，不挑选单次有利数字。
- “内核时间”在每次 `predict/correct` 外用同一计时器累计；“统一回放时间”
  还包含共同调度循环，两个范围互不混淆。
- `proposed` 的 11 次静止更新保留在计时内，所以加速结果没有通过删掉该功能获得。

直接执行 `D:\conda-envs\imu-uwb-inekf\python.exe` 会缺少 Conda 注入的 DLL
搜索路径。本次严格通过 `conda run -p` 启动；Conda 进程启动时间不在滤波回放
计时范围内。

## 4. 六轮原始耗时

| 轮次 | 顺序 | 原作者回放/s | proposed 回放/s | 配对加速 |
|---:|---|---:|---:|---:|
| 1 | 原作者→proposed | 45.5002 | 18.7047 | 2.4326x |
| 2 | proposed→原作者 | 44.6035 | 18.7208 | 2.3826x |
| 3 | 原作者→proposed | 44.4922 | 18.5902 | 2.3933x |
| 4 | proposed→原作者 | 44.7412 | 18.6291 | 2.4017x |
| 5 | 原作者→proposed | 44.5381 | 18.6217 | 2.3917x |
| 6 | proposed→原作者 | 44.5172 | 18.6012 | 2.3932x |

配对加速中位数 2.3933x，MAD 为 0.0050x。顺序反转没有改变结论。

## 5. 精度与效果

精度使用额外的一次完整回放记录共同的最小历史；该次回放不进入正式耗时统计。
两边使用各自版本的预期配置：原作者入口的 TDoA 方差 0.13、1 rad yaw 先验，
以及 `proposed` 对 const1 的 TDoA 方差 0.05、2 rad yaw 先验、静态偏置初始化
和 11 次静止更新。因此这是“原作者版本对当前版本”的系统级算法比较，不是只替换
一个矩阵公式的单因素消融。

| 指标 | 原作者 InEKF | proposed | 相对变化 |
|---|---:|---:|---:|
| 位置 RMSE | 0.108003 m | 0.097396 m | -9.82% |
| 速度 RMSE | 0.060901 m/s | 0.052396 m/s | -13.97% |
| 姿态测地 RMSE | 0.063826 rad | 0.081477 rad | +27.65% |
| yaw RMSE | 0.062300 rad | 0.081096 rad | +30.17% |
| yaw 持续进入 10° | 0.000 s | 10.438 s | 初始瞬态代价 |
| yaw 持续进入 5° | 0.000 s | 23.989 s | 初始瞬态代价 |
| UWB 拒绝 | 0/67,205 | 0/67,205 | 相同 |

目检 `comparison.png`：

- 两条 XY 轨迹均连续闭合，无发散或异常跳变；
- `proposed` 的位置误差曲线多数时间低于原作者；
- `proposed` 初始 yaw 误差峰值约 31°，约 24 s 后进入 5°，后半程通常低于原作者；
- 耗时柱图和 JSON 中位数一致。

## 6. 复现命令与交付物

```powershell
D:\miniconda\Scripts\conda.exe run --no-capture-output `
  -p D:\conda-envs\imu-uwb-inekf python `
  experiments\benchmark_author_original_vs_proposed.py -i `
  E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz `
  E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv `
  --warmup-duration 30 --repeats 6 `
  --output-dir results\fair_author_original_vs_proposed_const1_full
```

交付：

- 公平基准：`imu-uwb-inekf/experiments/benchmark_author_original_vs_proposed.py`
- 原始 JSON：`imu-uwb-inekf/results/fair_author_original_vs_proposed_const1_full/metrics.json`
- 比较图：`imu-uwb-inekf/results/fair_author_original_vs_proposed_const1_full/comparison.png`
- 数学/方法 PDF：`tex/fair_author_original_vs_proposed_benchmark.pdf`

## 7. 适用边界

该结果证明的是这台 Windows 主机、指定 Conda 环境、单线程、完整 const1 文件上的
离线 Python 回放速度。它不是目标嵌入式平台的 WCET，也不能从单条数据推出所有
场景精度。绝对秒数会随 CPU 电源状态和桌面负载变化；预热、交替顺序与配对中位数
使相对结论比单次墙钟时间可靠。若更换机器或 NumPy/SciPy，应重新运行同一脚本，
不要复用本报告的绝对秒数。
