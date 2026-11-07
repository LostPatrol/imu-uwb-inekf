# 原始论文代码与当前最新编译实现的公平对比入口

日期：2026-07-24  
代码目录：`E:\0syncthing\Code\InEKF\imu-uwb-inekf`  
原作者只读备份：`E:\0syncthing\Code\InEKF\imu-uwb-inekf - backup`

## 1. 完成内容

新增统一入口：

`experiments/benchmark_author_original_vs_latest.py`

它直接比较两套单滤波器实现：

- `author_original`：只读导入论文作者备份提交 `f3b664e` 的 `inekf.py`；
- `task03_compiled`：当前最新、完整 15 状态、float64 的 Numba 编译实现。

入口输出：

- `comparison.png`：XY 轨迹、位置误差、速度误差、姿态测地误差、带符号 yaw
  误差和对数坐标耗时柱图；
- `nis_and_rejections.png`：UWB NIS 与累计拒绝数；
- `metrics.json`：计时原始样本、分布统计、精度、调用计数、事件计划哈希和源码溯源。

为支持公平计时，编译核新增 `record_history=False` 模式。它仍执行全部 IMU 预测、
全部 UWB 门限与更新，只不把逐事件状态、协方差和 NIS 写入历史数组；额外返回实际
处理和拒绝的 UWB 数量，入口据此断言没有漏跑。正常估计入口和结果曲线仍使用完整
历史，算法公式与数值路径未被改变。

## 2. 公平性约束

- 两边共享同一个预计算事件计划和相同输入数据；
- 每轮严格核对 IMU 预测数与 UWB 处理数；
- 数据加载、事件计划构造、滤波器构造、JIT 编译/缓存加载、历史记录、指标计算和
  绘图均不计入正式回放；
- 正式计时两边均关闭 Python GC，数值库线程固定为 1；
- 先分别预热，再按 AB/BA 交替顺序测量，不挑选单次有利结果；
- 用额外的一次完整历史回放绘制效果曲线，该次不混入正式速度统计；
- 原作者备份未修改，提交和源码 SHA-256 写入 `metrics.json`。

原作者源码信息：

- commit：`f3b664e0e8c86b62c6ed2ea2a112261810228e1c`
- `inekf.py` SHA-256：
  `97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0`

注意：这是“原作者发布版本”与“当前完整生产版本”的系统级效果比较。两边保留各自
预期配置，因此不是只替换某个公式的单因素消融。

## 3. 30 秒正式验证

数据：

- `anchor_const1.npz`
- `const1-trial1-tdoa2.csv`
- `--max-duration 30 --warmup-duration 10 --repeats 4`

每个正式计时回合完全相同地执行 30,267 次 IMU 预测和 14,289 次 UWB 更新。

| 指标 | 原作者实现 | task03_compiled | 结果 |
|---|---:|---:|---:|
| 回放中位数 | 7.998309 s | 0.235335 s | 33.9869x |
| 回放 MAD | 0.030330 s | 0.006429 s | 两边均稳定 |
| 回放范围 | 7.898692--8.036066 s | 0.225013--0.241969 s | 无顺序翻转 |
| 耗时下降 | — | 97.0577% | 不含 JIT |

四组逐轮配对加速为
`34.4754x / 35.7138x / 32.9605x / 33.2059x`，配对中位数
`33.8406x`，MAD `0.7574x`。本轮 JIT 缓存加载/预热为 `0.1336 s`，已单独记录，
没有计入稳定态回放。

该约 34 倍结果不能简单解释为此前 `2.39x × 16.5x`。旧的
`author_original vs proposed` 与 `proposed vs task03_compiled` 使用了不同计时
入口和历史边界；本入口在两边正式计时中都关闭逐事件历史写入，因此必须以这里的
直接测量为准。

## 4. 效果与风险—回报

| 指标 | 原作者实现 | task03_compiled | 相对变化 |
|---|---:|---:|---:|
| 位置 RMSE | 0.116266 m | 0.107016 m | -7.956% |
| 速度 RMSE | 0.076748 m/s | 0.072891 m/s | -5.025% |
| 姿态测地 RMSE | 0.054714 rad | 0.157598 rad | +188.040% |
| yaw RMSE | 0.052949 rad | 0.157491 rad | +197.441% |
| yaw 持续进入 10° | 0.000 s | 10.438 s | 最新版有初始瞬态 |
| yaw 持续进入 5° | 0.000 s | 23.989 s | 最新版有初始瞬态 |
| UWB 拒绝 | 0/14,289 | 0/14,289 | 相同 |

位置和速度在该数据段有所改善，但姿态/yaw 指标明显变差，不能表述为“结果全面更好”。
主要原因是当前版本为未知全圆航向使用 2 rad 的保守 yaw 先验，而原作者在本条真实
初始 yaw 恰好接近 0° 的数据上使用更窄先验。该代价换取的是大初始 yaw 条件下不
过度自信，属于此前坐标一致性修复已经明确记录的风险—回报，不是编译加速导致的
数值误差。task03 编译核与其 Python `proposed` 基线已有逐元素等价测试。

目视检查结果图：

- 两条 XY 轨迹连续、无发散和异常跳变；
- 最新版的位置误差及多数速度误差较小；
- 最新版的初始姿态/yaw 瞬态明显，约 24 s 后持续进入 5°；
- NIS 均低于门限 25，两边累计拒绝始终为 0；
- 耗时柱图与 JSON 中位数一致。

## 5. 用户复现命令

在已激活环境的 VSCode PowerShell 终端、项目目录
`E:\0syncthing\Code\InEKF\imu-uwb-inekf` 下运行完整文件：

```powershell
python experiments\benchmark_author_original_vs_latest.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --warmup-duration 30 --repeats 4 `
  --output-dir results\manual_author_original_vs_latest
```

先快速检查 30 秒数据：

```powershell
python experiments\benchmark_author_original_vs_latest.py -i `
  "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz" `
  "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv" `
  --max-duration 30 --warmup-duration 10 --repeats 2 `
  --output-dir results\manual_author_original_vs_latest_30s
```

查看结果：

```powershell
Invoke-Item .\results\manual_author_original_vs_latest\comparison.png
Invoke-Item .\results\manual_author_original_vs_latest\nis_and_rejections.png
Get-Content -Raw -Encoding UTF8 `
  .\results\manual_author_original_vs_latest\metrics.json
```

完整文件会额外运行原作者实现的 4 轮计时、预热和一轮曲线记录，可能需要数分钟；
控制台每完成一个阶段都会即时打印 JSON 进度。

## 6. 验证与限制

- 入口已通过 2 秒 smoke test 和 30 秒正式数据验证；
- 自动测试共 17 项，覆盖编译路径全历史/无历史一致性及 UWB 处理计数；
- `comparison.png` 与 `nis_and_rejections.png` 已人工目视检查；
- 结果仅代表当前 Windows 主机、指定 Conda 环境、单线程和该数据段，不是嵌入式
  WCET，也不能从单条 const1 数据外推所有场景精度；
- 绝对耗时受 CPU 电源和桌面负载影响，应优先看 AB/BA 配对分布和相对加速；
- 若要比较不同初始航向，应显式传入相同的 `--yaw-deg`，并分别报告场景。

正式结果目录：

`imu-uwb-inekf/results/fair_author_original_vs_latest_const1_30s`

数学与计时边界说明：

- `tex/fair_author_original_vs_latest.tex`
- `tex/fair_author_original_vs_latest.pdf`（3 页，已逐页渲染检查）
