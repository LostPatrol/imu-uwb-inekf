<!-- Direct author-code comparison, frozen sensor protocol, and scientific scope. -->
# Pfeiffer 作者代码与 Delft 窗口对照

本目录直接导入 `third_party/pfeiffer_uwb_simulator_2021/UWBsim` 的作者 MHE、EKF、Mahony 及 Drone。没有重写滤波方程，也没有修改第三方源码。`evidence/author_sources.csv` 逐文件核对本地 Python 文本与官方提交 `cb816bdb291ac933577427448883acfa7c457c33`，仅正规化 CRLF/LF；全部相同。冻结参数、原始 CSV SHA256、训练/测试清单在 `evidence/protocol.json`。B 的首次执行来源单独冻结在 `protocol_common_execution.json` 和 `driver_common_execution.py.txt`；随后仅修正 A 的 hold 估计时间戳记录并重放 A，未改变 B 的算法或数据。

来源为 Pfeiffer、de Wagter、de Croon 的 [RA-L 2021 论文](https://doi.org/10.1109/LRA.2021.3095519)、[TU Delft 官方出版页](https://research.tudelft.nl/en/publications/a-computationally-efficient-moving-horizon-estimator-for-ultra-wi/)、[作者代码](https://github.com/SUPfeiffer/uwb-simulator/tree/cb816bdb291ac933577427448883acfa7c457c33)、[作者出版实验脚本](https://github.com/SUPfeiffer/uwb-simulator/blob/cb816bdb291ac933577427448883acfa7c457c33/publication/headless_Nanchors.py) 及 [4TU 原始数据 DOI](https://doi.org/10.4121/14827680)。这里比较的是该已发表版本，未将其命名为 2023/2025 的改进 MHE。

## 冻结协议

使用六种运动各自较晚的 TWR 记录，较早的六条记录只做静态 radio 校准。与 `radio_inertial_models/evidence/paired_acquisition` 一致：1.5 s 初始 IMU 段、IMU/UWB 共同相对真值移时 0.20 s、端点 5/10/20/30 s、2/4/8 s 窗口中不超出端点的十一组合、all8 与最低四基站 lower4。坐标使用测绘基站与 OptiTrack 原坐标，不进行评估轨迹配准或平移对齐。后续记录的真值仅用于合法时间范围截取与 B 的最终评分。该数据有开发暴露，不是新的盲测。

作者参数取出版脚本而不是 GUI 类默认值：MHE 50 Hz、20 个内部节点（约 0.4 s）、一次梯度迭代、步长 0.015、mu=10、RANSAC 十次、抽样比例 0.4、阈值 1.7 m；EKF 100 Hz、outlierThreshold=1500。两个基站数均使用作者对应步长。保留 0.03 kg、drag=(0.15,0.15,0)、`thrust=accZ*mass`、gyro 单位转换、作者姿态处理与原生起飞判据；没有额外电机或控制器输入。

| 条件 | A：作者原生连续重放 | B：相同传感器窗口适配 |
|---|---|---|
| 调用 | 原生 `Drone.step`，外部按 1 ms 时钟推进 | 原生 MHE/EKF 类，窗口内原 CSV 行直接喂入 |
| 历史 | 从首个 OptiTrack fix 连续至最后端点 | 每个 2/4/8 s 窗口重新初始化 |
| 平移、速度初值 | 作者默认零；不取真值 | 作者默认零；不取真值 |
| 初始姿态 | 作者默认水平且 yaw=0 | 每窗作者默认水平且 yaw=0 |
| 高度输入 | 开启：作者用 `otZ` 合成带噪高度测量 | 禁用 |
| radio 方差 | 作者 `_get_meas_stdev_real`，在线使用测试真值 | raw 固定 0.25 m；static 使用六条较早日志 MAD |
| radio 均值 | 原始 TWR | raw 零均值；static 减去六条较早日志逐基站中位残差 |
| radio 包 | 原生 Drone 的变化检测、调度与选包 | 与本方完全相同的 changed 且正距离事件，按原始行内 anchor ID 顺序 |
| 评分时钟 | 保留原生未校正时钟，并另给共同物理时钟重评分 | 与现有 paired acquisition 相同的实际端点 |

A 保留作者代码真实依赖的额外真值信息，仅用于证明该方法在其原生条件下能够运行。A 的原生时钟、调度与 B 不同，未声称它是无额外传感器的公平窗口比较。`actual_endpoint_s` 为原生输出采样时刻，`estimate_timestamp_s` 为最近一次滤波更新时刻；两者的 hold 间隔单列核验。A 各 duration 行是同一连续估计的重复连接键；统计只使用 96 个唯一端点，不将 264 个连接行当独立窗口。重放至约 31.7 s，未声称复现论文完整轨迹、十次种子与随机基站实验。

原生 Drone 为两种方法都计算 radio `stdDev`，但该版 MHE 的梯度与 RANSAC 不使用 `stdDev`；因此 A 的在线真值方差直接影响 EKF，而 MHE 的额外真值信息主要来自高度测量。B static 给作者类与本方的校准接口完全相同，但没有强改作者 MHE 方程以加入其原本没有的噪声权重。

B 的 2/4/8 s 是外部可用数据段，不是将作者内部 horizon 扩成 2/4/8 s。作者 MHE 始终使用出版的 N=20，保留其到达先验和单步计算模式。每个窗口使用独立局部时钟，避免 Mahony 第一帧以全局大时间差积分；端点必要时强制执行一次更新，确保最终当前窗事件被消费。这些输入调度适配被明确列出，不称为作者原生完整实验。

本方 `BiasPrior` 直接复用已有 joint 求解器。最初参考臂使用两个 CV 平移拟合、四个均匀 yaw 起点、每起点最多 120 次 raw 拟合、0.10 m/s² 高斯身体加速度偏置先验。随后独立补验真正的主要方法 `Profile1_BiasPrior`：同样两次 CV 拟合，原始圆约束 profile 单个 yaw 候选、零 bias，直接进行一次 BiasPrior raw 拟合；没有先做一次七态 raw 拟合，也不重复 CV。该补验单列协议，未重跑或挑选作者结果。

raw/static 与作者使用完全相同的 radio 修正及 sigma；两次 CV 拟合均计入 `nfev`。本方使用原有静止前缀估计 roll/pitch/IMU bias，继承截至窗口的陀螺积分姿态，再估计自由 yaw。作者适配使用每窗零姿态，且其 MHE 不估计未知绝对 yaw。因此本对照是共同原始传感器和 radio 校准下的直接实测，尚未统一姿态先验、模型及总计算预算，不能据此证明任意航向重启或某方法在所有条件下更优。

## 实测结果

B 最初参考臂 792 行及随后 Profile-1 补验 264 行均输出有限结果。每个表格单元覆盖六架次共 66 窗，按 **XY / Z / 3D RMSE（m）** 列出；所有失败如有发生均会保留。本次未发生作者断言或数值异常。有限输出与数值优化器宣告收敛分别记录。

| 校准与布局 | 作者 MHE | 作者 EKF | 本文 Profile-1 + BiasPrior 单起点 |
|---|---:|---:|---:|
| raw / all8 | 0.150 / 0.549 / 0.569 | 0.172 / 0.506 / 0.534 | 0.205 / 0.523 / 0.562 |
| static / all8 | 0.121 / 0.321 / 0.343 | 0.105 / 0.232 / 0.254 | 0.133 / 0.245 / 0.279 |
| raw / lower4 | 0.219 / 0.235 / 0.322 | 0.229 / 0.243 / 0.334 | 0.397 / 0.241 / 0.465 |
| static / lower4 | 0.187 / 0.876 / 0.896 | 0.188 / 0.869 / 0.889 | 0.204 / 0.854 / 0.878 |

static/all8 的 3D<0.5 m 窗口数分别为 MHE 56/66、EKF 65/66、本方 63/66。static 校准在 lower4 的垂直误差明显更差，不能只选取其水平改善汇报。本方在本表中没有全面优于已发表方法，尤其作者 EKF 在 static/all8 更好；MHE 与本方的水平/垂直指标各有差别。该结论由新的作者代码执行得到，不来自论文历史表格。

四起点参考与主要方法有实际差异：raw/lower4 四起点 XY/3D 为 0.303/0.390 m，单 Profile-1 为 0.397/0.465 m；25/66 窗单起点 3D 比四起点差超过 1 mm，最大差 1.123 m。Profile-1 的 raw/all8、raw/lower4、static/all8、static/lower4 分别有 2、51、0、11 个 raw 拟合到达 120 次上限；均保留其最终有限估计，未称其已收敛。static/all8、lower4 总 nfev 中位数为 39.5、65.0，含两次 CV。

参考阶段 static/all8 每窗 wall 中位数为 MHE 1.529 s、EKF 0.680 s、四起点本方 0.185 s；Profile-1 补验为 0.048 s。各阶段受同机并发影响，不是相同运算预算，不能据此作严格加速比或嵌入式性能结论。A 去重后在共同物理时钟的 3D RMSE 为 MHE all8/lower4 0.242/0.293 m、EKF 0.250/0.366 m；这些数值仍含作者特有高度信息与更长历史。

## 证据与复现

在项目根使用项目环境运行：

```powershell
& .\.venv\Scripts\python.exe -u .\imu-uwb-inekf\experiments\published_window_comparison\driver.py --workers 6 --verify-sources
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\published_window_comparison\analyze.py
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\published_window_comparison\profile_main.py --workers 2
```

每进程 BLAS=1。六个 worker 以架次独立运行；没有用实时运行时间上界截断难解窗口。

| 文件 | 内容 |
|---|---|
| `common_windows.csv` | B 六架次×十一窗×两布局×六臂，共 792 行，带原始 packet/IMU hash、实际起止时刻、消费事件数和有符号 XYZ 误差 |
| `profile1_author_windows.csv` / `profile1_author_overall.csv` | 作者两个方法与真正单起点本文主要方法，raw/static 各 132 窗；论文外部表应使用此文件 |
| `profile1_windows.csv` / `profile1_vs_four_start.csv` | 独立补验的 264 行与逐窗四起点对照，不将参考臂冒称主要方法 |
| `profile1_checks.json` / `profile1_protocol.json` | 每窗严格两 CV 加一次 raw 拟合、输入配对、预算上限及优化器未收敛数；实际拟合来源冻结在 `profile1_fit_execution.py.txt` |
| `common_overall.csv` / `common_duration.csv` / `common_per_log.csv` | XY/Z/3D、覆盖、0.5 m 端点成功比例、计算时间、预算分层 |
| `common_with_acquisition_reference.csv` | 逐窗连接旧 paired acquisition 的几何均值结果；该旧结果校准不同，仅供位置匹配溯源 |
| `paired_effects.csv` | raw/static 内逐窗直接对照，不跨校准条件混合 |
| `native_unique_endpoints.csv` / `native_overall.csv` | A 原生时钟唯一端点与分层结果 |
| `native_estimates_common_clock.csv` / `native_common_clock_overall.csv` | 保持 A 估计原值，按 B 实际物理端点真值重评分；仍有 A 特有的真值信息与历史 |
| `native_runs.csv` | 每架次/方法/布局一次连续重放的总时长和更新数；不要对累计端点计算时间重复求和 |
| `*__A_trace.csv` | A 连续估计和真值轨迹，可追溯原生调度 |
| `checks.json` | 完整行数、公共事件/IMU 同一性、与旧证据事件数匹配及失败清单 |
| `common_errors.png` | 共同窗口 XY/Z 分布可视化 |
| `profile1_author_errors.png` | 同一作者结果与真正 Profile-1 主要方法的 XY/Z 分布 |

“成功”在这里仅定义为有限的单端点 3D 误差小于 0.5 m，不是论文的整段轨迹 RMSE 指标，也不是已证明的闭环收敛。精度统计单独标注 finite 数与请求数；失败保留在成功率分母。各窗重叠，不将其当独立随机样本。B wall time 包括输入与求解，A 另有完整重放时长；Python 桌面计时不对应嵌入式实现实时性能。作者类没有与 least_squares 相同的 nfev 定义，保留更新数而不伪造等价函数评价预算。
