# Task06 同数据集 IMU+UWB 定位精度文献调研与当前项目对比

- 日期：2026-08-03
- 数据集：TU Delft, `UWB dataset (Version 1)`
- 数据 DOI：<https://doi.org/10.4121/14827680.v1>
- 本地数据：`E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset`
- 当前正式结果：`imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_delay_full/real`

## 1. 结论先行

严格按“确实使用了 DOI `10.4121/14827680.v1` 中的相同公开日志”筛选后，
可核实的同行研究主要来自同一作者团队的一条研究链：

1. Pfeiffer 等，IEEE RA-L 2021，数据集原始配套论文；
2. Pfeiffer 等，IMAV 2023，改进 MHE 会议论文；
3. Pfeiffer 等，International Journal of Micro Air Vehicles 2025，2023 工作的
   扩展期刊版。

没有找到独立第三方团队在该数据集全部 12 条 TWR 日志上公开报告可直接复核的
IMU+UWB 三维位置 RMSE。大量论文只引用 2021 工作或使用其他 UWB 数据集，未纳入
“同数据集”数值表。TU Delft 数据集门户目前也只把 2021 原论文列为直接关联研究
输出；2025 论文虽然在正文中明确引用该数据 DOI，但数据引用索引并不完整。

论文图中，使用全部 8 个锚点的 TWR 实验约为：

- 2021：MHE 平均三维位置 RMSE 约 `0.20 m`，EKF 约 `0.23 m`，成功率 100%；
- 2023/2025：各 MHE 变体约 `0.20--0.21 m`，EKF 约 `0.27 m`，成功率 100%。

这些数字不能直接当作“纯 IMU+UWB”公平上限。作者公开的 2021 复现实验脚本：

- 显式开启高度量测；
- 用 `OptiTrack z` 乘 2% 随机噪声生成高度量测；
- 用每一时刻的 OptiTrack 真值计算 UWB/高度残差，并据此在线给量测设置标准差。

因此论文约 `0.20 m` 是带连续真值辅助的公开协议结果，不是只依赖原始 IMU+TWR
的盲测结果。2023/2025 是同一作者、同一数据与同一估计器框架的延续，也不应当
视为独立复现实证。

本项目当前正式 proposed `deep` 方法在相同 12 条 TWR 日志上：

- nominal：全程位置 RMSE 中位/均值 `0.331862/0.349604 m`，12/12 收敛；
- `[6,-6,3] m` 大初差：中位/均值 `0.356247/0.362371 m`，12/12 收敛；
- 两种初值合并 24 次：中位/均值 `0.338625/0.355987 m`；
- 合并末 5 s：位置 RMSE 中位/均值 `0.211746/0.217799 m`。

只看数值，本项目 nominal 全程均值比论文图读数约高 `0.15 m`（约 75%）；但由于
论文协议包含 OptiTrack 派生高度和真值依赖噪声，不能据此判定本项目算法本身
落后 75%。更可信的结论是：本项目当前 raw-IMU/raw-TWR 全程精度仍是
`0.33--0.36 m` 量级，尚未达到论文的 oracle-aided `~0.20 m` 图示水平；同时本项目
给出了论文没有报告的 9 m 初始位置偏差捕获结果，并保持 12/12 收敛。

## 2. 数据集身份与任务06协议

数据集 README 明确说明，这些日志用于 2021 论文
“A Computationally Efficient Moving Horizon Estimator for Ultra-Wideband Localization
on Small Quadrotors”。数据由 Crazyflie 2.1 在 TU Delft Cyberzoo 采集，包含：

- 8 个约呈立方体分布的 DWM1000 UWB 锚点；
- 原始 IMU：三轴陀螺和加速度计；
- TWR 与 TDoA 两种 UWB 模式；
- OptiTrack 三维位置真值；
- square、triangle、octagon、hourglass、star、random 六类轨迹。

Task06 使用全部 12 条 TWR 日志，不使用 TDoA 日志。当前正式回放还对同一行 IMU
与 TWR 共同应用 `0.20 s` 采集到日志时延修正，在校正后的采集时刻插值 OptiTrack
用于评价。该时延来自一条预指定日志标定和其余 11 条复核，是当前数据集上的
描述性校准，并不是跨数据集盲测参数。

## 3. 检索范围与排除规则

检索使用了：

- 数据 DOI 和完整数据标题；
- 2021 论文标题及其引用论文；
- 原始 CSV 文件名；
- TU Delft Research Portal、DataCite、Semantic Scholar、Crossref；
- 作者公开代码仓库和论文中的数据引用。

纳入条件：论文明确引用该数据 DOI/数据条目，或逐项描述与本数据完全一致的
12 TWR + 9/12 TDoA Cyberzoo 日志并在实验中实际回放。只在 related work 中引用
2021 论文、只使用相同硬件、使用 UTIL/NTU VIRAL/其他 UWB 数据集、或只做仿真
的工作均排除。

由于数据集引用元数据不完整，本报告不能声称绝对穷尽所有未被搜索引擎索引的
论文；但截至 2026-08-03，可核实的同数据定量研究就是下表所列同一作者研究链。

## 4. 同数据集论文结果

| 研究 | 同数据证据 | TWR 实验与统计 | 8 锚点三维位置 RMSE | 主要结论 |
|---|---|---|---:|---|
| Pfeiffer et al., RA-L 2021 | 论文脚注直接给出数据 DOI；数据集为论文配套数据 | 12 条 TWR；每个锚点数每条日志 10 次，共 120 次；成功为 RMSE `<1 m`；仅对成功运行求均值 | EKF `~0.23 m`；MHE `~0.20 m`；均 100% 成功 | MHE 在真实 TWR 上略优于 EKF，锚点减少时优势增大 |
| Pfeiffer et al., IMAV 2023 | 参考文献列出同一数据条目；正文称 12 条 TWR/9 条 TDoA Cyberzoo 飞行 | 10 个随机种子，共 120 次 TWR；成功为 RMSE `<0.5 m`；失败运行从精度统计移除 | EKF `~0.27 m`；SG/LS/Newton + RANSAC/SVOR 各 MHE 约 `0.20--0.21 m`；均 100% 成功 | 动态步长和 SVOR 主要改善鲁棒性/计算量，MHE 变体之间精度几乎无差别 |
| Pfeiffer et al., IJMAV 2025 | 正文参考文献明确给出 DOI `10.4121/14827680.v1` | 2023 工作的扩展期刊版；同样为 12 条 TWR、9 条 TDoA、10 个种子 | 与 2023 图示同量级：MHE `~0.20 m`，EKF `~0.27 m` | 期刊化确认动态步长与开关变量改进；不是独立数据或独立团队复现 |

以上数值来自论文图在 8 锚点处的读数，论文没有给出对应数值表，因此按
`~0.01 m` 精度报告，不能伪装成更多有效数字。

### 4.1 2021 公开复现代码的关键协议

固定到论文时期代码提交后，`publication/headless_Nanchors.py` 明确执行：

```python
params.drone.altitude_enable = True
```

`UWBsim/airframe/drone.py` 中高度量测是：

```python
alt = log_data['otZ'] * (1 + np.random.normal(0.0, 0.02))
```

这不是数据集中的独立气压计或 ToF 高度，而是 OptiTrack 真值高度的带噪版本。
同一文件随后调用 `_get_meas_stdev_real`，其中 TWR 的标准差由

```python
twr_true = np.linalg.norm(measurement.anchor - self.state_true.x)
twr_error = measurement.distance - twr_true
```

形成最近真值残差窗口后估计；高度量测也采用同样的真值残差逻辑。因此：

1. 论文图的 TWR 测距值来自真实日志；
2. 估计器同时收到由 OptiTrack 生成的高度伪量测；
3. 量测噪声权重还持续使用 OptiTrack 真值信息；
4. 这是一种 oracle-aided 离线评估，不是可在线部署的纯 IMU+UWB 输入协议。

这两处代码分别来自 2021-07-16 的提交 `955360a8736c` 和 2021-06-30 的
“RAL2021 Full Code”提交 `b43a7f6d9033`，不是本次调研看到的后期无关改动。

### 4.2 论文方法状态范围

2021/2023/2025 的 MHE 只优化六维位置/速度状态；姿态由互补滤波器提供，
没有把陀螺偏置、加速度偏置纳入 MHE 状态。论文动力学使用 IMU 导出的姿态、
角速度与总推力。当前项目的 tracker 则维护

\[
\delta x=
[\delta\theta^\top,\delta v^\top,\delta p^\top,
 \delta b_g^\top,\delta b_a^\top]^\top\in\mathbb R^{15},
\]

capture 还维护平方距离增广状态，并负责大初差捕获和 tracker 重置。任务复杂度
和误差来源不同，进一步限制了对 RMSE 数字作单因素归因。

## 5. 本项目结果演进

### 5.1 Task06 原始结果

| 初值 | 方法 | 全程 RMSE 中位 (m) | 全程 RMSE 均值 (m) | 末 5 s 中位 (m) | 收敛 |
|---|---|---:|---:|---:|---:|
| nominal | 普通 TWR EKF | 0.387111 | 0.396931 | 0.192065 | 12/12 |
| nominal | 普通 TWR--InEKF | **0.346125** | 0.383262 | 0.201253 | 12/12 |
| nominal | Task06 松耦合增广 | 0.404446 | 0.401514 | 0.334650 | 12/12 |
| large | 普通 TWR EKF | 0.438736 | 0.477435 | 0.192065 | 12/12 |
| large | 普通 TWR--InEKF | 0.418764 | 30.725277 | 0.229953 | 10/12 |
| large | Task06 松耦合增广 | **0.407519** | **0.408992** | 0.334650 | 12/12 |

其中 large 是在首帧真值位置上增加 `[6,-6,3] m`，初始位置误差范数为
`sqrt(81)=9 m`。旧报告有时称 8.49 m 是错误心算；准确值为 9 m，本报告按公式
纠正，不沿用错误描述。

### 5.2 当前正式 deep 结果

共同传感器时延修正和 capture-to-tracker 自适应两滤波器完成后：

| 初值 | 方法 | 全程 RMSE 中位 (m) | 全程 RMSE 均值 (m) | 末 5 s 中位 (m) | 收敛 |
|---|---|---:|---:|---:|---:|
| nominal | 当前 ordinary InEKF | **0.322048** | 0.353685 | **0.162547** | 12/12 |
| nominal | 当前 proposed deep | 0.331862 | **0.349604** | 0.211725 | 12/12 |
| large | 当前 ordinary InEKF | 0.551103 | 84.233855 | 0.369452 | 7/12 |
| large | 当前 proposed deep | **0.356247** | **0.362371** | **0.211746** | 12/12 |
| 合并 24 次 | 当前 proposed deep | 0.338625 | 0.355987 | 0.211746 | 24/24 |

相对 Task06 原增广法，当前 proposed deep 的全程 RMSE 中位数：

- nominal：`0.404446 -> 0.331862 m`，下降 `17.9%`；
- large：`0.407519 -> 0.356247 m`，下降 `12.6%`。

当前 ordinary InEKF 在 nominal 下的中位数略优，但 large 下出现 5/12 次未达到
0.5 m 收敛判据，均值被最大约 493 m 的发散严重拉高。proposed deep 的优势是
保持全部 24 次有限和收敛，而不是每个名义稳态指标都最小。

## 6. 论文数字与本项目数字怎样比较

### 6.1 仅按表面 RMSE 比较

用最接近论文“全程成功运行均值”的本项目 nominal proposed deep 均值：

\[
\frac{0.349604-0.20}{0.20}\approx 74.8\%.
\]

即论文 MHE 图读数约 `0.20 m`，本项目约 `0.35 m`。若用本项目中位数
`0.331862 m`，相对差约 `65.9%`。因此不能说本项目已经达到同数据集文献的
最低表面数值。

### 6.2 为什么这不是公平算法排名

| 差异 | 论文 2021/2023/2025 | 本项目当前正式回放 | 影响 |
|---|---|---|---|
| 高度信息 | 公开 2021 协议把 OptiTrack z 加噪后作为高度量测 | 不把 OptiTrack 位置作为滤波量测 | 论文 z 误差被直接约束，RMSE 更低 |
| 量测噪声 | 公开 2021 协议用真值残差窗口设置 UWB/高度标准差 | 固定 `sigma_range=0.20 m` | 论文有 oracle 权重优势 |
| 状态 | MHE 六维位置/速度，姿态由互补滤波器给出 | 15 维 InEKF + 独立平方距离增广 capture | 估计难度不同 |
| 统计 | 跨运行均值；失败运行从精度图移除；首 1 s 不计分 | 报中位/均值/最大值；当前 deep 全部 24 次均计入 | 论文协议更容易得到较低表面均值 |
| 初值 | 默认零位置并先运行 1 s 再计分 | nominal 用首帧真值；large 明确加 9 m 偏差，完整保留捕获瞬态 | large 结果无论文对应项 |
| 参数使用 | 同一数据上选步长、噪声模型和锚点协议 | 同一数据上标定距离偏置、时延和监督器阈值 | 两者都不是独立盲测 |

### 6.3 可以成立的结论

1. 文献在其公开协议下达到了约 `0.20 m` 的 8 锚点 TWR 平均三维位置 RMSE。
2. 本项目当前 proposed deep 的 raw-IMU/raw-TWR 全程 RMSE 为约
   `0.33--0.36 m`，表面数值仍高。
3. 论文公开协议使用 OptiTrack 派生高度和真值依赖噪声，因此约 `0.20 m` 不能
   作为纯 IMU+UWB 的公平目标线。
4. 本项目相对 Task06 已有明确进步，并在 9 m 大初差下保持 12/12 收敛；论文
   没有报告同等初值扰动，不能比较捕获鲁棒性。
5. 当前项目末 5 s 位置中位数 `0.212 m` 已接近论文全程 `~0.20 m`，但时间窗口
   不同，不能据此宣称“达到论文精度”。

## 7. 获得真正同口径结论所需的后续基准

若需要回答“算法本身谁更准”，应在同一脚本中同时运行作者 MHE 和本项目方法：

1. 关闭作者协议的 `altitude_enable`，不注入 OptiTrack 高度；
2. 把 `_get_meas_stdev_real` 改为固定且双方一致的量测噪声，或只用可在线获得的
   创新统计；
3. 使用相同的 0.20 s 时间模型、TWR 事件变化检测、距离偏置和锚点坐标；
4. 对双方使用完全相同的 nominal/large 初值和计分起点；
5. 同时报 12 条日志的逐条值、均值、中位数、最大值、成功率和末 5 s 指标；
6. 把参数开发日志与至少一组独立留出数据分开。

本轮是文献调研和已有结果对比，没有修改算法代码，也没有重新调参或重跑数据集。
当前 CSV/JSON 数值已复核，`all_methods_comparison.png` 与代表性大初差轨迹已用
原分辨率检查：proposed deep 轨迹连续、有限；ordinary InEKF 的真实发散离群点
被完整保留，没有被中位数或坐标裁切隐藏。

## 8. 主要来源

1. TU Delft 数据集：<https://doi.org/10.4121/14827680.v1>
2. Pfeiffer et al. 2021：<https://doi.org/10.1109/LRA.2021.3095519>
3. 2021 论文开放 PDF：<https://pure.tudelft.nl/ws/portalfiles/portal/103441292/A_Computationally_Efficient_Moving_Horizon_Estimator_for_Ultra_Wideband_Localization_on_Small_Quadrotors.pdf>
4. Pfeiffer et al. 2023 IMAV：<https://www.imavs.org/papers/2023/10.pdf>
5. Pfeiffer et al. 2025：<https://doi.org/10.1177/17568293251340100>
6. 2021 锚点批测脚本（固定提交）：<https://github.com/SUPfeiffer/uwb-simulator/blob/955360a8736c/publication/headless_Nanchors.py>
7. 2021 数据回放与真值辅助量测代码（固定提交）：<https://github.com/SUPfeiffer/uwb-simulator/blob/b43a7f6d9033/UWBsim/airframe/drone.py>
8. 当前项目正式结果：`imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_delay_full/real/summary.json`
9. 当前时延修正报告：`reports/common_sensor_delay_correction_report_2026-07-31.md`
