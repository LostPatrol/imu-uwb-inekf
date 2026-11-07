# Task06：Pfeiffer 同数据集论文代码、复现与公平对比

- 日期：2026-08-03
- 数据集：TU Delft `UWB dataset (Version 1)`，DOI `10.4121/14827680.v1`
- 本地任务：`imu-uwb-inekf-state-augment`
- 作者源码镜像：`third_party/pfeiffer_uwb_simulator_2021`
- 作者源码固定提交：`955360a8736cae3600a1fbed530a9ec19a2ad724`
- 正式输出：`imu-uwb-inekf-state-augment/results/task06_author_fair_compare`

## 1. 结论先行

1. **只有 2021 RA-L 论文公开了可核实源码。** 作者仓库
   `SUPfeiffer/uwb-simulator` 为 MIT 许可，论文和数据门户都直接链接该仓库。
   本地已固定到论文时期提交 `955360a...`，上游目录保持未修改。
2. **2023 IMAV 与 2025 IJMAV 没有找到公开的 LS/Newton/SVOR 实现。** 两篇论文
   没有 GitHub、source-code 或 code-availability 链接；作者公开的
   `uwb-simulator` 只有 2021 年 6 次提交，仍是 fixed-step SG+RANSAC。
   因而这两篇只能按公式独立重写，不能声称为“作者代码复现”，也不能在缺少
   完整调参与协议的情况下把重写结果用于可靠排名。
3. **2021 作者结果已在本地精确复现。** 12 条 TWR 日志、10 个种子、8 锚点下，
   作者原协议的 MHE/EKF 三维 RMSE 均值为 `0.201051/0.235375 m`，与论文图中
   约 `0.20/0.23 m` 一致。
4. **作者原协议不是纯 IMU+TWR。** 公开批处理脚本开启高度观测；回放代码把
   `OptiTrack z` 加 2% 乘性噪声后作为高度量测，并用当前 OptiTrack 真值残差
   在线估计量测标准差。需要精确区分：真值标准差影响作者 EKF；2021 MHE 的
   代价函数没有读取 `stdDev`，所以 MHE 的主要运行时真值帮助是伪高度观测。
5. **已切断这两条运行时真值路径并完成公平重跑。** 在与 Task06 相同的完整轨迹、
   首帧位置初始化、逐锚 LOSO 偏差/尺度下，作者 MHE/EKF 三维均值为
   `0.338650/0.321290 m`，XY 为 `0.189810/0.161072 m`。
6. **本项目不能声称全面超过。** 当前 Task06 InEKF 三维均值为 `0.333703 m`：
   比作者 MHE 低 `4.95 mm`，逐序列赢 `8/12`，但三维中位数反而高
   `9.89 mm`；XY 比作者 MHE 低 `19.78 mm`，逐序列赢 `12/12`。然而作者 EKF
   三维和 XY 分别比 Task06 好 `12.41/8.96 mm`，Task06 两项均只赢 `1/12`。
   再加上 2023/2025 无法做无真值源码复现，现有证据不支持“达到或超过所有研究”。
7. **截至 2026-08-03，没有核实到独立第三方在这 12 条日志上报告可直接复核的
   IMU+TWR 轨迹精度。** 可核实的定量复用仍是 Pfeiffer 同一团队的论文链。
   这是基于 DOI、文件名、数据门户、DataCite 和原论文引用网络的检索结论，不是
   对未被索引论文的绝对不存在证明。

## 2. 三篇论文的代码开放情况

| 论文 | 公开代码结论 | 可复现范围 |
|---|---|---|
| Pfeiffer et al., IEEE RA-L 2021 | **有**；MIT 仓库 `SUPfeiffer/uwb-simulator`，论文脚注与数据门户均链接 | SG+RANSAC MHE、作者 EKF、TWR/TDoA 仿真入口；本轮已复现 |
| Pfeiffer et al., IMAV 2023 | **未找到** LS/Newton/SVOR 源码或代码声明 | 只能从论文公式 clean-room 重写；不能称作者代码复现 |
| Pfeiffer et al., IJMAV 2025 | **未找到** LS/Newton/SVOR 源码或代码声明 | 与 2023 同样；网页开放论文不等于开放实现 |

作者 GitHub 公开列表中，`uwb-simulator` 最后一次推送仍在 2021 年；固定提交之后的
`cb816...` 只增加论文链接，没有 2023/2025 算法。作者的 Crazyflie firmware fork
虽有较晚推送，但公开树中没有可识别的 SG/LS/Newton/SVOR MHE 文件。因此本报告的
“未开源”应理解为“在论文、作者公开仓库及精确关键词检索中未找到”，不排除作者
持有未公开研究代码。

## 3. 三篇工作的关系与真正改进点

### 3.1 2021：先把 MHE 做到微型飞行器能运行

2021 工作把完整 MHE 简化为忽略过程噪声的 output-error MHE，只优化六维位置/机体
速度状态；姿态由互补滤波器提供。其代价为

\[
J(\xi_{t-N})=\mu\lVert\xi_{t-N}-\bar\xi_{t-N}\rVert^2
+\sum_{k=t-N}^{t}\sum_i
\lVert y_{i,k}-\mathcal H_{i,k}(\xi_{t-N})\rVert^2,
\]

每次只做一步固定步长梯度下降

\[
\hat\xi_{t-N}=\bar\xi_{t-N}-\alpha\nabla J(\bar\xi_{t-N}),
\]

并以 RANSAC 处理 UWB 长尾异常。重点不是单纯刷新最低 RMSE，而是：重尾噪声下比
EKF 更稳、锚点减少时仍可工作、并能在 30 g Crazyflie 上运行。论文报告 MHE 约占
15% CPU，EKF 约 11%。

### 3.2 2023：修复固定步长不稳，并降低 RANSAC 成本

2023 直接指出 2021 的两个瓶颈：

- 梯度是量测项之和；当每步量测数变化时，固定 `alpha` 的有效更新幅度随之变化，
  对 4 锚合适的步长可能在 8 锚时发散；
- RANSAC 鲁棒但要重复抽样和前向计算，对微控制器成本过高。

它增加两类动态步长：line search 在
`{0.5, 0.8, 1.0, 1.25, 2.0} alpha` 中选最低代价候选；Newton 法用 Hessian
逆矩阵实现分方向步长。异常处理则为每个量测引入开关变量

\[
\operatorname{sig}(s_{i,k})=\frac{1}{1+e^{-s_{i,k}}},
\]

并在代价中同时惩罚关断量测和量测残差，使估计器连续地开/关异常量测，称为
SVOR。12 条 TWR、9 条 TDoA、10 个种子的图显示：8 锚 TWR 时各 MHE 版本精度
几乎没有可见差异；改进主要体现在不同量测数下的成功率与计算量，而非把 8 锚
精度从约 0.20 m 显著降到更低。

### 3.3 2025：2023 的期刊化扩展，而不是第三条独立算法

2025 与 2023 标题、作者、两项核心贡献、主要公式、12 TWR/9 TDoA×10 的离线协议
以及 SG/LS/Newton×RANSAC/SVOR 对照均相同。期刊版补充并更完整地阐述了机载实验：
采用 Newton+SVOR，把 Flow Deck 的 ToF 高度和光流量测加入带协方差权重的代价，
窗口限制为 80 个量测；报告 MHE 约 25% CPU，EKF 约为其五分之一。其贡献是把
2023 方案更完整地验证和归档，不应作为第三个相互独立的数据集复现实验计票。

因此这一研究链持续解决的核心问题是：**在 UWB 非高斯/长尾噪声、量测数变化和
极小机载算力下，让简化 MHE 同时可运行、不过度发散、异常处理成本可接受。**
它没有重点解决完整惯导偏置状态、yaw 漂移或纯 UWB 的垂直弱可观性；2025 也明确
承认姿态仍来自互补滤波，yaw 需要正确初始化。

## 4. “真值辅助”具体是什么，以及切断方法

### 4.1 作者 2021 公开协议的两条运行时真值路径

批处理脚本设置 `params.drone.altitude_enable = True`。日志回放每一时刻构造

\[
z_k^{\rm pseudo}=z_k^{\rm OptiTrack}(1+\varepsilon_k),
\qquad \varepsilon_k\sim\mathcal N(0,0.02^2),
\]

并送入 MHE 与 EKF。它不是日志中的独立气压计或 ToF，而是评分真值的带噪副本。
同时 `_get_meas_stdev_real` 计算

\[
e_{k,i}^{\rm twr}=\tilde\rho_{k,i}
-\lVert a_i-p_k^{\rm OptiTrack}\rVert,
\]

再用最近残差窗口的标准差设置 UWB `stdDev`。该权重被作者 EKF 使用；2021 MHE
梯度和 RANSAC 直接使用残差，未读取 `stdDev`。因此不能笼统地说“两条路径都同等
改善 MHE”：MHE 的核心 oracle 收益来自高度，EKF 同时受高度和真值权重影响。

这里称作 oracle-aided 是为了标明传感器协议，而不是说作者伪造结果。论文模型本来
允许独立 barometer/ToF 高度；问题只在于把这个离线协议的 3D 数字与纯 IMU+TWR
系统不加说明地排名。2025 的真实机载实验使用 Flow Deck ToF，不属于真值作弊。

### 4.2 本轮无运行时真值适配

上游作者目录没有修改。本项目适配器在运行时：

1. 关闭 `altitude_enable`；
2. 把 `_get_meas_stdev_real` 替换为冻结尺度，不读取 `state_true`；
3. 对待测序列 `j`，只用其他 11 条日志冻结逐锚参数

\[
\hat b_i=\operatorname{median}_{k\in\mathcal D_{-j}} e_{k,i},
\qquad
\hat\sigma_i=\operatorname{clip}
\left(1.4826\operatorname{MAD}_{k\in\mathcal D_{-j}}e_{k,i},0.08,0.35\right),
\]

测试时只使用 `rho_corrected = rho_raw - b_i` 与冻结 `sigma_i`；
4. 双方都以首帧 OptiTrack 位置初始化并从完整轨迹 `t>=0` 计分。

第 4 项仍是一帧初始化真值，第 3 项仍是用其他日志真值做监督式传感器标定；它们
不是完全无外部真值的盲测，但双方一致，且待测日志运行中不再注入真值。部署时应
用独立标定场次冻结参数，而不是把这次 LOSO 包装成跨场景泛化结果。

## 5. 本地复现实验

### 5.1 执行规模与完整性

- 12 条有效 TWR 日志，8 锚点；
- 3 协议：`paper_oracle`、`no_oracle_raw`、`no_oracle_loso`；
- 每协议 10 个种子，MHE 与 EKF 同时评分；
- 共 360 个回放任务、720 行方法结果；
- 保留作者 1 ms 仿真步长；4 进程墙钟约 1400 s；
- 720/720 行有限，无真值协议中 0 行出现运行时真值量测标志。

`paper_oracle` 使用作者零位置初始化、开启伪高度、真值残差标准差并去掉前 1 s；
`no_oracle_raw` 关闭两条真值路径、不校正距离偏差、固定 0.20 m 尺度；
`no_oracle_loso` 采用上述公平逐锚 LOSO 参数和完整轨迹。

### 5.2 作者代码三协议结果

| 协议 | 方法 | 3D均值 | 3D中位 | XY均值 | z均值 | `<0.5 m` |
|---|---|---:|---:|---:|---:|---:|
| 作者原协议 | MHE | 0.201051 | 0.201250 | 0.191061 | 0.061595 | 120/120 |
| 作者原协议 | EKF | 0.235375 | 0.234154 | 0.232630 | 0.035726 | 120/120 |
| 去真值、无偏差标定 | MHE | 0.563604 | 0.578655 | 0.187121 | 0.530934 | 20/120 |
| 去真值、无偏差标定 | EKF | 0.558660 | 0.577495 | 0.182602 | 0.527470 | 20/120 |
| 去真值、公平LOSO | MHE | 0.338650 | 0.318420 | 0.189810 | 0.275932 | 110/120 |
| 去真值、公平LOSO | EKF | **0.321290** | **0.313002** | **0.161072** | **0.274212** | **120/120** |

单位均为 m。去真值、无偏差标定时，XY 只小幅变化，z 从约 0.04--0.06 m 增到
约 0.53 m，直接说明原协议的三维优势主要来自高度约束。加入公平逐锚 LOSO 后，
z 回落到约 0.275 m，但仍是三维误差主体。作者无真值 MHE 的 10 次失败全部来自
`10_09_28_twr_star`，该序列十个种子均约 0.542 m；失败没有从精度表删除。

### 5.3 与当前 Task06 的同口径结果

| 方法 | 3D均值 | 3D中位 | XY均值 | XY中位 | z均值 |
|---|---:|---:|---:|---:|---:|
| 作者 MHE，无真值LOSO | 0.338650 | **0.318420** | 0.189810 | 0.189320 | 0.275932 |
| 作者 EKF，无真值LOSO | **0.321290** | **0.313002** | **0.161072** | **0.165110** | **0.274212** |
| Task06 direct | 0.334476 | 0.332875 | 0.171259 | 0.174492 | 0.283950 |
| Task06 InEKF | 0.333703 | 0.328305 | 0.170033 | 0.172807 | 0.283201 |
| Task06 lifted | 0.361756 | 0.357657 | 0.170750 | 0.173223 | 0.317015 |

单位均为 m，均为 12 条日志的逐序列 RMSE 聚合。配对比较显示：

- Task06 InEKF 对作者 MHE：3D 赢 `8/12`，均值低 `0.004947 m`，但中位数高
  `0.009885 m`；XY 赢 `12/12`，均值低 `0.019777 m`（约 10.4%）。
- Task06 InEKF 对作者 EKF：3D 与 XY 均只赢 `1/12`；均值分别高
  `0.012414/0.008961 m`（约 3.9%/5.6%）。

因此最诚实的结论是：**当前 InEKF 已达到作者 2021 MHE 的公平 3D 量级，并在 XY
明显优于该 MHE；但它没有超过同一作者仓库中的 EKF，也没有证据覆盖未开源的
2023/2025 改进 MHE。** 不能把“对作者 MHE 的均值略优”扩展为“同数据集综合最优”。

本轮没有比较计算时间：作者 MHE/EKF 在同一模拟器中平均单步约
`3.80/0.82 ms`，但 Task06 使用另一套实现和计时路径，直接放在一起不构成硬件公平
的实时性排名。

## 6. 为什么只比较 XY 是有价值但不充分的

定义

\[
\operatorname{RMSE}_{XY}
=\sqrt{\frac1N\sum_k\left[(\hat x_k-x_k)^2+(\hat y_k-y_k)^2\right]},
\quad
\operatorname{RMSE}_{3D}^2
=\operatorname{RMSE}_{XY}^2+\operatorname{RMSE}_{z}^2.
\]

XY 对比能剥离作者伪高度观测的直接作用，且本轮代表轨迹显示三协议 XY 曲线基本
重合，因而比原论文 3D 数字更公平。但它仍不能消除：不同状态模型、姿态估计、
初值、偏差标定、门控与计分窗口带来的影响。因此本轮把 XY 与“完全切断运行时
真值后的 3D”同时报告，而没有只选对本项目有利的 XY。

## 7. 是否有其他作者使用同一数据集

检索采用：精确 DOI `10.4121/14827680.v1`、数据集完整标题、原始 CSV 文件名、
`anchor_positions.yaml`/字段名组合、TU Delft 数据门户、DataCite 关系以及 2021
论文的 OpenAlex 引用作品。

- TU Delft 数据门户目前只列 2021 原论文为关联 research output，并明确数据和代码
  均为该论文配套；
- DataCite 只记录该数据集 `IsSupplementTo` 2021 RA-L DOI；
- 2021 论文在 OpenAlex 中有 16 个引用作品。逐项检查标题和实验范围后，LocaRDS、
  UTIL、相对定位、GNSS MHE、UWB 节点自标定、移动机器人 MHE、调查综述等只把
  Pfeiffer 作为相关工作或使用自己的数据；
- 2025 同作者论文明确引用数据 DOI，2023 会议论文也明确使用同一 12 TWR/9 TDoA
  Cyberzoo 数据；没有核实到另一团队在这 12 条 TWR 日志上给出 IMU+UWB 轨迹 RMSE。

所以当前可比较文献并不是三组独立复现，而是同一作者团队的 2021→2023→2025
研究链。后续若出现未被当前索引收录、仅在代码仓库或学位论文中使用该数据的工作，
本结论应更新。

## 8. 代码、数据与复核入口

- 复现适配器：`imu-uwb-inekf-state-augment/scripts/estimation/pfeiffer_fair_compare.py`
- 单元测试：`imu-uwb-inekf-state-augment/tests/test_pfeiffer_fair_compare.py`
- 命令：`imu-uwb-inekf-state-augment/run_task06.ps1 -Mode author -AuthorSeeds 10 -Workers 4`
- 作者三协议逐任务结果：`results/task06_author_fair_compare/author_protocol_metrics.csv`
- 公平逐序列表：`results/task06_author_fair_compare/fair_common_protocol_metrics.csv`
- 汇总：`results/task06_author_fair_compare/summary.json`
- 图：`truth_assistance_ablation.png`、`fair_3d_comparison.png`、
  `fair_xy_comparison.png`、`representative_protocols.png`

单元测试覆盖真值协议开关、3D/XY/z 指标恒等式和逐序列 LOSO 参数查找。上游作者
仓库经 Git 检查仍固定在 `955360a...` 且工作树为空。所有正式图片以原分辨率逐张
检查：轨迹连续、XY 形状合理，退化集中在 z；没有 NaN、错位、裁切或隐藏失败。

## 9. 主要来源

1. 数据集与代码说明：<https://research.tudelft.nl/en/datasets/data-underlying-the-publication-a-computationally-efficient-movin/>
2. 数据 DOI：<https://doi.org/10.4121/14827680.v1>
3. Pfeiffer et al. 2021：<https://doi.org/10.1109/LRA.2021.3095519>
4. 2021 作者代码：<https://github.com/SUPfeiffer/uwb-simulator>
5. 2021 批处理固定提交：<https://github.com/SUPfeiffer/uwb-simulator/blob/955360a8736cae3600a1fbed530a9ec19a2ad724/publication/headless_Nanchors.py>
6. 2021 回放真值路径：<https://github.com/SUPfeiffer/uwb-simulator/blob/b43a7f6d9033459016e3ab8c52d42593d84f210a/UWBsim/airframe/drone.py>
7. Pfeiffer et al. 2023：<https://www.imavs.org/papers/2023/10.pdf>
8. Pfeiffer et al. 2025：<https://doi.org/10.1177/17568293251340100>

