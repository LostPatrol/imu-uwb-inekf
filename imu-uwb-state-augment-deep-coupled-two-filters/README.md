# IMU and Ultrawideband Sensor Fusion with Invariant Extended Kalman Filter

## 增广捕获/重置 + 原始 TWR 联合 InEKF 自适应跟踪

本目录是在 `imu-uwb-inekf-state-augment` 基础上独立建立的改进版本。
它保留原增广法的大初差捕获能力，同时新增一个使用原始 TWR 的 15 维
联合 InEKF 跟踪支路。原始距离更新通过同一协方差中的交叉项修正
姿态、速度、位置和 IMU 偏置，不再是“UWB 只更新另一个平移 KF”的纯松耦合。

核心不是无条件互相注入，而是有保护的两滤波器状态机：

- `capture`：原 `LiftedTWRInEKF`，只负责初始化、大初差/失锁捕获、
  健康监视和 tracker 离散重置；它始终后台运行，但健康 tracker
  权重达到 1 后不再参与位置/速度输出；
- `tracker`：原始 TWR `TWRInEKF`，维护
  `[theta,v,p,b_g,b_a]` 的 15 维联合协方差；
- 前 20 s 用于大初差捕获与交叉协方差成熟；满足距离残差、拒绝率和
  两支路一致性条件后才启用跟踪输出；
- 跟踪失效时自动退回捕获支路，并可用膨胀后的分块协方差离散重置跟踪器；
- 切入跟踪时 tracker 权重为 `0.3`；只有持续健康证据通过低通、滞回
  和保持时间后，才在 5 s 内平滑升至 `1.0`。健康度下降时权重快速
  回到 `0.3`；5 s 宽限后若平滑健康度仍低于 `0.27`，则继续平滑
  退到 `0`，由 capture 独立输出但 tracker 仍在后台积累证据；
  确认失锁则立即回到 capture 模式。
  这不是连续伪量测互灌，因此不会把共享的 IMU/UWB 数据当成独立信息重复计数。

所有真实和合成结果同时保留五种方法：

- 普通六维原始距离 EKF；
- 原 Task06 普通 TWR-InEKF（立即启用 NIS 门限、没有重置）；
- 原增广松耦合支路；
- 新增的原始 TWR 联合 InEKF tracker；
- 最终健康度驱动的自适应两滤波器输出。

普通 EKF 使用与增广支路完全相同的世界系加速度；普通 InEKF 使用同一
原始 IMU/TWR。测试会把这两条新增曲线逐样本与原 Task06 基线比较，
避免因重写对照方法而改变结果。

### TU Delft 共同传感器时延

独立标定表明，CSV 中同一行的 IMU 与 TWR 已经彼此配对，但二者相对
OptiTrack 的日志时间共同滞后约 `0.20 s`。因此真实回放使用

\[
t_k^{\rm acquisition}=t_k^{\rm log}-\tau_s,\qquad \tau_s=0.20~{\rm s},
\]

并在 \(t_k^{\rm acquisition}\) 对 OptiTrack 插值。实现不会只移动 UWB，
也不会破坏 IMU--UWB 的相对时序。`load_real_log` 的通用默认值仍为零；
本目录的深耦合真实数据入口显式启用 `0.20 s`，也可通过
`-SensorDelaySeconds` 覆盖。

`calibrate` 模式同时检查两类彼此独立的证据：逐锚去均值后的 TWR
残差，以及旋转不变的 IMU 比力模长与 OptiTrack 二阶导数的对齐。
指定的一条标定轨迹给出 TWR/IMU 最优值 `0.21/0.22 s`；按可辨识精度
取 `0.20 s` 后，其余 11 条日志的两类指标均改善。在线系统若拿不到
真实硬件采集时间，应使用带状态/IMU 缓冲和重传播的固定延迟更新；
离线把真值插值到采集时刻不能替代实时 OOSM 实现。

论文表中的“联合 tracker”是上述 `tracker` 支路自身的原始位置/速度
输出，不是普通 InEKF，也不是最终自适应输出；后者才按健康度权重融合
`capture` 与 `tracker`。原先“同行即时生效版本”的说法现已明确为
“零延时假设”：同一 CSV 行的 IMU/TWR 仍保持配对，但直接令
`t_acquisition=t_log`。为避免算法版本混杂，时间戳消融使用当前同一版
deep 参数分别取 `0.00 s` 和 `0.20 s`；24 次零延时结果位于
`results/deep_coupled_zero_delay_current`，严格配对汇总位于
`results/deep_coupled_time_alignment_self_comparison`。

### 合成大初差端到端蒙特卡洛

旧版合成试验总长和强制 capture 驻留时间同为 `20 s`，状态机要求
`elapsed > 20 s` 后才允许切换，因此所谓“自适应输出”实际满足
`w(t)≡0`，与 capture 逐样本完全相同。当前试验改为：

- 前 `20 s` 保留原 Task06 单锚点持续激励和大初差捕获；
- 后 `25 s` 使用固定八锚点房间几何，总距离事件率仍为 `20 Hz`；
- IMU 含固定残余加速度偏置
  `[0.080,-0.056,0.040] m/s²`，滤波器初始偏置估计为零、标准差
  `0.20 m/s²`，使联合协方差的偏置校正通路可以被检验；
- 只有至少四个近期不同锚点，且 tracker 的量测前距离 RMSE 比 capture
  低至少 `0.40 m` 时，才允许“决定性量测优势”覆盖分支位置分歧；
  单锚点拟合不能触发该覆盖；
- 验收强制要求 20/20 次切入、末 5 s 输出与 capture 不同并达到过
  `w=1`，同时检查成功数、末段位置逐次非退化和四项核心指标中位数。

逐次激活证据写入 `synthetic/adaptive_activation.csv`。因此即使以后误把
试验缩短为 capture-only，验收也不会再因两条相同轨迹而平凡通过。

运行入口会强制使用
`D:\conda-envs\imu-uwb-inekf`：

```powershell
# 28 项数值、状态机、时间对齐、基线等价和真实回归测试
.\run_deep_coupled.ps1 -Mode test

# 独立扫描共同 IMU+TWR 时延并生成 JSON/CSV/诊断图
.\run_deep_coupled.ps1 -Mode calibrate

# 30 s 真实大初差回放 + 1 次 45 s 合成端到端检查
.\run_deep_coupled.ps1 -Mode quick

# 12 条真实日志 × 2 种初差 + 20 次合成 Monte Carlo
.\run_deep_coupled.ps1 -Mode full

# 显式覆盖共同传感器时延
.\run_deep_coupled.ps1 -Mode full -SensorDelaySeconds 0.20

# 对两份同算法真实批测生成时间戳配对汇总与差值图
D:\conda-envs\imu-uwb-inekf\python.exe scripts\estimation\compare_deep_time_alignment.py `
  --zero-delay results\deep_coupled_zero_delay_current\real\paired_metrics.csv `
  --corrected results\deep_coupled_delay_full\real\paired_metrics.csv `
  --output results\deep_coupled_time_alignment_self_comparison

# 显式设置合成端到端时长；小于 30 s 会被拒绝
.\run_deep_coupled.ps1 -Mode synthetic -SyntheticDuration 45
```

正式结果输出到 `results\deep_coupled_delay_full`，时延标定输出到
`results\time_alignment_calibration`。配对批测把同一个估计器内部、
从未被 tracker 覆盖的 `capture` 支路作为松耦合基线，因此输入、时序和随机性
完全相同。`acceptance_summary.json` 会自动检查全程/末 5 s 的位置和速度
四项指标是否存在退化。
合成验收还要求状态机被非平凡激活，不能再用 `w≡0` 获得通过结果。

边界必须说明：两个支路使用相同 IMU 和 UWB，却没有维护滤波器间互协方差，
所以没有把二者协方差当作独立信息融合；输出加权也不提供严格一致的融合协方差。
真实数据只有位置真值，姿态和偏置改善只能从滤波结构推断，尚不能用真值验证。
当前成果是经过真实数据回归保护的工程改进，不是完整有偏 IMU--UWB 系统的
UGES/AGAS 证明。公式、实测结果与限制见项目根目录的报告和
`tex/deep_coupled_two_filter_twr.pdf`。

This repository presents an Invariant Extended Kalman Filter (InEKF) to perform sensor fusion of IMU and UWB data.  The dataset used is the UTIL dataset provided by the University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute)/ [UofT Robotics Institute](https://robotics.utoronto.ca/) and comparisons are made with the reference Error-State Kalman Filter (ESKF).

## Conda Environment Installation

The dependencies are managed via the `conda` package manager with packages from `conda-forge`.  Install the latest release of [`miniforge`](https://github.com/conda-forge/miniforge/releases/lates

On Linux or macOS run the following commands to install and activate the `conda` development environment:

    ./conda/setup-environment.sh
    conda activate imu-uwb-inekf

## Dataset Installation

To install the dataset run the following commands:

    curl -L https://github.com/utiasDSL/util-uwb-dataset/releases/download/dataset-v1.0/dataset.7z > dataset.7z
    tar xvf dataset.7z

## Running the Estimator Scripts

To run the estimators and generate plots run the following commands:

    cd scripts/estimation/
    python3 main.py -i ../../dataset/flight-dataset/survey-results/anchor_const1.npz ../../dataset/flight-dataset/csv-data/const1/const1-trial1-tdoa2.csv

There are 4 constellation sets (const1, const2, const3, and const4), with up to 7 trials each in both TDoA 2 and TDoA 3 (const#-trial#-tdoa#).

# Data Parsing Scripts for UTIL: Ultra-wideband Dataset

Detailed information and instructions: [https://utiasdsl.github.io/util-uwb-dataset/](https://utiasdsl.github.io/util-uwb-dataset/.)

The dataset paper:  [https://arxiv.org/pdf/2203.14471.pdf](https://arxiv.org/pdf/2203.14471.pdf)

## Dataset Usage

* Kailai Li, Ziyu Cao, and Uwe D. Hanebeck. "Continuous-Time Ultra-Wideband-Inertial Fusion." arXiv preprint arXiv:2301.09033,(2023). [Paper Link](https://arxiv.org/pdf/2301.09033.pdf), [Open Source Code](https://github.com/KIT-ISAS/SFUISE).

* Wenda Zhao, Abhishek Goudar, and Angela P. Schoellig. "Finding the right place: Sensor placement for uwb time difference of arrival localization in cluttered indoor environments." IEEE Robotics and Automation Letters 7, no. 3 (2022): 6075-6082. [Paper Link](https://ieeexplore.ieee.org/document/9750886).

## Credits

This dataset was the work of [Wenda Zhao](https://williamwenda.github.io/), [Abhishek Goudar](https://www.linkedin.com/in/abhishek-goudar-47b46090/), [Xinyuan Qiao](https://www.linkedin.com/in/xinyuan-sam-qiao-8b15ba17a/), and [Angela P. Schoellig](https://www.dynsyslab.org/prof-angela-schoellig/). If you use the data provided by this website in your own work, please use the following citation:
```
@INPROCEEDINGS{zhao2022uwbData,
      title={UTIL: An Ultra-wideband Time-difference-of-arrival Indoor Localization Dataset},
      author={Wenda Zhao and Abhishek Goudar and Xinyuan Qiao and Angela P. Schoellig},
      booktitle={International Journal of Robotics Research (IJRR)},
      year={2022},
      volume={},
      number={},
      pages={},
      doi={}
}
```
-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute)/ [UofT Robotics Institute](https://robotics.utoronto.ca/)
