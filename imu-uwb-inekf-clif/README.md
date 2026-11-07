# IMU and Ultrawideband Sensor Fusion with Invariant Extended Kalman Filter

## Task 06：TWR 平方距离状态增广

本目录从论文作者备份独立复制，只在本目录实现任务 06。主要新增：

- 普通单距离 `TWRInEKF` 基线（真实批测和合成蒙特卡洛均包含）；
- 提升状态
  `chi=[q,s,h,p_x,p_y,p_z,v_x,v_y,v_z,1]`；
- 去偏半平方距离与距离相关方差；
- 可选的逐锚距离偏差/稳健噪声尺度；
- 3.5σ 硬拒绝与 2σ Huber 连续降权（256 次初始化预热）；
- 只依赖锚点的常观测行，不在 UWB 更新处计算位置雅可比；
- TU Delft 真实 TWR 回放、Theodosis 单锚点大初差实验、单元测试与图片输出。

请使用项目指定的 PowerShell 入口（它会强制使用
`D:\conda-envs\imu-uwb-inekf`）：

```powershell
# 下载并校验约 10 MB 的 TU Delft 数据集
.\run_task06.ps1 -Mode download

# 22 项数值/数据加载/作者公平协议测试
.\run_task06.ps1 -Mode test

# 30 s 真实 TWR + 20 s 单锚点大初差对比
.\run_task06.ps1 -Mode quick

# 全部真实日志的末 5 s 因果消融
.\run_task06.ps1 -Mode diagnose

# 12 条真实 TWR 日志 × 2 种初差 × 3 种方法，以及 20 次合成 Monte Carlo
.\run_task06.ps1 -Mode full

# 原任务06全部对比 + LOSO逐锚/鲁棒门控改进 + 组件消融 + 3D/XY双口径
.\run_task06.ps1 -Mode improve

# 2021作者源码原协议 + 去真值辅助 + 同LOSO标定的公平3D/XY对比
.\run_task06.ps1 -Mode author -AuthorSeeds 10 -Workers 4
```

真实数据只提供 OptiTrack 位置而没有姿态真值。当前第一阶段结构是
“IMU-InEKF 姿态/偏置传播 → 世界加速度 → 提升平移观测器”；UWB 不回馈姿态和偏置，
所以它不是统一的 group-affine InEKF，也不能直接继承 `eq_ob.pdf` 或 Theodosis
论文的完整 UGES/AGAS 结论。详细边界、公式和结果见项目根目录的任务报告与
`tex/task06_state_augmentation_twr.pdf`。

`diagnose` 会比较普通距离 EKF、六维平方距离 EKF、独立 `q/s/h` 增广以及
每次更新后投影回物理流形的诊断版本，并扫描有效加速度过程噪声。投影版本
只用于解释稳态误差来源，会牺牲独立辅助状态带来的大初差线性观测机制，
不是默认算法的隐式替换。

`improve` 对每条待测日志只使用其余 11 条日志的 OptiTrack 残差，按锚点
冻结中位偏差和 `1.4826*MAD` 噪声尺度；待测日志真值仅用于事后评分，
不会在滤波过程中注入位置、高度或在线噪声。该协议是留一序列外的离线标定，
部署时仍需要独立标定数据。结果位于
`results/task06_per_anchor_robust`，详细结论见项目根目录新增报告和
`tex/task06_per_anchor_robust.pdf`。

`author` 固定使用根目录 `third_party/pfeiffer_uwb_simulator_2021` 的作者MIT源码
提交 `955360a8736cae3600a1fbed530a9ec19a2ad724`，不修改上游文件。运行适配器
分别执行论文原始真值辅助协议、关闭OptiTrack高度和真值残差噪声的raw协议，
以及与当前Task06共享LOSO逐锚标定/首帧位置/全程计分的公平协议。三组均报告
3D、XY、z和成功率。2023/2025的LS/Newton/SVOR实现未公开，因此不会把依据
论文公式自行猜测的实现伪装成作者源码复现。

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
