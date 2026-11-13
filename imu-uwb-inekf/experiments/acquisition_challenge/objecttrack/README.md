<!-- Evidence audit for ObjectTrack; this directory contains no reproduced estimator. -->
# ObjectTrack 文献与公开实现核查

核查日期：2026-11-13。状态：**仅文献核查，未复现，未运行本项目数据，未产生同包 baseline 成绩**。

## 论文身份

Fan Jiang、Frank Dellaert、Ashutosh Dhekne，*ObjectTrack: 6DoF Object Tracking Through UWB-IMU Fusion*，IPIN 2025，1–6页。
[作者托管原文](https://faculty.cc.gatech.edu/~dhekne/ObjectTrack_IPIN2025.pdf)；
[DOI](https://doi.org/10.1109/IPIN66788.2025.11213306)；
[Crossref 注册元数据](https://api.crossref.org/works/10.1109%2FIPIN66788.2025.11213306)（本次实际核验标题和三位作者）。

## 可核验结论及适用边界

| 核查项 | 原文证据与边界 |
| --- | --- |
| 单标签+IMU | V-A-4 / Fig. 12 确有该消融：90分位 APE 相对三标签约增加1 cm。没有把曲线读成精确逐序列数值。 |
| 启动 | V-A-4 称单标签需估计初始位姿、IMU偏置，初始化较久、随后收敛；没有给等待秒数或统一成功阈值。 |
| 未知位置/航向 | 上述描述与估计初态相符；但未说明随机初始位置/航向分布、先验强度或独立重复冷启动协议。不能由此认定满足本项目的未知位置/航向挑战。 |
| 真值参与 | III-C 的安装标定使用 ARKit/OptiTrack 外部轨迹；在线图使用标定后的锚点、天线偏置和标签杆臂。V 的评估还使用全站仪及 OptiTrack。原文未充分披露单标签线上初值具体如何设置，不能断言使用或完全不使用真值初值。 |
| 指标归属 | V-A-1 的多数动作90分位 APE <10 cm 属三标签+IMU后处理；V-A-2 的12–17 cm 属三标签实时；二者均不能作为单标签实时成绩。 |
| 延迟 | III-B-3 的约10 ms 是在线处理延迟，不是冷启动等待。 |
| 数据 | V-A 写明未来将开放原始数据/轨迹；该 PDF 未提供可下载数据集地址。本次未核实可用的正式发布包。 |

来源均为上述作者原文的指定节/图；这是证据摘要，不是实验结果。

## 公开代码核查

已查作者[主页](https://fanjiang.me/)、[Dhekne 论文列表](https://faculty.cc.gatech.edu/~dhekne/)、作者主页链接的 [ProfFan GitHub](https://github.com/ProfFan)，以及论文题名与代码/数据关键词。**未找到可核验为 ObjectTrack 论文融合估计器及单标签冷启动驱动的公开作者实现**。这表示本次核查未取得，并不证明不存在；GitHub 仓库列表分页检查还遇到匿名 API 限流，不能宣称完成全部仓库穷举。

已核实的关联开源资产须区分：

- [magic-loc-rs](https://github.com/ProfFan/magic-loc-rs)：README 自述为 muFiu 平台固件，列出硬件、协议和 central 仓库。
- [magic-loc-central](https://github.com/ProfFan/magic-loc-central)：README 自述为串口解码及 ZeroMQ 消息转发；把融合列作下游节点。这不是可直接运行的论文融合估计器。
- [uFiμ HotMobile 2024 作者原文](https://faculty.cc.gatech.edu/~dhekne/HotMobile_2024_uFimu.pdf)：公开的是固件/驱动，链接 magic-loc-rs。不能把该链接标为 ObjectTrack 或下述 ICRA 论文完整实现。

因此本目录没有伪造移植驱动。形成可信同包 baseline 尚需：可追溯的作者融合源码与许可、完整初态/噪声/标定配置、明确单标签在线输出和时间戳接口，再接入本项目相同输入、初始化信息限制及评价流程。自行从方程实现只能标为独立实现，不能冒称作者复现。

## 与 Jiang ICRA 2024 区分

另一论文为 *Robust Indoor Localization with Ranging-IMU Fusion*，作者 Fan Jiang、David Caruso、Ashutosh Dhekne、Qi Qu、Jakob Julian Engel、Jing Dong。
[作者原文](https://faculty.cc.gatech.edu/~dhekne/Robust_Indoor_Localization_with_Ranging_IMU_Fusion_ICRA_2024.pdf)、
[arXiv:2309.08803](https://arxiv.org/abs/2309.08803)、
[IEEE/DOI](https://doi.org/10.1109/ICRA57147.2024.10611274)。

其 V 节明确先累计10秒测距/IMU联合图，再批优化初始化 iSAM2；这10秒不能移植成 ObjectTrack 的启动时间。实现采用 C++/GTSAM，评估为 Project Aria IMU、硬件同步、已在真值坐标系定位的锚点。所查作者原文、arXiv 和作者主页未给可核验的论文完整官方源码链接；上述 uFiμ 固件不能替代其非对称噪声模型和 D-iSAM2 实现。
