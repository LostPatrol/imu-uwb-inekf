<!-- Cold-start and perturbed-prior single-tag navigation benchmark. -->
# 单标签定位从启动到首次可用的位置

本基准从共同的原始测距和机体系 IMU 出发，检验未知位置启动和错误先验恢复。主对照输出测绘基站坐标系内的 IMU 原点。桌面离线按包回放分别评价最早何时能报告可用 XY，以及首次可用后精度能否维持。

本次结果见[中文实验总结](./RESULTS_zh.md)、[逐条件指标](./evidence/acquisition_scores.csv)和[两场地逐记录启动图](./evidence/cold_acquisition.png)。

## 场地、共同输入与干预

| 数据 | 未知启动与恢复的独立记录 | 位置与航向参考 |
|---|---|---|
| Delft，单标签 TWR | 6 条预先指定的飞行，每条各取开头和一个中点。另 6 条训练飞行提供距离常量与噪声尺度 | OptiTrack 位置。未记录独立姿态，航向干预相对于重力对齐后的零航向 |
| ISAS，单标签绝对距离 ToA | Walk1 标定，Walk2、Walk3 各取开头和一个中点 | Walk1 拟合 tracker→IMU、VIVE→测距系和时差，冻结后用于测试记录。半段训练得到的世界旋转相差约 0.4–1.7°，有效时间偏差约 36 ms |
| Toronto UTIL，单标签 TDoA | 四个基站布置各一条 trial4，保留原始带符号差距、杆臂和完整姿态 | Vicon。Oursland 与 SFUISE 的补充比较单列，详见 toronto_heading 和 toronto_sfuise |

两组绝对测距测试释放前给紧邻的 1.5 s 重力对齐前缀，清除此前位置/速度估计。在 15 s 内提供所有已到测距，IMU 保持原始机体系。外参和测距校准对同一记录内所有方法相同。SFUISE 使用自己的噪声权重，Pfeiffer MHE 的作者核忽略逐包标准差。位置真值较晚开始的 Delft 飞行全部保留，只在真值支持的时段评分。

恢复时点按传感器时钟确定。基准单独生成首状态的水平中心
\[
\bar p_{0,xy}=p^{\rm ref}_{0,xy}+m d,\qquad
\bar\psi_0=\operatorname{wrap}(\psi_{\rm ref}+\delta).
\]
这里 \(m\) 取 1、5、10、20 m，\(d\) 取水平四个坐标轴方向，\(\delta\) 取 0°、90°、180°。每个中心配宽/紧两种标准差，分别为 20 m/120° 和 0.10 m/5°。每条恢复记录共 96 种条件。Delft 的 \(\psi_{\rm ref}\) 是传感器规范零值。两组绝对测距的先验 Z 用测绘基站高度均值，速度和倾角均由允许的输入确定。估计器求解只使用允许的中心和标准差，各种扰动的传感器字节相同。

## 从首次发布到首次可用

记 \(a_k\) 为读取必要样本的时刻，\(s_k\) 为报告状态对应的物理时刻，\(c_k\) 为本次实际测得的增量计算耗时。分别以 \(a_k\) 和串行队列 \(b_k=\max(a_k,b_{k-1})+c_k\) 评价。前者衡量需要观察多久，后者进一步计入本次机器负载。完整冷启动输入跨度等于释放后时长加 1.5 s。Jiang 十秒批图的回溯状态只在批求解完成后可用。SFUISE 发布优化后的控制点估计。已测绘的基站开场即提供，ROS 采集基站列表的等待另计。

在网格 \(t\) 取当时已发布的最后一个状态，计算 \(\|\hat p_{xy}-p^{\rm ref}_{xy}(t)\|\)。网格间隔 0.5 s，连续三个样点不超过 0.5 m 时，第三个样点为确认时间。另存首个样点、0.25/1 m 敏感性、首发状态的 XY 误差、2/10/15 s 定点 XY/航向/垂向误差。真值仅在事后判定持续性。估计器中止后停止计有效输出。扰动计数反映条件覆盖，独立样本仍是录制次数。

## 方法与对照口径

| 实现 | 启动 | 错误先验作用于哪里 |
|---|---|---|
| SFUISE 作者数值核移植 | 滑窗增长阶段联合估计导航系至 UWB 系的刚体变换、重力与样条状态 | 调整自由变换的数值初值。原实现无协方差接口，宽、紧两列引用同一次运行。Windows 适配对首个控制点的无效第二元素访问显式重复端点，验证与合法重复控制点的端点运动学量相同。ROS 和可视化由因果包输入输出代替 |
| Jiang 等十秒初始化的独立实现 | 十五维状态含姿态、速度、位置和两类偏置。联合优化 IMU 预积分及非对称距离残差。同一时刻多锚共享运动状态，保留每个距离因子 | 为首状态的位置和物理航向增加高斯因子。十秒后每秒累积图重新优化。原文未公开的初值、噪声和优化常量随实现存档，作者 D-iSAM2 的真实运行时间未知 |
| 本文旋转体轴加计偏置窗口、原仿射窗口 | 平移候选和一个 lifted 航向候选，原始距离做一次非线性精化。冷启动 2 s 首发，此后每 0.5 s 更新，窗最长 8 s | 原生重启独立于错误先验。另测将约束施加于重建的物理首状态，所有条件用同一个传感器候选。恢复主快照用 [0,2] 和 [0,10]，持续轨迹单列为 [0,t] 增长至 15 s，先验始终绑定 t=0 |
| Pfeiffer 作者 EKF/MHE，Delft | 零种子，推力由体轴 z 加速度和质量构造。保留四旋翼推力/阻力和姿态模型，关闭需要真值的额外观测 | EKF 设置初始协方差并保留内部上限，MHE 只能改位置种子，每段 16 个不同位置。手持 ISAS 单列 |

论文窗口位置模型为 \(p_i=p_e+\tau_i v_e+R_z(\psi)(d_i-B_i\delta b_a)\)。有标签杆臂时，距离预测加上该时刻姿态作用后的杆臂。外部先验对应
\[
\|p_0-\bar p_0\|^2_{\Sigma_p^{-1}}+
\operatorname{wrap}(\psi_0-\bar\psi_0)^2/\sigma_\psi^2 .
\]
弱激励下，多锚可定位，航向敏感度在消除位置、速度和偏置方向后可以很小。代码中的纯数据航向信息单列，先验曲率不计为运动证据。ObjectTrack 单标签消融、Theodosis 观测器的输入假设及 Toronto 作者核对照见各自目录。

## 重放

在项目根目录使用项目 Python 环境，先执行 prepare.py 导出 Delft，按 isas_data/README.md 导出 ISAS。streams 供估计器使用，scoring 单独存放参考。各实现的 README 给出回放入口和作者版本。score.py 与 summarize.py 生成逐条件长表、配对差、按场地汇总与逐录制冷启动图。共同协议见 protocol.json，实验总表位于 evidence。

    $py = '.\.venv\Scripts\python.exe'
    $src = 'imu-uwb-inekf/experiments/acquisition_challenge'
    & $py "$src/prepare.py" --out 'agent/codex/acquisition_challenge_20261113'
    # Run the individual estimators before scoring.
    & $py "$src/validate_benchmark.py"
    & $py "$src/score.py"
    & $py "$src/summarize.py"
    & $py "$src/consolidate.py"
    & $py "$src/time_native.py"
