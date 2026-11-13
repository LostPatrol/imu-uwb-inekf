<!-- Independent full inertial-graph reproduction and its scientific limits. -->
# Jiang 2024：10 秒联合图初始化的独立复现

来源：[ICRA 正文](https://faculty.cc.gatech.edu/~dhekne/Robust_Indoor_Localization_with_Ranging_IMU_Fusion_ICRA_2024.pdf)、[arXiv 全文](https://arxiv.org/html/2309.08803v1)、[第一作者出版页](https://fanjiang.me/)。检索了论文、作者站、ProfFan 公开仓库及 D-iSAM2/题名搜索，没有找到本方法的公开作者融合实现。2026-11-13 查询 PyPI 的 GTSAM 4.2/4.3 文件，均无 Windows 轮子，本项目环境也未装 GTSAM。因此这里**不是作者代码或 D-iSAM2 的完整复现**，不能将本目录结果直接标作已发表系统的性能。

正文 §V 明确先积累 10 秒 range/IMU 图并做 batch optimization，再初始化 iSAM2。故本实现 10 秒前没有发布状态，10 秒优化完成后只发布当前状态，不把回溯平滑轨迹当早期在线输出。随后到 15 秒采用每秒累积全图重新优化，是明确的 batch continuation 近似，不是作者的增量 Bayes tree/trust-region 调度。

`factor_graph.py` 对每个不同的测距时刻设置 15 维状态：SO(3) 姿态、世界速度和位置、三轴 gyro/accel bias；同时间多基站共享状态，但每一个共同测距包各有因子，没有抽稀。相邻状态使用 IMU 预积分旋转/速度/位置残差，传播完整 9×9 协方差及六偏置一阶修正。bias 用一阶 Gauss–Markov 因子；基站坐标固定为测绘输入。这相当于作者的有限方差 anchor prior 被固定坐标近似。支持已知 body lever。全部位置、姿态和偏置联合优化，不是纯测距 WWLS 或自由多项式轨迹。

测距因子采用等权 max-mixture：LOS Gaussian 与正残差 half-Cauchy，残差方向为 measured−predicted。优化目标是分支最大密度的负对数，不能用于解释为经过标定的后验概率。论文未提供分支权重、切换细节及 gamma 数值；这里冻结 gamma=2×共同 sigma。正文仅给出半 Cauchy 密度、建议用 Gaussian IQR 或数据估计 gamma，没有唯一可还原的数值模型。

论文未公开的其他参数独立固定为：accel/gyro 白噪声密度 0.10 m/s²/√Hz、0.01 rad/s/√Hz；bias drive 为 0.01 m/s²/√s、0.0001 rad/s/√s，相关时间 1000 s。bias prior 标准差为 accel 0.1 m/s²、gyro 0.01 rad/s。使用共同外部静态校准后的测距及其 sigma，不重复修正。没有从测试真值调参，没有随机种子或多初值择优。

原生启动层仅以 `(p,v,yaw)=(0,0,0)` 和共同 IMU tilt/bias 为单初值，图中只有 factory/static bias prior，没有 pose/速度/高度先验。共同 tilt 只是姿态初值。论文没有公开 batch 初值，故这也是本实现假设。

错误先验层在同一图上另外加入首状态的位置/物理航向 prior。位置/yaw 均来自共同 `priors.csv`，位置标准差及 yaw 标准差原样使用，不增加速度或 roll/pitch 观测。这是**共同 framework prior 适配层**，不能混作作者原生启动。航向为 `atan2(R[1,0],R[0,0])`，先验残差取 wrapped 航向差；bias 先验不变。宽先验和紧错误先验均保留，不用测距重新覆盖初值。

求解使用自有稀疏 LM、解析稀疏 Jacobian、稀疏直接法。旋转严格在世界左切空间求步，使用 `Exp(delta_theta) R` retraction，存回主值 rotation vector；其余状态加性更新。最多 80 步，每步最多 14 个阻尼尝试、相对目标改善容限 1e-7。保留相同局部 retraction 下的数值彩色 Jacobian 作为代数参考；解析导数独立用中央差分核验，三种 canonical yaw（0、π、π+0.2），带 lever、bias 与两种测距分支的最大 scaled 差为 1.03e-8，主值重写前后目标差三例均为零，±π 与多圈旋转往返一致。`validate_jacobian.py` 可重跑。触顶和无下降步保留最终有限结果并标出 termination，不能统称收敛。该数值解法及一阶偏置修正不等同 GTSAM 的全部预积分实现，也不包含作者的 D-iSAM2。先验和噪声数值不作结果导向调优。

输入接口仅读取 `streams/<episode>/{imu,radio,anchors,priors}.csv` 与 `meta.json`，不打开评分目录，也不接收真值 pose 或额外高度。IMU 是 SI body vectors，R0 为 sensor-only tilt；锚点/测距必须同坐标系。TDoA 不能冒充绝对测距喂入这个模型。真值与基站不在同一固定坐标系的日志不能通过轨迹对齐变成严格绝对定位证据。

从项目根运行：

```powershell
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\acquisition_challenge\jiang\smoke_check.py
& .\.venv\Scripts\python.exe -u .\imu-uwb-inekf\experiments\acquisition_challenge\jiang\run.py --streams .\agent\codex\acquisition_challenge_20261113\streams --output .\imu-uwb-inekf\experiments\acquisition_challenge\jiang\evidence --workers 5
```

每进程 BLAS=1，主检使用六进程；按 episode 复用不可变预积分。持续压力测试先用六进程完成预声明首/尾子集；16 进程扩容出现主机提交/换页压力后终止，保留已完成的 317 case，改用既有 `split_handoff.cpu_replay.replay_pool` 的八个单线程、P-core-first 进程继续，并将每 worker 缓存限制为当前 episode 的六张图。旧 case 不重算，audit 拼接而非覆盖，始终只有一个 CSV 写者。转移清单在 `continuation_resume_transition*.json`。逐 case 写 checkpoint，失败/触顶不删除。`native.csv`/`stress.csv` 汇总每次发布时间、估计状态时刻、实际优化 wall time、终止原因和状态。`wall_s` 包含图构建加求解；共享图构建成本也分摊记录到每个独立 stress case，不把预积分当作零成本。

完整先验矩阵可用 `stress_parallel.py --streams ... --output ... --episode <episode>`，将独立 case 分配给多个进程，每进程缓存只读预积分，父进程串行 checkpoint，避免同一 CSV 并发写入。支持重复 `--episode`。不根据结果好坏选取 case。

`continue_stress.py` 使用相同参数与初始先验，将每个已注册 10 秒解继续至 11、12、13、14、15 秒；每次只读取当前时刻前的测距/IMU，图保留全部过去因子。历史状态只用于优化，不重新发布。首状态位置/yaw prior 始终是原始外部 prior，不以优化后的首状态替换。warm start 只复制精确同时间节点，再从最后匹配状态用有偏置修正的 IMU 传播。首轮未保存全状态的 ISAS 会按同配置重算 10 秒状态，并逐 case 核验与已注册端点的差；`continuation_resume_audit.csv` 保留差值。

`states10` 是用于继续计算的全状态缓存，`states15` 是终点与原始 prior 的审计缓存。`stress_10to15.csv` 含完整压力因果轨迹。各时刻 `wall_s` 含完整构图成本及当次求解；研究运行包含 Numba 冷编译及同机其他实验争用，不对应作者 C++ iSAM2 或嵌入式实时延迟。取得传感器到 10 秒的数据不等于 10 秒墙钟就能发布结果；首次有效必须另外计入计算等待。

统一执行入口为 `final_pipeline.py --streams ... --output ... --workers 6`。它覆盖 16 个 native、8×96 个压力图、全部压力继续至 15 秒，并在同一局部坐标和固定设置下对全部 16 个 native 首发做数值/解析 Jacobian 对照。`timing_sample.py` 的计时样本在看结果前固定为每个 recovery 的 prior 索引 0/31/63/95；每例重新构图，记录 warm prepare/solve/total，JIT warmup 单列。

`scratch_additive_rotation_20261113_2105` 保存早期全局 rotation-vector 加性求步实现的全部已有输出。该表示可能在多圈坐标出现数值病态，已作为实现缺陷统一修复，不能以其结果批评作者方法。其他 `preliminary_*` 目录属于共同输入更新前的试跑，不进入最终主结果。

`smoke_check.py` 是有必要的完整因子代数检查：合成 10 秒移动 IMU＋200 条 range，位置初值误差 5 m，既无 pose prior 也无真值输入求解器。误差仅在求解完成后与合成定义核验；它不代替真实数据测试。
