# Oursland 原作者 InEKF 外部对照

本目录直接按文件导入 `imu-uwb-inekf - backup/scripts/estimation/inekf.py`，不修改作者文件，不把本项目的 15 状态核称为原文实现。协议在 `protocol.json`。所有定位误差均为测绘世界坐标中的绝对误差，无真值对齐。

已完成主组结果见 `RESULTS.md`：原 P 下当前方法没有总体胜出，作者航向明显更好；不能使用宽先验或 100 Hz 适配压力组代替该结论。

## 四种口径必须分开

1. **主外部对照：native queue / sensor only / original prior。** const1-trial4-tdoa2、const2-trial4-tdoa3、const3-trial4-tdoa2、const4-trial4-tdoa3-traj1，名义与 180° 两种初始世界航向扰动，普通/10–20 s 断距。作者原类和当前递推核使用相同原始加速度计时间、插值陀螺、作者事件队列、相同传感器初值、原 main 的初始不变协方差 P 及 TDoA 方差 0.13。弱冷启动、动态交接、断距恢复分别在 2/16/26 s 后评分。当前 retain/full/evidence/bank 的前缀、shadow、后续递推均改用此原生队列，交接候选重新计算；候选自身的冻结 100 Hz 窗口积分模型和 0.14/0.22 测量方差保留。当前交接政策将原 P 经共同初始 X 拉回物理坐标，再用于初始映射和共同交接重置，因此初始不变 P 与作者逐元素相同。主结论只适用这四个预定录制。
2. **原 main 健康正对照：original truth。** 第一真值位置/姿态、零速度/偏置、原始 P、原始噪声、原始原生队列；这是有真值初值的健康检查，不能混入无真值性能统计。本次仅截取 45 s，不能声称复现论文完整轨迹统计。
3. **原生队列宽先验敏感性。** `native_paired` / `native_policies` 的同一四录制实验使用共同的较宽物理先验。它单独说明更换初始不确定度的效果。观察到的作者偏置索引顺序差异没有经过因果消融，不把它当作误差差异的确定原因，也不取代原先验主结果。
4. **100 Hz 适配压力组。** 原预登记 trial4/5 全部 16 个录制、0/±90/180°，采用共同 100 Hz 递推并批量送入全部原生无线包。发现调度适配问题后暂停，实际完成范围以 `holdout/records.json` 为准，不声称已完成全 16 个录制。作者 `correct` 不刷新 `X_hat/P_hat`，同 IMU 间隔内后包会覆盖前包修正；此适配会放大作者原码在宽初始协方差下的脆弱性。该组只能说明适配限制，不能用其大误差宣称击败原论文。

## 原文和源码核对

| 原文位置 | 可确认内容 | 与本次区别 |
| --- | --- | --- |
| PDF p1，印刷 p154，摘要 | 180° 初始航向误差时 InEKF 航向 RMS 0.08 rad（约 4.6°），参考 ESKF 1.34 rad（约 77°） | 原文数字，既不是本次结果，也不是本次当前方法的对照数字 |
| PDF p5，印刷 p158，§IV、Fig.3 | UTIL、IMU+原生 TDoA、Vicon 参考；按 constellation 画位置/速度/姿态 RMS 箱线图；卡方拒绝无线离群值 | 本次保存逐录制 XY、Z、物理 wrapped yaw 和 SO(3)，不从图上猜精确均值 |
| PDF p5–6，印刷 p158–159，§V、Fig.4–5 | 初始角取 16 个等分方向；const3 手动试验即使真值初值也不理想，因此被排除；yaw std 从 0.01 增到 1 rad，TDoA 方差从 0.05 增到 0.13 | 本次保留预定 const3 录制，任何失败不删除；不按结果换录制 |
| backup `main.py:228–230,255–274` | 初位置 `pos_vicon[0]`，姿态来自第一 Vicon 四元数；速度/偏置零；姿态扰动是 SO(3) 对数向量相加 | 主对照没有真值初值，使用前 0.6 s 无线 CV+重力对齐，扰动为世界 Z 左乘 |
| backup `main.py:200–208,262–268` | 原始 std：姿态 .01/.01/1 rad，速度 .01 m/s，位置 .1 m；后六维 .0001×3、.000001×3；TDoA 方差 .13 | 主对照保留原 P；另列敏感性使用共同物理 std [.08,.08,π,.7×3,.5×3,.02×3,.2×3] |
| backup `main.py:135,281–307` | `TDOA2=True`；每事件查询 `t[k-1]`；`imu_dt=t[k]-t_imu_prev`；同时间戳只取第一包；无每次校正后的 prior 刷新 | 原真值健康组严格保留始终 ring；无真值 paired 按实际 TDoA2/3 协议保留 ring/全对，双方一致 |
| backup `main.py:321–346,428–438` | 用平滑样条插值真值位置及 SO(3) 对数向量分量，再计算分量 RMS | 本次直接位置插值、姿态 SLERP，计算 wrapped Euler yaw 与 SO(3) 相对角，不能将本次角误差与原文原数冒称完全同指标复现 |

作者 `inekf.py` 原样保留的特殊行为包括：负高度时位置 Z 钳至 0、速度清零；原始 NIS 5σ 门；普通 `expm`；`predict` 将 `bias[:3]` 当加计偏置而 `bias[3:]` 当陀螺偏置，误差传播矩阵的后六维顺序与此不一致；连续 `correct` 共享未刷新的 `X_hat/P_hat`。包装仅把共同输入偏置从 gyro-first 转成其 predict 的 accel-first，后续原生更新不修补。

共同无真值初值在 t0+0.6 s 可用：最初 0.2 s 加计均值定重力倾斜、yaw=0；0–0.6 s 原生 TDoA 拟合天线恒速位置/速度并减去杠杆臂；仅当首 0.6 s 的加计模长偏离 1g 均小于 .05g 且陀螺模长均小于 2°/s 时，取平均陀螺为初始 gyro bias，否则 bias=0，acc bias 始终从 0 起。已被该初值消费的无线包不再递推。以上是为同条件比较引入的共同包装，不是作者原 main 的真值初值流程。

原生断距组保留双方相同的事件时钟，10–20 s 内只跳过无线 `correct`；无线位置重建/交接候选也看不到这些测量值。当前核按同一 call time 使用无额外时间补偿的观测，避免给任一递推核不同的时序处理。

## 运行与产物

在项目根 PowerShell 使用项目环境：

```powershell
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\native_schedule.py --initialization original_truth --names const1-trial4-tdoa2 --yaws 0 --output native_smoke
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\native_schedule.py --initialization sensor_only_original_prior --names const1-trial4-tdoa2 const2-trial4-tdoa3 const3-trial4-tdoa2 const4-trial4-tdoa3-traj1 --yaws 0 180 --output native_original_prior
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\native_policies.py --initial-prior original --output native_policies_original_prior
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\run.py --output holdout
& .\.venv\Scripts\python.exe .\imu-uwb-inekf\experiments\split_handoff\oursland\analyze.py
```

`native_original_prior/metrics.csv` 为主对照的原作者和直接新核逐录制连续结果；`native_policies_original_prior/metrics.csv` 为重新生成的交接政策；`analysis_original_prior/native_paired_cases.csv` 给出每行绝对误差和相对作者的差值（负值表示当前政策较低），另有逐情境汇总与图、每录制重跑命令 CSV。宽先验敏感性在 `native_paired` / `native_policies` / `analysis`，可用 `analyze.py --prior common` 重建图表。每个 `.npz` 可检查初值/交接状态和时间序列，`records.json`/`provenance.json` 保存调用数、失败与源哈希。

评分阶段在全部指定轨迹生成后开始。主网格为 t0+2.05/16.05/26.05 至 t0+44.95 的 20 Hz 零阶保持输出，真值位置线性插值、姿态 SLERP，无配准。whole 从 t0+.65 s 起；短轨迹必须同时观察失败/样本数，不能将存活前缀 RMS 当成完整成功。四录制的小样本不支持推断所有 UTIL 工况或论文全部初角分布。
