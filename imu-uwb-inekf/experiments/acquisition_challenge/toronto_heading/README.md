<!-- Independent TDoA heading experiment, kept separate from common TWR rankings. -->
# Toronto UTIL 单标签 TDoA 真实航向干预

该支线用于补充只有位置真值的数据：用 Vicon 四元数直接评机体真实航向。**它不是共同 TWR 排名，不能与 Delft/ISAS 的 TWR 分数混排。** 未纳入 Jiang/ObjectTrack 或 SFUISE；结果仅对应未修改的 Oursland 作者 core 与本目录明确说明的外接初始化/回放器。

## 数据与来源

[UTIL 官方项目页](https://utiasdsl.github.io/util-uwb-dataset/)、[官方代码及数据发布](https://github.com/learnsyslab/util-uwb-dataset)、[IJRR 数据论文](https://doi.org/10.1177/02783649241230640)。官方说明 Vicon pose 为机体真值、锚点位置由 Leica 全站仪测量，IMU 与机体原点对齐；机体至 UWB 杆臂为 `[-0.01245, 0.00127, 0.0908] m`，与作者滤波器相同。

固定四条录制为 `const1/2/3-trial4-tdoa2` 和 `const4-trial4-tdoa2-traj1`，不按结果筛选。原始文件位于 `E:/datasets/dataset/flight-dataset/`。`data/<recording>/` 包含：

- `truth.csv`：原始 `t_pose,pose_xyz,pose_qxyzw`，无轨迹对齐、无坐标拟合。
- `imu.csv`：原始加速度采样时间，单位 g；陀螺按作者做法插值至加速度时间，单位 deg/s。
- `radio.csv`：TDoA2 原始有向环观测，符号为 `rhoB-rhoA`，单位 m。
- `anchors.csv`、`survey.txt`：公开 surveyed anchors 与原 survey 文件。
- `sensor_base.npz`：首段1.5秒传感器初始化及其连续运行至15秒的机体状态；不含真值构造的状态。
- `meta.json`：原始输入与survey的 SHA-256、杆臂、首段静止检测和传感器拟合结果。

## 协议与真值隔离

共同首段为1.5秒。外接初始化用 TDoA 做天线位置/速度 CV 拟合：使用录制开始至 `t_acc[0]+1.5s` 的全部有向环无线包（只取时间戳不超过前缀结束的包），并用加速度重力方向确定倾角；航向没有外部先验，以重力对齐的任意零航向开始。仅检测到静止时使用首段陀螺均值。**这里使用 IMU+TDoA 前缀，与共同 TWR 组仅用 IMU 的前缀不同，不能跨组排位。** 没有使用 ToF、高度、光流、磁力计、真值 Z/倾角/速度。该接口并非作者自带的无先验初始化，应称“外接 sensor-only bootstrap + unchanged author core”。其宽协方差是必要的滤波内部初始化，不是额外提供的正确位置/航向先验。

主干恢复固定在录制绝对时间15秒，释放后运行15秒。恢复基础状态为上述 sensor-only 轨迹的状态；只把 XY 和世界 z 轴航向中心设置为真值 XY/yaw 加人为误差。Z、倾角、速度、偏置保持传感器估计，不从真值补齐，不把错误施加到 IMU 或无线观测。世界 z 轴左乘旋转保留倾角。真值仅在构造受控错误先验中心和最终评分时读取。

每条96例：4个 XY 幅度 `{1,5,10,20} m` × 4方向 `{+x,-x,+y,-y}` × 航向误差 `{0,90,180}°` × 2种物理协方差。位置/航向标准差分别为 wide `(20m,120°)`、tight `(0.10m,5°)`；Z标准差0.5m、倾角5°、速度0.5m/s在两组固定。`P=M C Mᵀ` 映射物理世界左姿态/加性速度/加性位置误差至作者不变误差坐标，其中 `M[v,theta]=[v]×`、`M[p,theta]=[p]×`。这不是只改位置块的伪物理先验。

## 作者 core 与时间含义

运行时直接只读导入 `../imu-uwb-inekf - backup/scripts/estimation/inekf.py`。其 SHA-256、输入数据指纹在 `protocol.json` / `meta.json` 留档；不复制或修改作者公式。保留作者 `.13 m²` 测距方差、偏置约定、门限、地面 clamp。作者 main 的事件 dt 规则保持：用前一事件的 IMU/首个同时间戳 radio，状态标记下一事件时间。TDoA2 使用公开8锚点有向环。

回放器约50Hz记录估计器输出；记录实际状态时间及单核实际计算服务队列完成时间 `available=max(previous_available,event_time)+event_compute_time`。此外，作者陀螺线性插值可能引用后一原始 gyro 包：`finalize_clock.py` 根据每条录制前30秒所有加速度样本的插值上界，给整条发布队列统一加上该录制最大的前视等待。这是保守缓冲，保证发布不早于任何已引用后续gyro包的到达；状态和时间戳不改，不回填输出。`causal_clock.json` 同时报告最大/中位原始插值等待及实际加到每个输出的最大/中位缓冲。

计算服务时间为此次本机测得，包含操作系统/并行争用导致的波动；不是跨机器固有延迟。数组 `outputs` 前两列分别是相对释放的 `state_s,available_s`，其中 available 已含插值缓冲；`outputs_unbuffered` 仅保留缓冲前队列供审计，最终指标均使用 `outputs`。没有给 t=0 伪造输出。

## 评分与结果文件

释放后 `{0,.5,...,15}s`，取已经完成发布的最新输出并保持，与**当前时刻**插值 Vicon 的 XY 和真实机体 yaw 比较；没有把估计对齐到较早的输出真值。四元数使用 SLERP、位置线性插值、yaw为机体 x 轴在 surveyed 世界XY平面的 atan2，误差wrap到180°内。主指标是 XY≤0.5m 连续1秒（三格），同时保留进入时间、确认时间；0.25m/1m作敏感性分析。t=0无输出记 unavailable，不计成功。2/10/15秒端点同时报告 XY 与真实 yaw，XY capture 并不代表航向正确。

`metrics.csv` 保留每例失败、native首次输出/可用时间、runtime、三种捕获门限和端点；`summary.csv` 分协方差/注入航向汇总；`results/*.npz` 含输出、评分格点、受扰初态及两种坐标下协方差。`overview.png` 展示主捕获率及10秒XY/yaw。`execution.json` 保存四worker性能核选择、总墙时、完成数及作者文件结束指纹。

```powershell
& 'E:/0syncthing/Code/InEKF/.venv/Scripts/python.exe' 'E:/0syncthing/Code/InEKF/imu-uwb-inekf/experiments/acquisition_challenge/toronto_heading/run.py' --export --run
& 'E:/0syncthing/Code/InEKF/.venv/Scripts/python.exe' 'E:/0syncthing/Code/InEKF/imu-uwb-inekf/experiments/acquisition_challenge/toronto_heading/finalize_clock.py'
& 'E:/0syncthing/Code/InEKF/.venv/Scripts/python.exe' 'E:/0syncthing/Code/InEKF/imu-uwb-inekf/experiments/acquisition_challenge/toronto_heading/audit.py'
```

仅重评分可使用 `--summary`；已完成case按独立JSON恢复，不覆盖旧结果。重新改变协议或源码后应使用新的结果目录，不能把旧checkpoint当新实验。所有任务仅输出本目录，不改主评分器、备份作者源码或系统设置。

## 本次结果

2026-11-13 完成384恢复+4首段。四worker绑定逻辑核0/2/4/6，core回放墙时1042.83秒；全部388例经过统一因果gyro缓冲重评分，`audit.json` 为 `passed:true`。数值失败0，但很多轨迹没有恢复到目标误差；未按结果删除轨迹。作者core文件结束指纹与开始一致。图已人工查看：`overview.png`。

| 物理先验 | 恢复数 | 15s内≤0.25m持续1s | ≤0.5m持续1s | ≤1m持续1s | 10s XY中位(m) | 10s真实yaw中位(°) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tight | 192 | 53 | 58 | 60 | 16.39 | 38.25 |
| wide | 192 | 58 | 59 | 60 | 37.81 | 36.15 |

180°注入子集中，tight为0/64捕获、wide为15/64捕获；对应10秒真实yaw中位分别164.92°与99.04°。宽协方差并没有普遍解决位置/航向大误差恢复，XY短暂达标也不能当作航向已恢复。

| 外接sensor-only首段 | ≤0.5m持续1s确认(s) | 10s XY(m) | 10s真实yaw(°) |
| --- | ---: | ---: | ---: |
| const1 | 1.5 | 0.119 | 6.19 |
| const2 | 1.5 | 0.032 | 6.55 |
| const3 | 1.5 | 0.163 | 14.32 |
| const4-traj1 | 2.5 | 0.031 | 2.12 |

以上为固定四条录制的受控先验实验，不宣称所有新场景中的普遍成功率。外接无先验首段与恢复矩阵采用不同初始化任务，分别呈现。
