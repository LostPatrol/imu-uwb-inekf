# InEKF 论文与代码审阅：实时化和适用范围扩展

日期：2026-07-23  
审阅对象：

- 论文：Oursland and Mehrabian, *An Invariant Extended Kalman Filter for IMU-UWB Sensor Fusion*, IEEE RO-MAN 2024。
- 代码：`E:\0syncthing\Code\InEKF\imu-uwb-inekf` 当前工作树。
- 数据：`E:\datasets\dataset\flight-dataset`，重点复核论文明确删除的 constellation 3 四个手动飞行数据。

## 1. 结论摘要

### 1.1 如何大幅降低计算复杂度

目前的首要问题不是 InEKF 的 15 维状态本身过大，而是把一个具有固定稀疏结构的递推器写成了高频 Python 动态数组程序：

1. 每个约 1 kHz IMU 样本都动态构造 `np.block`、`block_diag`、Adjoint 和多个临时矩阵；
2. 每次传播调用通用 15×15 `expm`；
3. 每个约 228--476 Hz UWB 标量观测都用完整 Joseph 形式做多个 15×15 乘法；
4. 每个 IMU/UWB 事件都做 `SO(3)` 对数、SVD 正交投影，并保存完整 15×15 协方差历史；
5. 时间流仍通过 `unique + dict + Python for` 合并，不是固定内存的双指针流式调度。

30 s const1 的 cProfile 显示，`predict`、全量状态记录、`correct` 的累计耗时分别为 10.24、4.14、2.21 s（profile 总计 16.94 s，profile 本身有明显开销，只应用于热点排序）。其中反复执行的 `block_diag`、`np.block`、`SO3_log`、`expm` 和 SVD 累计耗时分别为 3.60、3.22、3.87、0.88、0.73 s。通用矩阵指数是热点，但动态构造和诊断记录合计比它更严重。

推荐的机载实现应采用：

- 1 kHz 名义状态传播，但把协方差传播与 UWB 更新限制在 100--250 Hz，使用 4--10 个 IMU 样本的流形预积分；
- 固定大小、预分配的 C/C++ 数组，直接填充稀疏 \(A\)、\(GQG^\mathsf T\)，不再调用 `np.block`/`block_diag`；
- 用二阶结构化离散
  \[
  \Phi \simeq I+A\Delta t+\tfrac12A^2\Delta t^2,
  \quad
  Q_d\simeq Q_c\Delta t+
  \tfrac12(AQ_c+Q_cA^\mathsf T)\Delta t^2
  \]
  取代每步通用 `expm`，但必须以 NEES/NIS 和全数据集误差验证截断阶数；
- 标量 UWB 更新使用秩一协方差更新
  \[
  c=Ph^\mathsf T,\quad S=hc+R,\quad
  K=c/S,\quad P^+=P-cc^\mathsf T/S,
  \]
  并以平方根/UD 形式或周期性 Joseph 校正保证数值稳定；
- 不在实时核中计算旋转向量、SVD、全量历史和绘图，只保留定长环形诊断缓冲；
- 旋转正交性用群指数本身维持，仅在误差阈值超限时做低频极分解；
- 用双指针消费按时间排序的 IMU/UWB 队列，避免 `np.unique`、浮点字典和全量事件表。

这样可把稳态传播从“通用稠密 \(O(n^3)\) + 高频分配”降为固定稀疏 \(O(n^2)\)，标量更新也降为 \(O(n^2)\)，内存从 \(O(Nn^2)\) 历史降为 \(O(n^2)\) 固定工作区。最终加速倍数必须在目标机载 CPU/MCU 上测量，不能用桌面 Python 比例冒充。

### 1.2 如何让原本失败的数据集通过

论文第 V 节明确说明 constellation 3 的四个手动飞行数据因快速机动而被删除，甚至在真值初始状态下两个滤波器都没有可接受结果。当前 `proposed` 在真值姿态/位置初值下已能稳定跑完四组：

| 数据集 | 位置 RMSE / m | 速度 RMSE / (m/s) | 姿态 RMSE / rad | yaw RMSE / rad | UWB 拒绝率 | 回放/数据时长 |
|---|---:|---:|---:|---:|---:|---:|
| const3 manual1 TDoA2 | 0.267 | 0.236 | 0.054 | 0.052 | 1.06% | 28.7/67.9 s |
| const3 manual2 TDoA2 | 0.331 | 0.268 | 0.137 | 0.136 | 1.24% | 53.8/124.8 s |
| const3 manual3 TDoA3 | 0.242 | 0.197 | 0.132 | 0.131 | 1.82% | 45.4/118.0 s |
| const3 manual4 TDoA3 | 0.258 | 0.212 | 0.097 | 0.095 | 1.76% | 53.3/165.5 s |

这些结果说明当前偏置顺序修正、事件驱动传播、初始静止陀螺标定和稳健 Lie 运算已经使“真值附近跟踪”明显优于论文公开实现。但是它还没有解决机载冷启动和全局收敛：

- `_initial_filter` 在 `pipeline.py:112-123` 直接使用 Vicon 真值姿态和位置；
- manual3 前 60 s 在 90° yaw 初差下位置 RMSE 为 0.211 m，但 yaw RMSE 仍为 0.473 rad，5° 内未收敛；
- 同一数据在 180° yaw 初差下位置 RMSE 达 571.8 m，83.3% UWB 被拒绝，轨迹发散到公里量级。

180° 失败形成明确闭环：

\[
\text{大初值误差}
\rightarrow \text{局部线性化残差大}
\rightarrow \chi^2\text{ 硬门限拒绝校正}
\rightarrow \text{纯 IMU 漂移}
\rightarrow \text{更大的残差与更多拒绝}.
\]

因此，扩大适用范围的优先级应是：

1. **无真值的全局初始化/重定位**：静止 IMU 给 roll/pitch 和陀螺偏置，鲁棒 TDoA 多边定位给位置；yaw 使用 8--16 个短时并行假设或 1--3 s 鲁棒滑窗批优化，只在启动/失锁时运行。
2. **物理预门控**：TDoA 必须满足
   \[
   |z_{BA}|\le \|a_B-a_A\|+\epsilon.
   \]
   manual3/manual4 分别有 193/221 条观测违反该不等式，最大绝对测量达 334.6/176.9 m，可在进入滤波前直接剔除。
3. **从硬拒绝改为重尾鲁棒更新**：对有限但可疑的残差使用 Huber/Student-t 权重，而不是全收或全拒；连续高拒绝率触发协方差膨胀或重定位，避免锁死。
4. **按锚点对建模**：const3 manual2 不同锚点对的稳健噪声尺度约 0.086--0.365 m，且偏置方向不同；当前只按 `"const4"` 与否选固定 \(R\)（`pipeline.py:176`）过于粗糙。应在线估计有界的 pair-specific bias 和 \(R_{ij}\)。
5. **批量处理相关 TDoA**：共享锚点和相近时间戳的 TDoA 并不独立。应按无线包/短时间窗组成小批量更新，显式给出相关噪声，或降低有效更新频率，防止过度自信。
6. **时延和外参**：快速机动时，IMU--UWB 时间偏移会产生
   \[
   \frac{\partial h}{\partial\delta t}
   \simeq \nabla_p h^\mathsf T v_{\rm tag}.
   \]
   先离线标定固定时延，再在可观测时以有界一维状态慢速估计；同时统一右不变传播、测量和注入坐标后再启用完整杠杆臂 Jacobian。
7. **大残差迭代更新**：对通过物理门限但线性化误差大的观测组，最多执行 2--3 次 iterated InEKF/Gauss--Newton 更新，避免单次 EKF 线性化无法跨越大 yaw 初差。

## 2. 论文与当前代码的对应关系

论文式 (40) 使用

\[
\Phi=e^{A\Delta t},\qquad
\hat P_{k+1}=\Phi \check P_k\Phi^\mathsf T+
LQ_kL^\mathsf T\Delta t.
\]

代码 `inekf.py:421-450` 每个 IMU 区间重新构造 \(A\)、Adjoint 和样本噪声，并在默认路径调用通用 `scipy.linalg.expm`。代码把噪声解释为“每样本方差”并使用 \(\Delta t^2\)，而论文写成连续噪声密度乘 \(\Delta t\)。两者单位不同，不能直接互换；移植前应通过静止数据 Allan 方差和采样率明确 \(\sigma_g,\sigma_a,\sigma_{bg},\sigma_{ba}\) 的单位。

论文式 (35)--(38) 先写左不变 UWB 观测，再用 Adjoint 变换到右不变坐标。当前默认 `correct`（`inekf.py:500-520`）沿用这一做法，但完整杠杆臂直接 Jacobian 只在 `verified` 中启用。该直接 Jacobian 已通过中心差分，却在大 yaw 实验中可能降低收敛，说明“有限差分局部正确”仍不足以证明整套误差定义、残差符号、状态注入和 reset 坐标全局一致。应先做统一推导和大误差仿真，再进入生产路径。

论文只用固定卡方门限拒绝异常值。数据复核表明：

- const1 的真值 TDoA 残差稳健尺度约 0.087 m，\(|r|>0.5\) m 比例为 0；
- const3 manual1--4 的稳健尺度约 0.148、0.176、0.233、0.212 m；
- \(|r|>0.5\) m 比例约 6.9%、10.8%、10.8%、10.2%；
- TDoA3 还有少量百米级物理不可能值。

固定高斯噪声和单一 5σ 硬门限不能同时处理“普通 LOS、pair bias、NLOS 重尾、报文毛刺和大初值误差”五种情形。

## 3. 实时化的建议实施顺序

### P0：先分离实时核与研究记录

- 新增 `embedded`/`realtime` 输出策略，只存当前 \(X,P,b\)、计数器和最近定长 NIS 环形缓冲。
- `_append_state` 不再在每个 UWB 事件计算 `SE3_2_vec`；评估回放才保存抽稀轨迹。
- 去除滤波内重复 `nis_history` 与外部 `output["nis"]` 双份记录。
- 用两个递增索引流式合并 IMU/UWB，不构造 `events`、字典和浮点键。

这一阶段不改变滤波数学，风险最低；profile 中全量状态记录占 24% 左右累计时间，并造成约 9.8 万至 20.6 万份状态/协方差历史。

### P1：固定结构的矩阵核

- 初始化时缓存 \(Q\) 的对角项、重力叉乘矩阵、单位矩阵和所有工作区。
- `_error_dynamics` 直接向预分配 15×15 数组的非零块写值。
- \(GQG^\mathsf T\) 利用对角 \(Q\) 按列缩放，避免两个稠密乘法。
- 标量 UWB 用秩一更新；三维零角速率更新使用 Cholesky/UD。
- 用 `float32` 和 `float64` 两套回放检查误差、PSD 与 NEES，再决定目标机精度。

### P2：多速率预积分和结构化离散

- 姿态/速度/位置仍以 IMU 原速率传播；
- 协方差和偏置 Jacobian 每 \(M=4,5,10\) 个样本或在 UWB 到达时传播一次；
- 累积旋转、速度和位置增量，并包含 coning/sculling 补偿；
- 比较一阶、二阶、闭式块转移和现有 `expm`，选择满足一致性要求的最低阶方案。

不建议只把 IMU 粗暴降采样，因为快速机动数据正是需要扩展的范围。

### P3：目标硬件实现

- C/C++ 固定大小矩阵，禁止运行期堆分配；
- 给出每类事件的 WCET、p99 延迟、栈/堆峰值和 CPU 占用；
- 最坏 1.1 kHz IMU + 500 Hz UWB 下，建议验收目标为平均 CPU 占用不超过 30%，单次 IMU deadline 不超过 1 ms，且连续 10 分钟无 deadline miss。

## 4. 扩展适用范围的建议实施顺序

### R0：数据完整性层

- finite、ID、时间单调性、重复包检查；
- TDoA 基线物理界；
- 每锚点对的短窗中位数/MAD；
- 锚点健康度和连续异常计数。

### R1：鲁棒自适应观测层

Student-t 权重可写为

\[
w(\nu)=\frac{\nu_0+1}{\nu_0+\nu^2/S},\qquad
R_{\rm eff}=R/w,
\]

并给 \(R_{\rm eff}\) 设置上下界。极端物理错误直接拒绝，中等 NLOS 残差降权，正常 LOS 保持高信息量。若最近窗口拒绝/低权重比例持续过高，不应继续静默纯惯导，而应进入 recovery。

### R2：全局启动与 recovery

启动目标函数建议为

\[
\min_{p_0,\psi,b_g,b_a,\delta t}
\sum_j \rho\!\left(
\frac{z_j-h_j(X(t_j+\delta t))-\beta_{i_jk_j}}
{\sigma_{i_jk_j}}
\right)
+\|\omega_m-b_g\|^2_{W_g}
+\|a_m-R^\mathsf Tg-b_a\|^2_{W_a}.
\]

多 yaw 假设只在前 1--3 s 或失锁时运行，选出明显占优的似然后回到单滤波器，因此不会长期乘上 8--16 倍计算量。若各假设不可区分，保持 yaw 大协方差并限制控制器动作，不伪造已经收敛。

### R3：时空标定与完整不变 Jacobian

- 先在数据集上离线估计每段固定 \(\delta t\) 和 UWB 杠杆臂；
- 只有 Fisher 信息/可观测性足够时才在线更新；
- 对传播、观测、注入、reset 写同一右不变误差定义的解析推导；
- 保留中心差分单测，并新增 90°/180° 大误差 Monte Carlo 与 NEES 测试。

## 5. 必须改进的评价方法

`pipeline.py:309-314` 的“持续收敛时间”只检查从某点开始的未来 2 s，因此 manual2 明明随后长期保持 5--10° yaw 误差，指标仍报告约 0 s 收敛。建议同时报告：

- first-entry：首次进入阈值；
- sustained-entry：从该点直到数据末尾至少 95% 样本在阈值内；
- last-exit：最后一次离开阈值的时间；
- 末段 20%/50% RMSE；
- 发散率、最大连续拒绝时间、recovery 次数；
- NIS 覆盖率和有仿真真值时的 NEES 覆盖率。

可部署验收不能使用 Vicon 初始化。建议对全部 79 个 CSV 执行 yaw 初差
\(\{0^\circ,45^\circ,90^\circ,135^\circ,180^\circ\}\) 和多个位置初差/时延/NLOS 注入，分别统计 TDoA2、TDoA3、自动、手动和 constellation。

## 6. 本轮验证与图像复核

- `tests/test_inekf.py`：9/9 通过。
- 完整运行 const3 四个手动数据，结果见 `imu-uwb-inekf/results/review_const3_manual1` 至 `review_const3_manual4`。
- 运行 manual3 前 60 s 的 90°/180° yaw 初差；180° 图见 `results/review_const3_manual3_yaw180_60s`。
- 目视检查：
  - yaw 0° 四组轨迹连续、无数值跳变或发散；
  - manual2 位置误差存在约 1.1 m 峰值，yaw 长期偏 5--10°；
  - manual3 位置误差峰值约 0.7 m，yaw 在约 15--95 s 多数时间超过 5°；
  - manual3 yaw 180° 时轨迹快速漂移到公里量级，与 83.3% UWB 拒绝率一致，不是绘图尺度或指标假象。

## 7. 推荐的最小下一阶段

下一阶段不应同时重写全部模块。建议先完成一个可证伪的 P0+P1+R0+R1 原型：

1. 实时核关闭全量历史，预分配 \(A,Q,\Phi,P\) 工作区；
2. 物理 TDoA 门控 + pair-specific MAD/Student-t 更新；
3. 连续高拒绝率 recovery 状态机，但暂不在线标定时延；
4. 对 manual3 yaw 180° 使用启动期 16-yaw 假设选择；
5. 在全部 79 个数据、5 个 yaw 初差和目标硬件上验收。

只有该阶段证明 manual3 yaw 180° 不再锁死、四个手动集误差不退化且目标硬件满足 deadline，才继续 P2 多速率预积分和 R3 在线时空标定。

