# task02 算法优化精简执行报告

日期：2026-07-23  
代码目录：`E:\0syncthing\Code\InEKF\imu-uwb-inekf`  
审计输入：`tasks/02InEKF_IMU-UWB_算法优化精简分析.md`

## 1. 结论

本轮没有机械执行第三方报告，而是逐项和原论文、公开代码及当前改进版核对。
最终在当前已修复的右不变协方差/门限版本上，落地了五项低风险优化：

1. 以精确终止的三次多项式替代每 IMU 步的通用 `15x15 expm`；
2. 利用过程噪声对角/零列结构，避免稠密零块构造和连续稠密乘法；
3. 将标量 TDoA Joseph 协方差更新改为等价 rank-one 更新；
4. 直接计算当前已验证观测路径的 `H_left Ad(X^-1)`，不再构造增广伴随；
5. 用共享三角函数的 `Gamma1/Gamma2` 闭式 IMU 积分改善快速旋转传播。

此外，取消了每步不必要的 SVD 旋转投影，并为高频结果记录增加了
`SO3_log` 解析快路径；近 pi 仍回退到 SciPy 稳健实现。

三组代表数据的最终端到端中位加速为 1.292x--1.313x。速度、姿态和 yaw
RMSE 全部不变或略有改善；const3 快速手动、180° 初差的位置 RMSE 增加
0.0014%，已如实保留。三组 UWB 拒绝数完全不变。

## 2. 第三方建议审计

| 建议 | 判断 | 本轮处理 |
|---|---|---|
| 偏置统一为 `[b_g,b_a]` | 正确，但已修复 | 不重复修改 |
| 修正事件时间对齐 | 正确，但已修复 | 不重复修改 |
| 修正小角度左 Jacobian | 正确，但已修复 | 保留现有连续级数 |
| 统一右不变 H/P/NIS | 正确，且已完成最终修复 | 保留 `P_r=Ad_X P_l Ad_X^T` 和转换后单次门限 |
| 合理初始 yaw 协方差 | 正确，且已修复 | 保留 2 rad 全圆未知先验 |
| `A^4=0`，解析 Phi | 对当前冻结矩阵成立 | 已实现并和 `expm` 做单元等价测试 |
| `3x3` 分块传播 P | 数学可行，但 NumPy 中未必更快 | 本轮只去除严格零块；未引入高风险 Python 切片循环 |
| 标量 rank-one 更新 | 正确 | 已实现；与 Joseph 逐元素回归 |
| 直接 UWB Jacobian | 原则正确，但报告给的是完整杆臂模型 | 只直接展开当前成熟路径，未启用会改变大 yaw 行为的研究版完整 H |
| 连续 PSD 的有限阶 Qd | 推导可行，但会改变当前“每样本方差”参数语义 | 未偷换模型；Van Loan 留在 `verified` 研究模式 |
| Gamma1/Gamma2 IMU 积分 | 正确 | 已实现，共享 Exp 的 sin/cos 和矩阵幂 |
| 分锚点对自适应/Huber | 方向合理，但缺少冻结标定与独立验证 | 未默认启用，避免在线自适应掩盖残余模型错误 |
| 协方差降频/15→9 维 | 属于有性能风险的近似 | 未采用 |
| 离线锚点、杆臂、时延标定 | 正确但超出本轮可证范围 | 未伪造标定结果 |

特别说明：报告关于 `A^4=0` 的结论一开始看似可能忽略状态相关偏置耦合。
实际按当前矩阵写成 `A=[[A0,B],[0,0]]` 后，`A0^3=0`，所以
`A^4=[[0,A0^3 B],[0,0]]=0`；该项经代码数值测试确认后才采纳。

## 3. 数学与实现

### 3.1 精确状态转移

原论文式 (40) 为：

```text
Phi = exp(A dt)
```

当前冻结传播矩阵满足 `A^4=0`，因此精确地：

```text
Phi = I + A dt + 1/2 (A dt)^2 + 1/6 (A dt)^3
```

这不是截断近似。测试同时检查 `A^4` 和 `structured_transition(A,dt)` 对
`scipy.linalg.expm(A*dt)`，误差约为机器精度。

### 3.2 稀疏过程噪声

令 `M = Phi Ad_X`、`d` 为 15 个对角方差乘 `dt^2`，则：

```text
Qd = (M * d[None, :]) @ M.T
```

位置噪声三列为零。该公式和旧的 `M @ diag(d) @ M.T` 数学等价，不改变参数。

### 3.3 标量 UWB 更新

```text
u = P H^T
S = H u + R
delta = u residual / S
P = P - u u^T / S
```

保留 `S>0` 检查和更新后协方差对称化。单元测试对照 Joseph 形式。

### 3.4 当前观测路径的直接展开

本轮没有把 `verified` 中的完整杆臂 Jacobian强行切到生产路径。对当前
`H_r=H_l Ad(X^-1)` 路径，只把等价矩阵乘法展开为：

```text
H_theta = -h_p R^T p^
H_pos   =  h_p R^T
```

它和旧增广伴随路径逐元素一致。

### 3.5 闭式 IMU 积分

在采样区间内常角速度、常体坐标比力假设下：

```text
R+ = R Exp(phi)
v+ = v + g dt + R Gamma1(phi) a dt
p+ = p + v dt + 0.5 g dt^2 + R Gamma2(phi) a dt^2
```

`Exp/Gamma1/Gamma2` 共享一次 `phi^`、一次平方和一组 `sin/cos`。
`Gamma2` 对 20001 点数值积分验证。

## 4. 自动测试

命令：

```powershell
D:\miniconda\Scripts\conda.exe run --no-capture-output `
  -p D:\conda-envs\imu-uwb-inekf python tests\test_inekf.py
```

结果：16/16 通过。新增覆盖：

- `A^4=0` 与解析 Phi/expm 等价；
- 稀疏/稠密过程噪声等价；
- Gamma2 数值积分；
- 直接/伴随 UWB Jacobian 等价；
- rank-one/Joseph 等价；
- 20000 次旋转乘法后正交误差约 `2e-12`，无需每步 SVD。

`compileall` 同时通过。

## 5. 数据集配对结果

方法：同一进程只加载一次数据，每组 `pre_optimization` 和 `proposed`
各运行四次，奇偶轮反转顺序；耗时取中位数。`pre_optimization` 是当前
坐标一致性修复后的 task01 路径，不是论文原始错误代码。

| 数据 | 前/后耗时 s | 加速 | 位置 RMSE 前/后 m | 速度 RMSE 前/后 m/s | 姿态 RMSE 前/后 rad | yaw RMSE 前/后 rad | 拒绝前/后 |
|---|---:|---:|---:|---:|---:|---:|---:|
| const1 自动 30 s，yaw 0° | 6.293 / 4.794 | 1.313x | 0.107017 / 0.107016 | 0.072909 / 0.072891 | 0.157868 / 0.157598 | 0.157761 / 0.157491 | 0 / 0 |
| const4 手动 60 s，yaw 90° | 12.215 / 9.325 | 1.310x | 0.355052 / 0.355050 | 0.186773 / 0.186721 | 0.197206 / 0.197167 | 0.191783 / 0.191743 | 151 / 151 |
| const3 快速 60 s，yaw 180° | 10.361 / 8.016 | 1.292x | 0.205951 / 0.205954 | 0.171065 / 0.171033 | 0.227189 / 0.227187 | 0.226291 / 0.226288 | 215 / 215 |

相对变化：

- const1：位置 -0.0017%，速度 -0.0250%，姿态 -0.1713%，yaw -0.1716%；
- const4：位置 -0.0007%，速度 -0.0278%，姿态 -0.0195%，yaw -0.0206%；
- const3：位置 +0.0014%，速度 -0.0184%，姿态 -0.0009%，yaw -0.0011%。

误差变化远小于数据场景间差异，没有通过放宽 NIS 门限获得速度或精度。

## 6. 图片检查

检查文件：

- `imu-uwb-inekf/results/task02_optimization/comparison.png`

目视结论：

- 三组前后轨迹和位置误差曲线几乎完全重合；
- 曲线连续，无 NaN、跳变或发散；
- const1 规则圆轨迹闭合合理；
- const4 手动轨迹虽局部偏离 Vicon，但前后没有新增峰值；
- const3 快速轨迹本身复杂，180° 初差仍稳定收敛，没有重现过去的门限闭锁。

## 7. 文件与复现

- 核心实现：`scripts/estimation/inekf.py`
- 入口变体：`scripts/estimation/pipeline.py`、`scripts/estimation/main.py`
- 单元测试：`tests/test_inekf.py`
- 配对脚本：`experiments/run_task02_optimization.py`
- 数值结果：`results/task02_optimization/metrics.json`、`metrics.csv`
- 图：`results/task02_optimization/comparison.png`
- 数学说明：`tex/task02_structured_optimization.tex/.pdf`

复现：

```powershell
D:\miniconda\Scripts\conda.exe run --no-capture-output `
  -p D:\conda-envs\imu-uwb-inekf python `
  experiments\run_task02_optimization.py --repeats 4
```

## 8. 限制

- 本结果是 Windows 桌面 Python 回放耗时，不是 Crazyflie 目标 MCU 的 WCET；
- 稠密 `15x15` 协方差传播仍在，嵌入式 C/C++ 可进一步做固定分块展开；
- 尚未完成采样率不变性、模拟真值 NEES、偏置真值 RMSE；
- 没有在缺少标定证据时启用自适应 UWB，也没有用降频/降维近似美化速度；
- const3 的 yaw RMSE 仍约 0.226 rad，结构化计算保持了现有效果，但没有解决
  单标签 TDoA 的根本航向可观测性限制。
