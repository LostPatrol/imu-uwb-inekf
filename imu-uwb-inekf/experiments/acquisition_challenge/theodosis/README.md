# Theodosis 二次输出观测器：适用条件核查

对应原文为 Dionysios Theodosis、Soulaimane Berkane、Dimos V. Dimarogonas，*State Estimation for a Class of Linear Systems with Quadratic Output*，并非 Pfeiffer。已发表于 **IFAC-PapersOnLine 54(9), 261–266, 2021**（MTNS 2020 专辑），[正式 DOI](https://doi.org/10.1016/j.ifacol.2021.06.149)；本地 `2006.02688v2.pdf` 与 [arXiv v2](https://arxiv.org/abs/2006.02688v2) 文件 SHA256 相同。

原文 §6 式(43)–(45)假定**惯性系加速度 (u) 已知**，使用单个原点锚的半平方距离；Theorem 3 的全局收敛还需要扩展系统一致可观，§6 给出加速度/jerk 的持续激励充分条件。其状态没有姿态或 IMU 偏置：

\[
\dot p=v,\quad\dot v=u,\quad y=\tfrac12\|p\|^2,
\qquad z=(q_0,q_1,q_2,p,v),
\]
\[
q_0=\tfrac12p^Tp,\ q_1=p^Tv,\ q_2=v^Tv;
\quad\dot q_0=q_1,\quad\dot q_1=q_2+p^Tu,\quad\dot q_2=2v^Tu.
\]

**不能把本基准未知初始航向的 body IMU 直接代入，作为原法的完整冷启动行。** 实际需要 (u=R(a_m-b_a)+g)，陀螺只能传播相对姿态，静止重力前缀不提供初始 yaw。若只有一个原点锚，则绕重力轴同时旋转 (p,v,R) 不改变 body IMU 或距离，绝对 yaw 本身不可辨；多锚已知几何可破除此对称，但本文未估计该未知姿态，异步多锚也是扩展。ISAS 的非零杆臂量测 (r_j=\|p+R\ell-a_j\|) 还需要姿态相关处理。

项目现有 [`observer.py`](../../split_handoff/quadratic_observer/observer.py) 的 `frozen_step` 正确接收**世界系**加速度，`quadratic_update` 是异步多锚扩展。[`run.py`](../../split_handoff/quadratic_observer/run.py) 的实测重放先由 Profile-1 估计启动 yaw，再传播 world input；[`translation_bridge.py`](../../split_handoff/translation_bridge.py) 使用已运行 InEKF 的速度增量。它们可作注明上游姿态条件的 p/v 模块对照，不能标成独立作者冷启动，也不能用 yaw 输入扰动冒充 yaw 估计结果。本次不新增不等信息的排行榜成绩行。

本次重跑原方程 0–20 s 仿真：held-input 一步误差 (1.14\times10^{-13})，积分成功；20 s 末位置/速度误差 0.0957 m / 0.0118 m/s，未达到旧脚本自定 (10^{-3}) 门槛。原文未给初始估计，脚本采用全零；原文 §6 的 (I_6) 参数与实际九维 lift/标量输出不合，脚本改用 (I_9)/标量1。故仅确认方程实现，不宣称精确复现原图。新检查输出：`agent/codex/acquisition_challenge_20261113/theodosis_simulation_check/`；公共数据未改。
