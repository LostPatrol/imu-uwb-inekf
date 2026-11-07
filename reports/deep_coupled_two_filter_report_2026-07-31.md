# 增广捕获/重置 + 原始 TWR 联合 InEKF 自适应跟踪报告

日期：2026-07-31  
实现目录：`imu-uwb-state-augment-deep-coupled-two-filters`

> 时间轴说明：本报告记录的是共同传感器时延修正前的阶段性结果。
> 当前正式实现、结果目录与数学说明已改用共同 IMU+TWR 延时
> \(\tau_s=0.20\,\mathrm{s}\)；应以
> `reports/common_sensor_delay_correction_report_2026-07-31.md` 和
> `results/deep_coupled_delay_full/` 为准。下文原 `deep_coupled_full/`
> 数值仅用于追溯，不能视为当前正式结果。
> 此外，下文 20 s 合成 MC 的 deep/capture 相同后来确认是状态机从未
> 获准切换，不是有效的自适应验证；修复见
> `reports/synthetic_monte_carlo_activation_fix_2026-07-31.md`。

## 1. 本轮目标与结论

本轮按要求只完成：

1. 增广 `LiftedTWRInEKF` 支路只负责初始化、大初差/失锁捕获、健康监视
   和 tracker 离散重置；
2. tracker 刚健康时以 0.3 权重接管，健康证据持续成立后平滑提高到 1.0；
3. 暂不强制投影增广辅助状态 \(q,s,h\)，原 capture 算法和参数保持不变。

最终实现通过正式验收：

- TU Delft 全部 12 条真实 TWR 日志；
- nominal `[0,0,0] m` 与 large `[6,-6,3] m` 两种初值，共 24 次配对；
- 相对同一估计器内部、从未被 tracker 改写的 capture 支路，全程位置、
  末 5 s 位置、全程速度、末 5 s 速度均为 **24/24 改善，0 次退化**；
- 20 次单锚点极大初值合成 Monte Carlo 仍为 **20/20 成功**；
- 普通 EKF、普通 InEKF、松耦合增广、联合 tracker、自适应输出五方法
  均完整保留；
- `compileall` 和 20/20 单元/回归测试通过。

自适应方案并未使本方法在所有指标上超过普通 EKF/InEKF。它解决的是：
保留增广法大初差捕获能力，同时在有持续证据时让联合 InEKF 真正接管稳态
位置/速度输出。普通 EKF 的末段位置中位数仍更低，这是结果边界，不作掩盖。

## 2. 最终结构

```text
                     +-> LiftedTWRInEKF capture ------------------+
同一 IMU + 原始 TWR -|   初始化 / 捕获 / 监视 / 离散重置         |-> 因果监督器
                     +-> 15维 raw-TWR joint InEKF tracker --------+      |
                                                                      v
                                                    自适应权重 0 -> 0.3 -> 1
```

### 2.1 capture 支路

capture 是未改动的 Task06 `LiftedTWRInEKF`：

- \(\chi=[q,s,h,p^\top,v^\top,1]^\top\)；
- 有效加速度过程标准差 `10 m/s²`；
- TWR 标准差 `0.20 m`，固定距离偏置 `-0.18 m`；
- NIS 门限 25，前 64 次更新预热；
- 本轮没有开启 \(q,s,h\) 物理流形投影。

它始终在后台独立传播和更新，不接受 tracker 的反馈。tracker 权重达到 1
以后，capture 对输出位置/速度的贡献为零，但继续作为健康参照和失锁恢复源。
这样，“只负责初始化、捕获和重置”指的是它的稳态输出角色退出，并非停止计算。

### 2.2 tracker 支路

tracker 是 15 维原始距离 `TWRInEKF`，误差状态为

\[
\delta x=[\delta\theta^\top,\delta v^\top,\delta p^\top,
\delta b_g^\top,\delta b_a^\top]^\top .
\]

对锚点 \(a_i\)，距离创新和更新为

\[
\nu_i=(\rho_i^m-b_\rho)-\|a_i-\hat p^-\|,\qquad
S_i=H_iP^-H_i^\top+\sigma_\rho^2,
\]
\[
K_i=P^-H_i^\top S_i^{-1},\qquad
\delta x_i=K_i\nu_i .
\]

量测雅可比的直接非零块是位置，但传播形成的姿态—速度—位置—偏置交叉
协方差使 \(K_i\) 的其他行也可非零，因此原始 TWR 可在同一次联合 Kalman
更新中修正完整 15 维误差状态。tracker 使用 128 次 gate 预热，并可被
健康的 capture 离散重置。

## 3. 状态机、健康度与自适应权重

### 3.1 捕获、切换与重置

监督器使用最近 96 次 TWR 更新。进入 track 模式仍需：

- 完整经历 20 s 捕获/交叉协方差成熟期；
- tracker 自上次重置后至少 64 次更新；
- 拒绝率不超过 0.20；
- 两支路位置分歧不超过 0.80 m；
- tracker 预拟合距离 RMSE 不超过 0.80 m，且不比 capture 差 0.08 m；
- 上述条件连续满足 12 次更新。

确认失锁条件为：非有限 tracker 状态、分歧超过 2.5 m、拒绝率超过 0.45，
或 tracker 距离 RMSE 比 capture 差 0.24 m 以上；连续 12 次后回到 capture。
如果 capture 已可信，则立即以 capture 状态和膨胀后的分块协方差重置 tracker。

### 3.2 原始健康度

定义 capture/tracker 预拟合距离 RMSE 为 \(J_C,J_T\)，位置分歧为
\(d_{CT}=\|\hat p_T-\hat p_C\|\)。为避免“两个支路位置几乎相同”时把微小
残差差异误认为强证据，定义单位分歧的距离收益

\[
\eta_k=
\frac{\max(J_C-J_T,0)}
{\max(d_{CT},0.10\ {\rm m})}.
\]

当 \(d_{CT}\le0.10\) m 时，距离证据分数直接为 1；否则将
\(\eta_k\) 从 0.03 到 0.12 线性映射到 \([0,1]\)。另分别构造：

- tracker 拒绝率分数；
- 两支路一致性分数；
- tracker 绝对距离 RMSE 分数；
- tracker 相对 capture 距离 RMSE 分数。

原始健康度取这些分数的最小值：

\[
\gamma_k=\min(
\gamma_{\rm reject},
\gamma_{\rm disagree},
\gamma_{\rm abs-range},
\gamma_{\rm rel-range},
\gamma_{\rm evidence})\in[0,1].
\]

使用 10 s 时间常数的一阶因果低通：

\[
\bar\gamma_{k+1}
=\bar\gamma_k+
\left(1-e^{-\Delta t/10{\rm s}}\right)
(\gamma_k-\bar\gamma_k).
\]

### 3.3 滞回授权与平滑交接

进入 track 模式时权重直接设为 \(w=0.3\)，以隔离 tracker 刚接管时的
局部线性化和重置瞬态。只有

\[
\bar\gamma\ge0.38
\]

连续保持 1 s，才授权全 tracker 输出；授权后若
\(\bar\gamma<0.25\) 则撤销。两个阈值形成滞回，避免健康度在边界抖动。

目标权重为

\[
w^\star=
\begin{cases}
0, & \text{capture 模式},\\
0.3, & \text{track 模式但未获全权授权},\\
1, & \text{track 模式且获全权授权}.
\end{cases}
\]

位置和速度输出为

\[
\hat p_o=(1-w)\hat p_C+w\hat p_T,\qquad
\hat v_o=(1-w)\hat v_C+w\hat v_T.
\]

从 0.3 到 1.0 的上升斜率为 \(0.7/5=0.14~{\rm s}^{-1}\)，因此完整交接
持续 5 s。撤销授权时以最大 \(1~{\rm s}^{-1}\) 向 0.3 回落；确认失锁则
绕过低通延迟，直接进入 capture 模式并令 \(w=0\)。

姿态和偏置在 track 模式取 tracker；本轮没有构造两支路融合协方差，
也不把加权输出反向注入任一滤波器，避免把共享 IMU/TWR 重复计数。

## 4. 实现过程中的失败版本

本轮没有只保留最好结果，以下中间版本均如实记录：

1. **健康布尔量一通过就固定爬升到 1。**  
   尾段总体明显改善，但 24 次真实回放中全程位置有 6 次、末段位置有
   2 次相对 loose 退化，最坏分别约 `+0.021 m`、`+0.158 m`，未通过验收。

2. **将瞬时距离证据连续映射为权重。**  
   恢复 24/24 不退化，但没有任何运行达到权重 1，过于保守，不满足
   “健康后由 0.3 平滑提高到 1.0”的目标。

3. **最终方案：低通 + 滞回 + 保持时间 + 斜率限制。**  
   24/24 四指标改善；12/24 运行实际到达权重 1，10/24 在结束时仍为 1。

## 5. 正式真实数据结果

数据集：

```text
E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset
```

### 5.1 相对原松耦合 capture 的严格配对

| 指标 | loose 中位 | 自适应中位 | 配对差中位 | 改善/相同/变差 | 最坏差值 |
|---|---:|---:|---:|---:|---:|
| 全程位置 RMSE (m) | 0.404446 | 0.362374 | -0.020259 | 24/0/0 | -0.003456 |
| 末 5 s 位置 RMSE (m) | 0.334650 | 0.232977 | -0.112113 | 24/0/0 | -0.008297 |
| 全程速度 RMSE (m/s) | 0.544147 | 0.494602 | -0.048951 | 24/0/0 | -0.039230 |
| 末 5 s 速度 RMSE (m/s) | 0.564690 | 0.480730 | -0.092421 | 24/0/0 | -0.022259 |

负差值表示改进。四项最坏差值仍小于零，不是仅靠中位数掩盖反例。

相对上一版固定 `w=0.3`，最终自适应方案还取得：

- nominal 全程位置 `0.375116 -> 0.357865 m`；
- nominal 末 5 s 位置 `0.273389 -> 0.232977 m`；
- large 全程位置 `0.379509 -> 0.366876 m`；
- large 末 5 s 位置 `0.273385 -> 0.232990 m`；
- 两种初值的末 5 s 速度均约从 `0.513` 降至 `0.480–0.481 m/s`。

### 5.2 五方法完整对比

| 初值 | 方法 | 全程位置中位 | 全程位置均值 | 末 5 s 位置中位 | 全程速度中位 | 末 5 s 速度中位 | 收敛 |
|---|---|---:|---:|---:|---:|---:|---:|
| nominal | 普通 EKF | 0.387111 | 0.396931 | **0.192065** | 0.518820 | **0.465887** | 12/12 |
| nominal | 普通 InEKF | **0.346125** | 0.383262 | 0.201253 | **0.374597** | 0.469439 | 12/12 |
| nominal | 松耦合增广 | 0.404446 | 0.401514 | 0.334650 | 0.544195 | 0.564690 | 12/12 |
| nominal | 联合 tracker | 0.357123 | 0.383023 | 0.201253 | 0.381490 | 0.470682 | 12/12 |
| nominal | 自适应输出 | 0.357865 | **0.378544** | 0.232977 | 0.494447 | 0.479990 | 12/12 |
| large | 普通 EKF | 0.438736 | 0.477435 | **0.192065** | 0.763727 | **0.465887** | 12/12 |
| large | 普通 InEKF | 0.418764 | 30.725277 | 0.229953 | **0.430683** | 0.476244 | 10/12 |
| large | 松耦合增广 | 0.407519 | 0.408992 | 0.334650 | 0.544147 | 0.564690 | 12/12 |
| large | 联合 tracker | 0.410627 | 0.846514 | 0.201098 | 0.463289 | 0.471466 | 12/12 |
| large | 自适应输出 | **0.366876** | **0.386690** | 0.232990 | 0.494602 | 0.481124 | 12/12 |

结论需要分开说：

- nominal 下，普通 InEKF 的全程位置/速度中位数更好；普通 EKF 的末段
  位置和速度中位数更好。
- large 下，自适应输出的全程位置中位数和均值最好，并保持 12/12 收敛；
  普通 InEKF 因 2 次 gate 锁死只有 10/12 收敛。
- 自适应输出相对普通 EKF：全程位置、末段位置、全程速度、末段速度分别
  改善 `18/24、8/24、23/24、4/24`。因此不能宣称它全面超过普通 EKF。
- 相对普通 InEKF，对应为 `10/24、8/24（另 3 次相同）、
  3/24、7/24（另 3 次相同）`。优势主要是大初值鲁棒性而非名义稳态速度。

### 5.3 权重实际工作情况

- 12/24 次真实运行实际达到 \(w=1\)；
- 10/24 次在结束时仍为 \(w=1\)；
- 单次运行的全 tracker 样本占比最高 0.333；
- 计入初始 capture 时段后，24 次平均权重的中位数为 0.2129。

这说明实现确实完成了“健康后 0.3 -> 1.0”，同时不是为了提高表格数值而
对所有轨迹强制全权接管。

## 6. 大初值合成 Monte Carlo

固定随机种子 20260731，20 次：

\[
p_0\sim\mathcal U([-50,50]^3)\ {\rm m},\qquad
v_0\sim\mathcal U([-10,10]^3)\ {\rm m/s}.
\]

成功定义为末 5 s 位置 RMSE 小于 1 m。

| 方法 | 成功数/20 | 末 5 s RMSE 中位 (m) | 全程 RMSE 中位 (m) |
|---|---:|---:|---:|
| 普通距离 EKF | 0 | 194.162 | 125.488 |
| 普通 TWR-InEKF | 0 | 194.533 | 125.641 |
| 松耦合增广 capture | 20 | 0.378612 | 15.972056 |
| 重置后的联合 tracker | 17 | 0.755034 | 21.337462 |
| 自适应两滤波器输出 | 20 | 0.378612 | 15.972056 |

20 s 合成试验仍处于规定的捕获成熟期，自适应输出逐样本等于 capture。
这正是预期保护行为：没有用 17/20 的 tracker 破坏 20/20 的捕获结果。

## 7. 测试、运行时间与可视化检查

`.\run_deep_coupled.ps1 -Mode test`：

- `compileall` 通过；
- 20/20 `unittest` 通过；
- 覆盖 capture 逐元素等价、普通 EKF/InEKF 基线等价、健康授权、
  0.3 -> 1.0 平滑爬升、退化回落、失锁回退和真实数据无退化。

正式真实五方法批测墙钟时间约 175.82 s，24 条 CSV 结果均可解析且所有
核心数值有限。

已用原分辨率检查：

- `results/deep_coupled_full/real/paired_deltas.png`：四项 24 根柱均在零线下；
- `results/deep_coupled_full/real/all_methods_comparison.png`：对数坐标完整显示
  普通 InEKF 和 tracker 的大初值离群点；
- `results/deep_coupled_full/real/representative/trajectory_and_errors.png`：
  平滑健康度和实际权重清晰显示 0.3 平台、持续授权和 5 s 爬升至 1；
- `results/deep_coupled_full/synthetic/representative.png`：普通方法发散与
  capture/自适应输出收敛均完整显示。

`tex/deep_coupled_two_filter_twr.tex` 已用 TeX Live 2026 XeLaTeX 连续
编译两遍，生成 8 页 PDF；最终日志无 overfull/underfull、未定义引用或
待重跑警告。随后用 Poppler 将全部 8 页渲染为 PNG 并逐页检查，公式、
两张结果表和四张结果图均未越界、裁切或重叠。

## 8. 文件清单

实现：

- `scripts/estimation/deep_coupled.py`
- `scripts/estimation/deep_coupled_experiment.py`
- `tests/test_deep_coupled.py`
- `run_deep_coupled.ps1`
- `README.md`

正式结果：

- `results/deep_coupled_full/acceptance_summary.json`
- `results/deep_coupled_full/real/summary.json`
- `results/deep_coupled_full/real/paired_metrics.csv`
- `results/deep_coupled_full/real/all_methods_summary.csv`
- `results/deep_coupled_full/real/representative/`
- `results/deep_coupled_full/synthetic/summary.json`
- `results/deep_coupled_full/synthetic/monte_carlo_metrics.csv`

数学说明：

- `tex/deep_coupled_two_filter_twr.tex`
- `tex/deep_coupled_two_filter_twr.pdf`

## 9. 限制

1. 本轮按要求没有解决 \(q,s,h\) 偏离物理流形；capture 的该固有风险仍在，
   但健康 tracker 全权接管后不再进入稳态位置/速度输出。
2. 两个滤波器使用相同 IMU/TWR，却未维护滤波器间互协方差；输出加权不带
   严格一致的融合协方差，也不反馈到滤波器内部。
3. 健康阈值、20 s 成熟期和时间常数基于当前描述性数据调整，没有独立留出集；
   24/24 无退化是已测回归结果，不是任意数据上的数学保证。
4. TU Delft 只提供位置真值，不能量化姿态和 IMU 偏置的真实改善。
5. 固定 `range_bias=-0.18 m`，未建立逐锚点 NLOS/时变偏置模型。
6. 当前结果不构成完整有偏 IMU–UWB 系统的 AGAS/UGES 证明。
