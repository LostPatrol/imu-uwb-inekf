<!-- Current conference manuscript scope, implementation map, and reproducible evidence. -->
# 会议稿修订

2026-11-11 版保留两项核心贡献：惯性形状专门化产生低成本航向候选，以及多项式干扰吸收
偏置、同时削弱航向信息的条件。正文 7 页，补充材料 3 页，中文数学说明 8 页。
当前未选目标会议；采用匿名 IEEE 会议格式，具体页限与匿名要求仍待核对。
正文以三幅彩色图建立视觉主线：窗口模型与投影流程、实际成本取舍、真实日志的目标与位置结果。
文字重点陈述方法和实验成果，数学条件就近定义，适用范围集中说明；细碎重跑核验保留在复现材料。

## 实现与证据

| 内容 | 入口或结果（相对主工程） |
| --- | --- |
| 惯性预积分、候选与原始距离精化 | `experiments/recovery_bootstrap/initializer.py`、`run_research.py`，本轮未改算法核 |
| 五策略实际成本对照 | `experiments/recovery_bootstrap/budget_candidate_study.py` |
| 新仿真结果 | `results/recovery_candidate_budget_20261111` 内 `metrics.csv`、`summary.csv`、`sequential_timing.csv`、`protocol.json`、`validation.json` |
| 真实终点重汇总 | `experiments/recovery_bootstrap/real_window_metrics.py` |
| 新真实结果 | `results/recovery_real_windows_20261111` 内 `per_log_summary.csv`、`per_log_layout_summary.csv`、`overall_frame0_summary.csv` |
| 真实核验 | 同目录 `verification.json`、`endpoint_coordinate_checks.csv`、`coordinate_verification.json` |
| 方法与真实结果配图 | `experiments/recovery_bootstrap/plot_paper_visuals.py` |

两种远场几何各 200 个固定种子，共 2,000 条方法结果；另做 400 次独立顺序计时。
所有策略都先做两次 CV 平移拟合，选优后每个航向只做一次七维精化。
它们的每次求解限额相同，总计算量不同；不把候选数量相同解释成等时预算。
仿真 miss 相对五方法找到的最好目标，阈值 `max(1e-3,1e-5*abs(best))`；
原十二起点参考的差值也单独保存。3 m 高度跨度下 Profile-1 与双起点同为零 miss，
中位耗时 47.1/66.6 ms；弱几何下 Profile-1 更便宜但保留更多 miss。

真实汇总复用原始 3,168 条拟合，不重跑完整滤波。66 个重叠时间窗经过两个布局、
四种等价 yaw 坐标得到 528 个相关配置。按日志报告位置 median/p95/终点 RMSE、目标 miss
和成本，并另报 frame 0 子集。窗口终点位置与锚点/OptiTrack 同坐标，不需逆转 yaw。
12 配置、36 次拟合核验的位置误差最大差 `8.91e-8 m`；独立真值插值一致。
Profile-1 的总体位置尾部略有改善，但 hourglass、random 尾部变差；不声称真实航向改善。

## 复现与论文维护

PowerShell 命令见主工程 [实验说明](../imu-uwb-inekf/experiments/recovery_bootstrap/README.md)。
本机使用根 `.venv/Scripts/python.exe`；新计时硬件 i5-12600KF，
历史真实窗口计时硬件 Core Ultra 7 265KF，不能混合比较。
缓存时钟在约 1.5 秒静态前缀后重新起算，两秒首次窗口对应加载器约 3.50/3.51 秒。
窗口请求取最后已到达样本，真值在实际终点插值。

根 `tex/build_recovery_paper.ps1` 构建三份 PDF；正文数值表使用
`tex/recovery_revision_budget_table.tex`。分日志精确值另保存在
`tex/recovery_revision_endpoint_table.tex` 及结果 CSV，正文以双面板图展示。
`plot_paper_visuals.py` 直接读取保存的分日志 CSV，生成模型示意与真实结果图；
`budget_candidate_study.py --plot-only` 更新彩色成本图。配图与来源清单在
`tex/recovery_bootstrap_figures`，重绘不运行估计器拟合。修改实验后须同步重建相应表图。
不提供 `--cache` 的 `real_window_metrics.py` 只从保存的历史 CSV 重建终点汇总；
`budget_candidate_study.py --summarize-only` 还会做六次求解核验，不能当作纯重绘入口。
英文算法框与补充公式明确 SVD、根筛选、CV 初值与优化停止规则。
精确投影的定理不自动适用于截断 SVD；偏置图的一次/两次外层精化混杂已披露，
匹配模型阶数实验使用相同双平移策略。

未开展可选 IMU 白噪声条件、新日期采集、外部完整系统复现或嵌入式测试。
现有结论限定在已测试的初始化取舍；完整 InEKF 的稳态与断测瞬态无普遍优势。
两套 11 月 11 日证据选择保存协议、逐次指标、汇总、计时及核验，共约 1.67 MiB；
范围见 [结果索引](evidence.md)。原始数据、缓存、完整历史轨迹和结果目录中的重复图不进入快照。
投稿材料仍须按选定会议提供合规、可取得的入口；Git 快照本身不代表匿名材料已经提交。
