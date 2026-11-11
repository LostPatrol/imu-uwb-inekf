<!-- Selected frozen evidence and exclusions for repository synchronization. -->
# 结果索引

本仓库保留选定正式实验的协议、CSV/JSON 指标、核验摘要，以及 TeX 实际引用的图片。
原始数据、缓存、完整 NPZ 历史、分片回放和重复诊断图留在原工作区，不进入 Git。
下表是可直接同步的结果快照；已保存的核验摘要记录当时的验证，完整轨迹核验需重新回放。

| 结果目录 | 文件数 | 大小 |
| --- | ---: | ---: |
| `imu-uwb-inekf-acc/results/fair_author_original_vs_acc_const1_30s` | 2 | 444 KiB |
| `imu-uwb-inekf-acc/results/full_all_constellations` | 7 | 187 KiB |
| `imu-uwb-inekf-fix/results/bias_parameters_const1_full` | 3 | 712 KiB |
| `imu-uwb-inekf-fix/results/bias_parameters_const4_60s_yaw180` | 3 | 730 KiB |
| `imu-uwb-inekf-fix/results/paper_alignment_const1_full` | 3 | 401 KiB |
| `imu-uwb-inekf-state-augment/results/task06_author_fair_compare` | 6 | 1109 KiB |
| `imu-uwb-inekf-state-augment/results/task06_full` | 10 | 1115 KiB |
| `imu-uwb-inekf-state-augment/results/task06_per_anchor_robust` | 20 | 1478 KiB |
| `imu-uwb-inekf-state-augment/results/task06_tail_diagnostic` | 3 | 248 KiB |
| `imu-uwb-inekf-update-improve/results/task04_compute_overhead` | 4 | 113 KiB |
| `imu-uwb-inekf-update-improve/results/task04_const1_30s` | 3 | 379 KiB |
| `imu-uwb-inekf-update-improve/results/task04_const1_30s_uwb50hz` | 3 | 387 KiB |
| `imu-uwb-inekf-update-improve/results/task04_const4_manual1_60s` | 3 | 417 KiB |
| `imu-uwb-inekf-update-improve/results/task04_const4_manual1_60s_uwb50hz` | 3 | 410 KiB |
| `imu-uwb-inekf/results/attitude_necessity_const1_yaw000` | 1 | 109 KiB |
| `imu-uwb-inekf/results/attitude_necessity_const1_yaw180` | 1 | 107 KiB |
| `imu-uwb-inekf/results/coordinate_fix_final` | 6 | 378 KiB |
| `imu-uwb-inekf/results/coordinate_fix_full60` | 3 | 83 KiB |
| `imu-uwb-inekf/results/cuda_feasibility_cpu_baseline_2026-07-24` | 2 | 578 KiB |
| `imu-uwb-inekf/results/fair_author_original_vs_latest_const1_30s` | 3 | 713 KiB |
| `imu-uwb-inekf/results/fair_author_original_vs_proposed_const1_full` | 2 | 349 KiB |
| `imu-uwb-inekf/results/manual_author_original_vs_latest` | 1 | 6 KiB |
| `imu-uwb-inekf/results/recovery_bias_20261107_final` | 5 | 38 KiB |
| `imu-uwb-inekf/results/recovery_bootstrap_20261107_final` | 12 | 7817 KiB |
| `imu-uwb-inekf/results/recovery_candidate_budget_20261111` | 7 | 770 KiB |
| `imu-uwb-inekf/results/recovery_geometry_20261107` | 6 | 2334 KiB |
| `imu-uwb-inekf/results/recovery_imu_affine_control_20261107` | 4 | 126 KiB |
| `imu-uwb-inekf/results/recovery_imu_mismatch_20261107` | 4 | 539 KiB |
| `imu-uwb-inekf/results/recovery_imu_polynomial_20261107` | 4 | 128 KiB |
| `imu-uwb-inekf/results/recovery_real_windows_20261111` | 13 | 935 KiB |
| `imu-uwb-inekf/results/recovery_window_candidates_20261107` | 9 | 1135 KiB |
| `imu-uwb-inekf/results/recovery_window_solver_20261107` | 5 | 678 KiB |
| `imu-uwb-inekf/results/roadmap_stage1_20261104_final` | 11 | 1450 KiB |
| `imu-uwb-inekf/results/task01_final_fast_const1_30s` | 3 | 211 KiB |
| `imu-uwb-inekf/results/task01_final_fast_const4_60s` | 3 | 235 KiB |
| `imu-uwb-inekf/results/task02_optimization` | 3 | 488 KiB |
| `imu-uwb-inekf/results/task03_compiled_optimization` | 2 | 579 KiB |
| `imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_delay_full` | 12 | 1445 KiB |
| `imu-uwb-state-augment-deep-coupled-two-filters/results/deep_coupled_time_alignment_self_comparison` | 4 | 130 KiB |
| `imu-uwb-state-augment-deep-coupled-two-filters/results/time_alignment_calibration` | 4 | 259 KiB |

`recovery_bootstrap_20261107_final` 和 `recovery_bias_20261107_final` 是正式真实回放结果，
无 `_final` 后缀的早先同名批次未纳入。历史报告中未列入上表的过程产物、完整轨迹和某些逐样本图，
可能仅在北京电脑本地存在；论文 TeX 的图片依赖已单独保留。

源码与表格保持既有科学含义。仅共同数据位置新增 `INEKF_TWR_DATASET` 环境覆盖，
以及将几何核验脚本从过程目录收纳到实验目录；估计器和优化器公式未随同步整理改写。

## 2026-11-11 修订结果

会议稿的最小复现快照共 20 个文件，约 1.67 MiB；论文实际引用的表格和图片另存于根 `tex`。

- `imu-uwb-inekf/results/recovery_candidate_budget_20261111`：400 个原种子窗口、
  五策略 2,000 条方法结果、400 次顺序计时、两种 miss 参考及核验。
- `imu-uwb-inekf/results/recovery_real_windows_20261111`：复用既有 3,168 条拟合的
  每日志/布局终点误差、frame 0 重汇总、历史成本、36 次窗口重跑与原始真值坐标核验。

前者保留 `protocol.json`、`metrics.csv`、`summary.csv/json`、`sequential_timing.csv`、
`validation.json` 和 `tolerance_sensitivity.csv`。后者保留协议、逐次终点指标、总体及
逐日志/布局汇总（含 frame 0）、历史顺序计时汇总和坐标/重跑核验表与 JSON。
真实窗口的逐次历史计时仍取自已保存的
`recovery_window_candidates_20261107/sequential_timing.csv`；新目录不重复存放它。
原始数据、缓存、NPZ 和结果目录中的重复图片不属于此快照。

新仿真计时使用本机 i5-12600KF；历史真实计时使用 Core Ultra 7 265KF。
本轮未重做 96/60 条完整滤波回放，也未新增独立采集。具体入口与限制见
[会议稿修订](conference-revision.md)。仅重汇总及重绘的命令见
[实验说明](../imu-uwb-inekf/experiments/recovery_bootstrap/README.md#manuscript-visuals)，无需原始数据或优化重跑。
