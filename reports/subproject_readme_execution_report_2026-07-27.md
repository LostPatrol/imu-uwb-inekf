# 子工程 README 整理执行报告

日期：2026-07-27  
工作区：`E:\0syncthing\Code\InEKF`

## 1. 完成内容

已为三个实际子工程重写或补全 UTF-8 中文 README：

1. `imu-uwb-inekf/README.md`
2. `imu-uwb-inekf-acc/README.md`
3. `imu-uwb-inekf-fix/README.md`

只读原作者目录 `imu-uwb-inekf - backup` 未修改。该目录的 Git 状态中只有任务开始前
已经存在的未跟踪 `.vscode/`；本任务没有写入备份源码或 README。

## 2. README 的定位

### 2.1 `imu-uwb-inekf`

将其明确为当前完整改进主线，按代码与冻结实验说明：

- `[theta,v,p,b_g,b_a]` 偏置/状态顺序和 SO(3) 小角公式修复；
- 右不变初始协方差映射、坐标一致 UWB NIS 和全圆 yaw 先验；
- 开机首次静止段检测、陀螺/加计可观测性分离与 ZARU；
- 精确三次状态转移、稀疏过程噪声、rank-one UWB 更新、直接 Jacobian、
  `Gamma1/Gamma2` IMU 积分；
- 完整 15 状态、float64 的 Numba 固定尺寸编译核；
- 代表数据精度、速度收益以及保守 yaw 先验的姿态瞬态代价。

### 2.2 `imu-uwb-inekf-acc`

将其明确为“作者算法纯计算等价加速”，特别说明：

- 作者 `inekf.py` 与备份 SHA-256 均为
  `97142a58a203b100fc539c2581024516fa585efe3329f9fb13af8b19f58045b0`；
- 它有意保留作者的偏置排列、坐标/预门限、噪声和每次 IMU/UWB 调用；
- 只改变事件计划、固定缓冲区、Numba 热循环和等价结构化计算；
- 79 个完整 CSV 的中位加速为 38.9794x，拒绝决定变化为 0，最大 RMSE
  绝对差不超过 `6.05e-15`；
- 速度提升不能写成定位精度提升，也不能把该目录当作已修复生产版。

### 2.3 `imu-uwb-inekf-fix`

将其明确为最小论文一致性修正加偏置参数实验：

- 说明作者版偏置块含义不一致及统一后的 `[b_g,b_a]` 排列；
- 说明左雅可比零角一阶符号修正；
- 列出四个偏置先验/随机游走参数的前后值和选取依据；
- 如实写明 const1 四项 RMSE 改善 2.651%--8.416%；
- 同时写明 const4、180° 初始 yaw 压力测试中位置/速度/yaw 退化，以及加计
  `b_y` 后半程持续爬升，避免将局部收益外推成普遍改进。

## 3. 证据核对

README 数字与以下既有报告和机器可读结果交叉核对：

- `task01_execution_report_2026-07-22.md`
- `paper_code_alignment_fix_report_2026-07-23.md`
- `bias_parameter_revision_report_2026-07-23.md`
- `coordinate_consistency_fix_report_2026-07-23.md`
- `task02_algorithm_optimization_report_2026-07-23.md`
- `task03_compiled_optimization_report_2026-07-24.md`
- `fair_author_original_vs_latest_report_2026-07-24.md`
- `author_original_compute_only_acceleration_report_2026-07-24.md`
- `full_all_79_author_vs_acc_final_conclusion_2026-07-24.md`

所有 README 中引用的上述报告、相关 LaTeX 源文件和 PDF 均已确认存在。

## 4. 图片检查

重新目视检查了：

- `imu-uwb-inekf/results/coordinate_fix_final/03_representative_manual3.png`：
  修复前轨迹离开真实区域数百米、累计拒绝持续上升；修复后位置误差主要为亚米级，
  yaw 从 180° 初差稳定收敛，无异常跳变。
- `imu-uwb-inekf-acc/results/full_all_79_parallel/summary.png`：
  79 文件加速分布中位约 39x，单文件约 31x--48x；const4 原算法误差分布较宽。
- `imu-uwb-inekf-fix/results/bias_parameters_const1_full/comparison.png`：
  两条轨迹连续且接近，修订版 RMSE 汇总较低，后半程姿态/yaw 改善可见。
- `imu-uwb-inekf-fix/results/bias_parameters_const4_60s_yaw180/bias_comparison.png`：
  加计 `b_y` 在约 35 s 后持续上升至约 `0.16 m/s²`，支持 README 中的限制说明。

## 5. 文档验证

- 三个 README 均可用 PowerShell `Get-Content -Encoding UTF8` 正常读取；
- 未发现 Unicode replacement character；
- README 行数分别为 202、180、163 行；
- 引用的报告、结果图及 LaTeX/PDF 路径均存在；
- 原作者备份与 `acc` 的核心 `inekf.py` 哈希一致；
- 本任务只改文档，没有修改算法代码，因此没有重新执行数据集回放或自动测试。

## 6. Git 与既有修改说明

任务开始前主工程已经存在 `.gitignore`、环境文件、估计器、实验、结果和测试等未提交
修改；`fix` 也已经存在两个算法文件的未提交修改。此次只编辑三个 README，并新增本
报告及 `MEMORY.md` 记录，没有覆盖或回退任何既有源码修改。

## 7. 最终结论

三个 README 现在分别清楚回答了“相对作者版本改了什么、为何修改、实测效果如何、
哪些结论不能外推、如何复现、应在什么场景使用该子工程”。其中：

- 主工程是完整修复/优化版本；
- `acc` 是严格隔离算法变化的计算加速版本；
- `fix` 是最小一致性和偏置参数实验版本；
- `backup` 仍保持只读基线。
