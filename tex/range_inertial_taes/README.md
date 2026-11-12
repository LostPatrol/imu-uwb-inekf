<!-- Journal manuscript entry point, template provenance, and figure data. -->
# 单标签测距与惯性位置获取 · TAES 期刊稿

[论文 PDF](range_inertial_taes.pdf) 和 [LaTeX 主文件](range_inertial_taes.tex) 是独立的英文期刊稿。作者署名和单位待补。文档类型为 IEEE Transactions on Aerospace and Electronic Systems 的 Regular Paper。随附的 IEEEtaes.cls 和 IEEEtaes.bst 来自学会原始研究论文模板，保留原文件。卷期、版权和收稿日期由出版方分配，草稿使用题名页脚。[TAES 作者指南](https://ieee-aess.org/publications/transactions-aes/author-information) · [官方模板压缩包](https://confcats-web-assets.s3.amazonaws.com/ieeeaess/documents/TAES+Template.zip)

第 III 节将平方测距中的多项式平移消去，只在单位圆上搜索航向。第 IV 节用旋转机体系加计偏置修正端点轨迹，并推导其与自由二次修正的关系。第 VI 节对比实测终点精度与候选搜索成本。第 VII 节给出与 15 维 InEKF 交接的条件后验，并对照 Oursland 原文中带杆臂的 TDoA 观测。

## 编译

源码包解压后在主文件目录执行下列 PowerShell 命令。两幅实测图已经内置，转弯解析图由 TikZ 从公式直接绘制。编译不需要下载数据集。

~~~powershell
$texBin = 'D:/Texlive/2026/bin/windows'
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes.tex
& "$texBin/bibtex.exe" range_inertial_taes
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes.tex
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes.tex
~~~

完整的 TeX Live 包含所需组件。精简安装需具备 helvetic、times、wrapfig、amsmath、amsfonts、booktabs、pgf 和 hyperref。文献库使用 24 条带 DOI 的记录。

官方分发的 IEEEtaes.cls 内部沿用 IEEEphot 类标识，TeX 日志会显示这一名称。主文件加载的是压缩包中 TAES 原始研究论文目录下的 IEEEtaes.cls。

## 图表来源

仓库中的 plot_flight_results.py 读取现有的逐窗结果。图 2 在 Delft 固定一个航向坐标系，每条飞行合并两个布局各 11 窗。QDrone 每条记录合并同一端点的三种时长。图 3 对齐五种方法的窗口键、事件数、端点和静态无线校准后，按布局和时长计算水平 RMSE。

~~~powershell
# Run from the InEKF repository root.
& .\.venv\Scripts\python.exe .\tex\range_inertial_taes\plot_flight_results.py
~~~

原始计算见仓库下 imu-uwb-inekf/experiments 中的 radio_inertial_models/evidence/paired_acquisition、published_range_only/results、published_window_comparison/evidence 和 unified_window_recovery/results。候选成本的模拟数据位于 imu-uwb-inekf/results/recovery_candidate_budget_20261111。正式署名可在主文件的 author、affil 和论文元数据中填写。
