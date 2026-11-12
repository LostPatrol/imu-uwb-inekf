<!-- TAES manuscript revision, reviewer response, figure provenance, and compilation. -->
# 单标签测距与惯性位置获取 · TAES 期刊稿第二版

[论文 PDF](range_inertial_taes_v2.pdf) · [LaTeX 主文件](range_inertial_taes_v2.tex) · [实飞轨迹与滤波回放图](figures/flight_trajectory.pdf)

## 审稿人问创新点时

> We address short-window position acquisition from asynchronous ranges to a single tag when gravity is known and heading is unknown. We use the measured inertial trajectory shape to eliminate translation and auxiliary variables in a squared-range relaxation. The remaining unit-circle problem gives one heading candidate for a fit to the original distances. We derive how a rotating body-fixed acceleration offset changes the endpoint-referenced trajectory and when its effect can be absorbed by a world-frame quadratic. On matched eight-anchor Delft windows of 8 s, horizontal endpoint RMSE is 0.113 m with our fit and 0.591 m with range-only quadratic reconstruction. On the same body-bias objective, one candidate matches four uniform starts within tolerance with 62.1% fewer function evaluations.

与 Pacholska 等人的纯测距基函数重建相比，本文使用实测 IMU 形状保留窗口内的转向。与依赖刚性标签几何的初始化相比，单标签靠跨时刻的运动约束航向。相对使用推力、阻力参数的四旋翼方法，本文只需重力对准、IMU 和已测锚点位置。圆约束的全局求解针对松弛候选目标。文中的 InEKF 交接是位置获取的应用验证。第 III、IV 节给出相应推导，表 IV 和图 3 给出两项对照数字。[Pacholska 等](https://doi.org/10.1109/LRA.2020.2970952) · [Goudar 等](https://doi.org/10.1109/LRA.2024.3354623) · [Pfeiffer 等](https://doi.org/10.1109/LRA.2021.3095519)

## 如何读图

- 图 1 在等比例平面上画两种**校正位移**。在相同的终端校正加速度下，两条曲线随转弯逐渐分离。这是解析例子。
- 图 2 按记录配对水平误差。Delft 每条飞行取固定航向坐标下的 22 个布局窗口，QDrone 每条记录取同一端点的三种窗口长度。
- 图 3 在相同线性刻度上显示纯测距二次重建和本文方法。两法使用同一组测距、标定与端点。相应的飞行动力学 EKF、MHE 与其他测距基线见表 V。
- 图 4 选取既定六条 Delft 评测记录中的第一条 triangle。原始日志为 43.82 秒，有效 OptiTrack 参考从回放时钟 −1.50 到 38.62 秒，跨度 40.12 秒。左图显示这段完整参考与 27.01–36 秒的连续估计。右图在整个评测时段以 20 Hz 采样三种策略的水平误差。人为断距区间为 [5, 25) 秒，25.01 秒收到首包，27.01 秒交接。论文表 VI 报告跨飞行中位数。

## 编译与重绘

在本目录运行下列命令。图片已随包附带，编译无需数据集或 Python。

~~~powershell
$bin = 'D:/Texlive/2026/bin/windows'
& "$bin/pdflatex.exe" -halt-on-error range_inertial_taes_v2.tex
& "$bin/bibtex.exe" range_inertial_taes_v2
& "$bin/pdflatex.exe" -halt-on-error range_inertial_taes_v2.tex
& "$bin/pdflatex.exe" -halt-on-error range_inertial_taes_v2.tex
~~~

图 1 由 TeX 依公式绘制。仓库根目录下，下列命令分别从逐窗结果重绘图 2、3，以及从随包的 `trajectory_data.npz` 重绘图 4。

~~~powershell
& .\.venv\Scripts\python.exe .\tex\range_inertial_taes_v2\plot_flight_results.py
& .\.venv\Scripts\python.exe .\tex\range_inertial_taes_v2\plot_trajectory.py
~~~

轨迹包保存实测参考、原滤波输出、评分时间、来源路径和 SHA-256。仓库实验缓存在位时，可指定 `plot_trajectory.py --extract --dataset <Delft数据目录>` 从原始记录重新提取。Delft 原始数据由 S. Pfeiffer、C. de Wagter 和 G. de Croon 以 CC0 1.0 发布。[4TU.ResearchData 数据集](https://doi.org/10.4121/14827680.v1)

## 书目与模板

沿用 IEEE AESS 的 TAES Regular Paper 官方类文件及 BibTeX 样式。该类内部保留 IEEEphot 标识。作者名按首字母在前、姓氏在后排版。官方样稿中作者首行比题名和后续行左移 10 pt。本版保留该缩进，并在书目内放宽右边距，减少窄栏词间空隙。2025 年期刊论文依出版记录使用 C. De Wagter，Delft 数据集依登记记录使用 G. de Croon。投稿前需补入作者署名、单位和通信信息。[TAES 作者指南与模板](https://ieee-aess.org/publications/transactions-aes/author-information) · [IEEE Reference Guide](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf)
