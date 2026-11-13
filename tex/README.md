<!-- Manuscript entry points and compilation requirements. -->
# 论文与数学说明

## 期刊稿

[TAES 第三版](range_inertial_taes_v3/range_inertial_taes_v3.pdf) 将核心贡献定位为未知航向窗口初始化。正文增加 18 个真实窗口的含候选生成计时、Delft 与 ISAS 的同包冷启动对照，并报告可用位置持续率和错误先验恢复的边界。[TeX 源码](range_inertial_taes_v3/range_inertial_taes_v3.tex) · [编译与证据入口](range_inertial_taes_v3/README.md)

[TAES 期刊论文第二版](range_inertial_taes_v2/range_inertial_taes_v2.pdf) 使用学会的 Regular Paper 专用模板。新增 Delft 三角航线的全段动捕参考和连续滤波回放，以等比例转弯示意图解释偏置修正，并逐条核对参考文献署名。[创新点答复与编译说明](range_inertial_taes_v2/README.md) · [第一版期刊稿](range_inertial_taes/range_inertial_taes.pdf)

## 会议稿与数学说明

[最近一次会议修订稿](recovery_window_revision.pdf) 独立保留。更早的会议材料如下。

- [英文论文](recovery_bootstrap_paper.pdf) 与 [TeX 源码](recovery_bootstrap_paper.tex)。
- [精简补充材料](recovery_bootstrap_supplement.pdf) 与 [TeX 源码](recovery_bootstrap_supplement.tex)。
- [中文数学说明](recovery_bootstrap_math_zh.pdf) 与 [TeX 源码](recovery_bootstrap_math_zh.tex)。
- `recovery_bootstrap_refs.bib`：参考文献。
- `recovery_bootstrap_figures`：当前论文配图。

2026-11-11 修订将主张集中于航向候选与多项式干扰边界，正文 7 页，补充材料 3 页，
中文说明 8 页。新增均匀双/四起点的实际成本对照，并按日志补报真实窗口终点位置误差。
正文现有三幅彩色图：方法与投影示意、候选成本取舍、分日志目标与位置结果。
正文补齐标定扰动与信息诊断定义，摘要和结论将耗时改善限定于对应几何的串行桌面计时子集。
目标会议尚未确定；当前使用匿名 IEEE 会议格式，具体页限与匿名政策须在选定会议后核对。
执行范围与复现入口见 [修订说明](../docs/conference-revision.md)。

其他 TeX/PDF 是算法修正、加速、增广方法与共同协议的历史说明，保留供比较；
当前结论以 [研究状态](../docs/research-status.md) 为准。

## 编译

Windows 安装含 IEEEtran、ctex 的 TeX Live 后，从仓库根运行：

精简安装若缺少英文模板、Times 字体或引用排版包，可用 TeX Live 包管理器补齐：

```powershell
& 'D:/Texlive/2026/bin/windows/tlmgr.bat' install ieeetran times cite preprint
```

`preprint` 包提供双栏参考文献平衡所需的 `balance.sty`。

```powershell
& ./tex/build_recovery_paper.ps1 -TexBin 'D:/Texlive/2026/bin/windows'
```

将 `-TexBin` 改为本机编译器目录。英文正文和补充材料使用 pdfLaTeX/BibTeX，中文稿使用 XeLaTeX。
中间文件写入忽略的过程目录，最终 PDF 位于本目录。

配图由保存的结果生成，构建 PDF 时直接使用本目录内的图，不依赖本机数据集。
需要重绘方法示意和真实结果时，从仓库根运行：

```powershell
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/plot_paper_visuals.py
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/budget_candidate_study.py --plot-only
Copy-Item -LiteralPath imu-uwb-inekf/results/recovery_candidate_budget_20261111/candidate_budget_compact.pdf, imu-uwb-inekf/results/recovery_candidate_budget_20261111/candidate_budget_compact.png -Destination tex/recovery_bootstrap_figures
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/analyze_research.py --results imu-uwb-inekf/results/recovery_bootstrap_20261107_final --figures tex/recovery_bootstrap_figures --bias-plot-only
```

重绘命令使用本机保存的结果 CSV；`--plot-only` 只更新图和绘图来源记录，不重新拟合。

历史 TeX 引用的项目内图片也已保留；请在 `tex` 目录下编译这些文档，使相对路径有效。
