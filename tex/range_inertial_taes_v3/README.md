<!-- Build instructions and evidence for the window-initialization manuscript. -->
# 单标签测距与 IMU 的未知航向位置初始化

本稿围绕如何为一个固定窗口的原始距离优化选取航向候选展开。表 IV 比较普通单起点、剖面航向候选和均匀四起点。第 VI-B 节补充实际串行耗时。表 VI 单独比较没有位置初值时从共同第一份 IMU 输入到首次确认可用位置。第 VII 节讨论有历史检查点的断距恢复，并用有姿态参考的 TDoA 记录检验位置与航向的关系。

[英文论文](range_inertial_taes_v3.pdf) · [窗口用时与逐窗结果](../../imu-uwb-inekf/experiments/radio_inertial_models/flight_seed_timing/README.md) · [启动与错误先验同包对照](../../imu-uwb-inekf/experiments/acquisition_challenge/RESULTS_zh.md)

## 编译

从本目录执行下面四步。模板、参考文献和论文使用的图已随源码附带。投稿时请补入作者姓名、单位和通信信息。

```powershell
$texBin = 'D:/Texlive/2026/bin/windows'
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes_v3.tex
& "$texBin/bibtex.exe" range_inertial_taes_v3
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes_v3.tex
& "$texBin/pdflatex.exe" -halt-on-error range_inertial_taes_v3.tex
```
