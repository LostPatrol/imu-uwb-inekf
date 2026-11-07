<!-- Research workspace entry point and portable setup instructions. -->
# IMU 与 UWB 不变卡尔曼滤波研究

本仓库保存 IMU/UWB 融合定位的算法源码、对照实验、论文和必要结果证据。
主工程包含坐标一致性修正及 Numba 回放核；新的测距惯性初始化研究位于
`imu-uwb-inekf/experiments/recovery_bootstrap`，与成熟 TDoA 路径分开。

| 目录 | 内容 |
| --- | --- |
| [imu-uwb-inekf](imu-uwb-inekf/README.md) | 当前主工程、自动测试、TWR 共同协议与初始化研究 |
| [tex](tex/README.md) | 论文、中文数学说明、配图及可阅读的 PDF |
| [docs/research-status.md](docs/research-status.md) | 当前研究结论、证据入口和适用边界 |
| `imu-uwb-state-augment-deep-coupled-two-filters` | 共同协议依赖的历史捕获器与双滤波器对照 |
| 其他 `imu-uwb-inekf-*` 目录 | 修正、加速、位姿更新、状态增广和 CLIF 历史变体 |
| `imu-uwb-inekf - backup` | 原作者对照源码，保持只读 |
| `reports` | 既有研究分析与实验报告，当前结论以研究状态文档为准 |
| `third_party` | 已保留许可声明的对照实现 |

## 获取与运行

保留仓库内各子目录的相对位置。Windows PowerShell 示例：

```powershell
git clone https://github.com/LostPatrol/imu-uwb-inekf.git InEKF
cd InEKF
conda env create -f imu-uwb-inekf/conda/conda-env.yaml
conda activate imu-uwb-inekf
cd imu-uwb-inekf
$env:MPLBACKEND = 'Agg'
python tests/test_inekf.py
python -m unittest discover -s experiments/recovery_bootstrap -p 'test_*.py' -v
```

数据集单独准备，见 [数据与复现](docs/reproduction.md)。运行实验前阅读对应说明；
已有研究中的时间轴校正、训练标定和数据使用边界不能省略。

## 同步范围

Git 保存源码、测试、环境说明、论文源文件/PDF、被论文引用的图片及选定实验表格。
原始数据、环境、Numba 缓存、完整回放数组、编译中间文件和过程目录由 `.gitignore`
排除。结果快照的范围见 [结果索引](docs/evidence.md)，日常更新使用 `git pull --ff-only`。

主工程源自 [Oursland 的公开实现](https://github.com/oursland/imu-uwb-inekf)，
对照基线为提交 `f3b664e0e8c86b62c6ed2ea2a112261810228e1c`。
使用代码和数据发表结果时，请保留原论文、数据集及第三方实现的引用和许可声明。
