<!-- Dataset locations, environment setup, and research reproduction entry points. -->
# 数据与复现

仓库根是完整研究工作区；主代码仍在 `imu-uwb-inekf`。各历史子工程和原作者对照
保留同级关系，已有实验直接使用这些相对路径。无需重新建立多个 Git 仓库。

## Python 环境

推荐 Python 3.11，使用主工程的 `conda/conda-env.yaml` 建立环境并激活。
已使用的研究环境包含 NumPy 1.26.4、Numba 0.61.2、SciPy、pandas 和 Matplotlib。
环境文件定义运行依赖，并非每个平台的完整锁文件。

北京电脑现有环境为 `D:/conda-envs/imu-uwb-inekf`。在那里直接调用解释器时，
需在当前进程 PATH 前置该环境及 `Library/bin`、`Scripts`；其他电脑优先使用
`conda activate imu-uwb-inekf`，无需复用北京电脑的绝对路径。

研究脚本使用进程并行时设置：

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
```

## 外部数据

| 数据 | 获取入口 | 既有默认位置 |
| --- | --- | --- |
| UTIL IMU/UWB 数据 | https://utiasdsl.github.io/util-uwb-dataset/ | `E:/datasets/dataset/flight-dataset` |
| TU Delft 测距/惯性数据 | https://doi.org/10.4121/14827680 | `E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset` |

Git 不包含原始记录。主工程 CLI 用 `-i` 指定输入文件；共同协议可用
`--dataset` 指定 TU Delft 目录；恢复研究先设置 `$env:INEKF_TWR_DATASET = '本机数据目录'`，
再执行 prepare 阶段生成本机缓存。未设置环境变量时，保留北京电脑的既有默认位置。
运行前执行对应入口的 `--help`，不要把北京电脑的数据盘路径当作其他电脑的固定要求。
部分历史批处理脚本仍保留当时的数据路径；复现这些旧实验时需调整其路径常量。

## 验证与实验入口

以下命令从 `imu-uwb-inekf` 子目录执行：

```powershell
python tests/test_inekf.py
python -m unittest discover -s experiments/roadmap_stage1 -p 'test_*.py' -v
python -m unittest discover -s experiments/recovery_bootstrap -p 'test_*.py' -v
python experiments/recovery_bootstrap/run_research.py --help
```

真实冷启动和偏置实验的完整命令见
[恢复研究说明](../imu-uwb-inekf/experiments/recovery_bootstrap/README.md)。
完整回放需要外部数据；下载仓库中的结果表格不能替代重新运行。

## Git 更新

```powershell
git status --short
git pull --ff-only
```

存在本地源码修改时，先提交或暂存自己的修改，再更新。无需同步 `.git` 之外的虚拟环境
或过程目录。Syncthing 的 InEKF 黑名单可以保留，Git 作为代码与研究材料的同步入口。

新增实验输出默认忽略。确需保存一套新的正式结果时，选择其协议、指标表、核验摘要及
论文使用的图片，使用 `git add -f -- <具体文件>` 加入；不要强制加入整个 `results`。
现有已跟踪的结果文件会继续被 Git 跟踪。
