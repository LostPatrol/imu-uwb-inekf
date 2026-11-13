<!-- Final execution coverage and interpretation of the independent graph adapter. -->
# Jiang-style 图复现：执行结果

本目录是独立的完整 range/IMU 图复现，未定位到作者融合代码，也未实现作者 D-iSAM2。模型、未公开参数的独立选择及近似边界见 README；不能将这里的成败直接归于已发表系统。

最终统一采用左 SO(3) retraction。16 个 native 各发布 10–15 秒六个状态，共 96 行；八个 recovery 各 96 个位置/yaw/置信度压力 case，共 768 个 10 秒解。全部继续到 11–15 秒，形成 4,608 个唯一 `(episode, case_id, state_s)` 输出；不存在补发 10 秒前的平滑状态。只使用共同传感器、测绘锚点、共同标定和外部位置/物理 yaw prior，没有读取评分真值。

终检全部通过：原始共同测距包数量逐集逐时刻相等，没有抽稀；没有 missing/duplicate；续跑状态与已注册 10 秒端点最大差 `8.64e-14 m`；native 保存的 rotation-vector 范数最大 `3.1414508 rad < pi`。

768 个 10 秒解中 104 个达到 80 步上限；4,608 个输出中 583 个达到该上限，全部保留。所有输出有限只代表数值可记录，定位成功、首次可靠时间及持续维持由共同坐标系的独立评分器判定。ISAS-Walk3 recovery 的 native 仍明显发散，未以数值修复掩盖。

同局部坐标、同初值、同限额的 16 个 native 数值/解析 Jacobian 对照中，15 个达到相对目标容限者端点最大差 `2.98e-8 m`；共同封顶的 ISAS-Walk3 recovery 端点差 `0.108 m`、yaw 差 `0.389 rad`，两版都保留，不择优。

固定索引 `0/31/63/95` 的 32 个 fresh warm 计时采用单个 P-core-preferred worker、BLAS=1，每例重新构图。完整构图加求解中位/95%/最大为 `1.560/4.392/5.689 s`，构图和求解中位分别 `0.263/1.237 s`；与注册端点最大差 `2.01e-14 m`。这是已预声明子集的桌面实现计时，不是作者 C++ 的延迟，首次可用还需等待 10 秒数据。

| 证据 | 内容 |
|---|---|
| `evidence/native.csv` | 最终 16 个未知启动/无外部先验 native 到 15 秒 |
| `evidence/stress.csv` | 全 768 个 10 秒压力图 |
| `evidence/stress_10to15.csv` | 全 4,608 个因果压力输出，保留全部负例 |
| `evidence/execution_checks.json` | 覆盖、唯一性、测距同一性、时间及主值旋转核验 |
| `evidence/coverage_and_range_audit.csv` | 逐集逐时刻共同测距数量核验 |
| `evidence/manifold_validation.json` | 独立中央差分、主值重写成本不变及 pi 附近检查 |
| `evidence/native_jacobian_comparison.csv` | 固定设置的数值/解析 Jacobian 首发对照 |
| `evidence/timing_warm_sample.csv` | 32 个固定样本的 fresh prepare/solve/total 实测 |
| `evidence/continuation_resume_audit.csv` | 768 个恢复 case 的 10 秒缓存端点核验 |
| `evidence/implementation_sha256.json` | 实现与既有 CPU 调度 helper 的 SHA256 |

早期加性旋转坐标、旧共同输入以及 16-worker 内存压力试跑均保留在 scratch/preliminary 或转移清单中，不构成最终主结果。完整 15 秒 case checkpoint 在切换并行度时跳过，不重算；主检参数、先验及目标没有为测试结果改变。
