<!-- Execution brief for the independent Toronto TDoA physical-heading branch. -->
# Toronto UTIL 真实航向干预执行简报

已完成4条固定trial4的原始Vicon pose、survey/锚点、杆臂、IMU和TDoA导出，源码及全部数据/结果位于本目录。该支线只评价单标签TDoA，不能替代共同TWR排名。

实现使用未修改的只读Oursland作者类。首段1.5秒外接IMU+TDoA-CV初始化；恢复在15秒，仅向估计初态注入受控XY/真实yaw中心误差，其他状态来自传感器轨迹。完成4×96恢复和4条无外部位置/航向先验首段。回放4worker性能核0/2/4/6，墙时1042.83秒。

全部388轨迹统一补入gyro线性插值的因果发布缓冲：四条最大缓冲1.172/1.177/1.187/1.222ms，原始插值等待中位约0.970ms。只延迟发布，不改状态值/状态时间、不回填；以最新已发布输出对当前Vicon真值，0.5秒网格三格确认持续1秒。输出2/10/15秒XY与真实yaw，含0.25/0.5/1m捕获敏感性。

0.5m主门限在15秒内捕获：tight58/192，wide59/192；10秒XY中位16.39/37.81m，真实yaw中位38.25/36.15°。180°tight0/64、wide15/64捕获，真实yaw中位164.92/99.04°。任务没有实现“大扰动普遍恢复”：原作者core仍存在大量未捕获与大端点误差，所有失败指标均保留。数值异常退出0，不代表精度成功。

四条外接sensor-only首段10秒XY为0.119/0.032/0.163/0.031m，yaw为6.19/6.55/14.32/2.12°。其prefix使用TDoA包，与TWR组IMU-only前缀不同，不能混为同种启动指标。

`audit.json` 全部通过：精确388计划覆盖；原作者core指纹未变；受控XY/yaw中心正确；非目标Z/velocity/bias差为0、tilt差≤1.1e-16；物理协方差映射一致；388时间线因果缓冲全部覆盖。`overview.png` 已人工查看，无布局遮挡。`metrics.csv`、`summary.csv`、`causal_clock.json`及每例NPZ/JSON完整可复核。

没有修改主评分器、Delft包、备份源码或其他分支，没有提交。全局MEMORY及主报告由父任务统一整合，本支线未改动。
