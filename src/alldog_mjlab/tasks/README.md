# Robot locomotion tasks

`tasks/__init__.py` 只导入 `black` 和 `wolf`。各机器人 `__init__.py` 使用 `register_mjlab_task` 注册 flat/rough × PPO/HIM 四个任务。

- `env_cfg.py`：完整的场景、传感器、观测、动作、指令、奖励、事件、终止、地形、课程及仿真配置。
- `rl_cfg.py`：PPO/HIM 网络、estimator、优化算法及 runner 参数。
- `runner.py`：同算法 checkpoint、课程状态和 PPO→HIM warm start 扩展。
- `mdp/rewards.py`：机器人自定义奖励；等价的框架函数直接在环境配置中注册。
- `mdp/events.py`：DR 参数、事件和控制扰动。
- `mdp/curriculums.py`：性能驱动速度课程、统计和版本校验。
- `mdp/him.py`：history、estimator velocity target 和 terminal target。
- `mdp/terrain.py`：自定义地形构造；数值和地形组合在 `env_cfg.py` 中定义。

Black 具有 8 项奖励、12 维动作、45 维 actor、259 维 critic。Wolf 具有 15 项奖励、16 维动作、53 维 actor、56 维 critic。每个机器人的 flat/rough 观测 specification 完全一致，HIM actor 使用相同基础观测的 6 帧 history。

训练 experiment 为 `black_locomotion` 与 `wolf_locomotion`。命令、训练/续训和策略导出示例见仓库根 README。
