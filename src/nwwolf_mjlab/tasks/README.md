# Task 层结构

```text
tasks/
├── __init__.py             # 加载任务族
└── velocity/
    ├── __init__.py         # 加载机器人任务
    └── black/
        ├── __init__.py     # 注册 black-flat
        ├── env_cfgs.py     # 环境、传感器、奖励、事件及 play 覆盖
        └── rl_cfg.py       # 标准 PPO 网络和训练参数
```

当前仅注册 `black-flat`（标准 PPO）。使用 MjLab 的 `train` / `play` CLI，不另设训练入口。

- 机器人模型、初始姿态和控制参数归 `robots/black`。
- 环境配置通过 MjLab velocity factory 创建，每次调用返回独立配置。
- 后续 `black-rough` 在 Black 通用环境设置之上增加粗糙地形覆盖；本阶段不添加占位任务。
- 自定义 MDP term 在实际需要时增加；HIM 算法应放在 `algorithms/himloco`，任务侧只维护所需观测和运行配置。
- 之前的 HIM 任务原型已移到仓库 `archive/himloco_prototype/`，不参与正式任务发现。

本次目录整理保持 `black-flat` 的训练配置、play 配置和 PPO 参数不变。
