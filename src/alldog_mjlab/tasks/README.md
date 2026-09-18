# Task 层结构

```text
tasks/
├── __init__.py             # 加载任务族
└── velocity/
    ├── __init__.py         # 加载机器人任务
    └── black/
        ├── __init__.py     # 注册 black-flat / black-rough
        ├── env_cfgs.py     # 环境、传感器、奖励、事件及 play 覆盖
        ├── params.py       # 人工调参入口（数值）
        ├── rewards.py      # Black reward 数学
        ├── terminations.py # Black stateful termination
        ├── terrain.py      # Black rough terrain primitive 与 generator 组装
        └── rl_cfg.py       # 标准 PPO 网络和训练参数
```

当前注册 `black-flat` 与 `black-rough`（均为标准 PPO）。使用 MjLab 的 `train` /
`play` CLI，不另设训练入口。

- 机器人模型、初始姿态和控制参数归 `robots/black`。
- 环境配置通过 MjLab velocity factory 创建，每次调用返回独立配置。
- `black-rough` 在 `black-flat` 之上覆盖 terrain generator、terrain curriculum、terrain
  scan 与 critic privileged height（critic 由 72 维增至 259 维，actor 仍 45 维），把
  `base_height` 换为 local terrain-relative 语义，并在 training 下末尾追加 native
  `out_of_terrain_bounds` safety truncation（play 移除）；其余 9 项 reward 与 DR /
  command 仍是 flat contract，rough PPO 尚未训练
  （见 `.ai/MIGRATION.md` §13 / §13.4 / §11.4 / §6.4）。
- 自定义 MDP term 在实际需要时增加；HIM 算法应放在 `algorithms/himloco`，任务侧只维护所需观测和运行配置。
- 之前的 HIM 任务原型已移到仓库 `archive/himloco_prototype/`，不参与正式任务发现。
