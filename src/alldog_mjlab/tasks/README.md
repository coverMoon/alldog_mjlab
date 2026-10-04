# Task 层结构

```text
tasks/
├── __init__.py             # 加载任务族
└── velocity/
    ├── __init__.py         # 加载机器人任务
    └── black/
        ├── __init__.py     # 注册 black-flat / black-rough
        ├── env_cfgs.py     # 环境、传感器、奖励、事件及 play 覆盖
        ├── black_config.py # 单一人工训练参数入口
        ├── rewards.py      # Black reward 数学
        ├── terminations.py # Black stateful termination
        ├── terrain.py      # Black rough terrain primitive 与 generator 组装
        └── rl_cfg.py       # BlackConfig → 标准 RSL-RL 配置
```

当前注册 `black-flat` 与 `black-rough`（均为标准 PPO）。使用 MjLab 的 `train` /
`play` CLI，不另设训练入口。

从最新匹配训练 run 的最新 checkpoint 导出 actor：

```bash
uv run export --task-id black-flat
```

默认输出到 `<run>/exported/policy.pt`；可用 `--load-run`、`--checkpoint` 与
`--output-dir` 指定来源或目标。

- 机器人模型、初始姿态和控制参数归 `robots/black`。
- 环境配置通过 MjLab velocity factory 创建，每次调用返回独立配置。
- `black-flat` 与 `black-rough` 共用 terrain scan 和 259 维 privileged critic（末尾
  187 维为高度扫描；actor 均为 45 维）。`black-rough` 在 `black-flat` 之上覆盖 terrain
  generator、terrain curriculum，把 `base_height` 换为 local terrain-relative 语义，
  并在 training 下末尾追加 native
  `out_of_terrain_bounds` safety truncation（play 移除）；其余 9 项 reward 与 DR /
  command 仍是 flat contract；约 500 iteration 的 PPO baseline 已完成，长期收敛与
  定量评估尚未完成
  （见 `.ai/MIGRATION.md` §13 / §13.4 / §11.4 / §6.4）。
- 自定义 MDP term 在实际需要时增加；HIM 算法应放在 `algorithms/himloco`，任务侧只维护所需观测和运行配置。
- 之前的 HIM 任务原型已移到仓库 `archive/himloco_prototype/`，不参与正式任务发现。
