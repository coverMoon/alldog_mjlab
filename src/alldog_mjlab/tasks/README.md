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

Black 使用共同 policy family `black_velocity`；MjLab 原生 train 将 run 命名为
`<timestamp>_flat` 或 `<timestamp>_rough`。模型和 PPO 配置共享，stage 默认值来自
`BLACK_CONFIG.runner.flat` / `.rough`。

| 操作 | 命令 |
|---|---|
| 新 flat | `uv run train black-flat` |
| flat → flat | `uv run train black-flat --agent.resume True` |
| 新 rough | `uv run train black-rough` |
| rough → rough | `uv run train black-rough --agent.resume True` |
| flat → rough | `uv run train black-rough --agent.resume True --agent.load-run '.*_flat$'` |

默认续训分别从最新 `*_flat` / `*_rough` run 的最新 `model_*.pt` 完整恢复。
指定 flat checkpoint 开始 rough stage：

```bash
uv run train black-rough --agent.resume True \
    --agent.load-run '2026-10-04_14-00-00_flat' \
    --agent.load-checkpoint 'model_500.pt'
```

Full resume 恢复 actor、critic、normalizer、optimizer（含已调整的学习率）、iteration
及 `common_step_counter`。flat → rough 会新建 rough 环境，不搬运 flat simulator
或 terrain runtime state。`max_iterations` 是恢复后追加的训练轮数：iter 1000 的
checkpoint 配置 5000 时，继续约 5000 轮；不会只训练到 iter 5000。

从最新同 stage run 的最新 checkpoint 导出 actor：

```bash
uv run export --task-id black-flat
uv run export --task-id black-rough
```

默认输出到 `<run>/exported/policy.pt`；可用 `--load-run`、`--checkpoint` 与
`--output-dir` 指定来源或目标。
Export 继续 actor-only 加载，与 critic/full-resume 兼容性独立。历史无 stage 后缀
run 不会被默认选中，且保持原路径；可显式访问：

```bash
uv run export --task-id black-flat --load-run 2026-09-18_19-37-17 \
    --checkpoint model_498.pt
```

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
