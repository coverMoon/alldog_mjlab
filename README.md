# AllDog MjLab

Black 四足与 Wolf 轮足机器人的速度跟踪强化学习工程，提供平地和崎岖地形上的 PPO/HIM 任务、性能驱动速度课程、领域随机化、checkpoint 续训和 TorchScript 策略导出。

## 安装

需要 Linux、Python 3.12+ 和 `uv`。CUDA 训练需要可用的 NVIDIA GPU 与驱动。

```bash
uv sync --locked
uv run list-envs
```

锁定环境使用 MjLab 1.6.0、RSL-RL 5.4.2、MuJoCo 3.14.0、mujoco-warp 3.14.0，具体依赖以 `uv.lock` 为准。MuJoCo 后端版本通过项目 override 固定。

机器人资产位于 `robots/black/xmls/` 与 `robots/wolf/xmls/`。Wolf 需要本地完整 STL；缺失资产时模型加载报错。STL 的现有 Git 忽略规则保持不变。

## 任务

| 机器人 | 平地 PPO | 崎岖 PPO | 平地 HIM | 崎岖 HIM |
|---|---|---|---|---|
| Black | `black-flat` | `black-rough` | `black-flat-him` | `black-rough-him` |
| Wolf | `wolf-flat` | `wolf-rough` | `wolf-flat-him` | `wolf-rough-him` |

每个机器人的 flat/rough 使用相同的 actor、critic、动作和基础奖励接口。HIM 在基础 actor 观测上启用 6 帧 history，并增加 estimator velocity target 与 terminal successor target。

## 训练

```bash
uv run train black-flat --env.scene.num-envs 4096 --agent.max-iterations 500 --agent.logger tensorboard
uv run train wolf-flat-him --env.scene.num-envs 4096 --agent.max-iterations 500 --agent.logger tensorboard
```

`train` 是命令，Task ID 是其第一个参数。PPO 和 HIM 均支持随机初始化训练。

新训练分别写入 `logs/rsl_rl/black_locomotion/`、`logs/rsl_rl/wolf_locomotion/`。Black 的 run_name 为 `flat / rough / flat_him / rough_him`；Wolf 为 `wolf_flat / wolf_rough / wolf_flat_him / wolf_rough_him`。默认训练不自动加载 checkpoint，也不搜索其他 experiment。

默认采样长度为 Black 每环境 64 步、Wolf 每环境 100 步；保存间隔 50 iterations。网络和算法参数集中在各机器人 `rl_cfg.py`。

同算法续训，显式指定来源 run：

```bash
uv run train wolf-flat-him --agent.resume True --agent.load-run '<run目录名>' --agent.load-checkpoint 'model_500.pt' --agent.command-curriculum-restore full
```

flat→rough 同算法续训使用同一个机器人对应的 rough Task ID，指定 flat run，并选择 `range` 恢复速度范围：

```bash
uv run train wolf-rough-him --agent.resume True --agent.load-run '<flat_HIM_run目录名>' --agent.load-checkpoint 'model_500.pt' --agent.command-curriculum-restore range
```

PPO→HIM warm start 仅支持各机器人的 flat HIM 任务。它复制兼容的 actor、critic 和动作分布参数，重新初始化 estimator、optimizer 和 iteration；课程只恢复速度范围：

```bash
uv run train wolf-flat-him --agent.warm-start True --agent.load-run '<flat_PPO_run目录名>' --agent.load-checkpoint 'model_500.pt'
```

历史实验或任意明确文件位置的 checkpoint 可以显式加载，保存位置仍使用新的 experiment：

```bash
# 同算法完整恢复；不再同时设置 --agent.resume
uv run train wolf-flat-him --agent.checkpoint-path '/absolute/path/model_500.pt' --agent.command-curriculum-restore range

# 从明确的 PPO checkpoint warm start
uv run train wolf-flat-him --agent.warm-start True --agent.checkpoint-path '/absolute/path/ppo_model.pt'
```

`warm_start` 与 `resume` 互斥。PPO/HIM 算法类型、模型结构、机器人来源和课程状态均需兼容；不兼容时加载报错。历史日志和 checkpoint 不会被移动或删除。

课程恢复选项为 `none / range / full / auto`。Black 默认 `range`，Wolf 默认 `none`；`auto` 对同地形阶段恢复完整统计、跨地形阶段只恢复范围。统计版本不兼容时 `auto` 降为 `range`，显式 `full` 报错。当前课程评分为 `tracking_lin_vel` 的实际 episode 累计值，除以 reward weight、policy dt 和 episode 步数。

## Play 与导出

```bash
uv run play wolf-flat-him --checkpoint-file '/absolute/path/model_500.pt' --env.scene.num-envs 1
uv run play wolf-flat-him --env.scene.num-envs 1

uv run export --task-id wolf-flat-him --load-run '<run目录名>' --checkpoint 'model_500.pt'
uv run export --task-id wolf-flat-him --checkpoint '/absolute/path/model_500.pt'
```

省略 play 的 checkpoint 时，只在当前任务 experiment 中搜索匹配 stage 的最新模型。Play 关闭 actor noise、课程和 DR events，保留 reset。Wolf actuator delay 仍按当前默认 profile 配置。

导出默认生成 `<run>/exported/policy.pt`，可用 `--output-dir` 指定其他目录。导出使用 runner 的 TorchScript 能力，重新加载结果并检查数值等价。训练 checkpoint `model_*.pt` 与可独立推理的 TorchScript `policy.pt` 是两种文件。

## 策略与控制接口

| 机器人 | PPO actor | Critic | HIM actor history | TorchScript HIM 输入 | 输出 |
|---|---:|---:|---|---|---:|
| Black | 45 | 259 | `[B,6,45]` | `[B,270]` | `[B,12]` |
| Wolf | 53 | 56 | `[B,6,53]` | `[B,318]` | `[B,16]` |

PPO TorchScript 输入为 `[B,45]` 或 `[B,53]`，dtype 为 float32。HIM 环境 history 为 oldest→newest；部署输入按 newest→oldest 展平。两种机器人的 estimator velocity target 均为 3 维。

Policy 腿顺序固定为 `FL → FR → RL → RR`。Black 每腿 `hip → thigh → calf`；Wolf 每腿 `hip → thigh → calf → foot`，`foot` 是驱动轮。所有部署后端必须显式映射模型顺序与 policy 顺序。

控制周期为 0.02 s（50 Hz），physics dt 为 0.005 s，decimation 为 4。腿动作采用 `q_target = q_default + 0.25 × action`。Wolf 轮动作采用 `dq_target = forward_sign × 10 × action`，符号为 `FL:+1 / FR:-1 / RL:+1 / RR:-1`，轮部 Kp=0。PD 和力矩上限在机器人 constants 中定义。

Black actor 布局为 command、base angular velocity、projected gravity、joint position error、joint velocity、previous action；critic 末尾包含 187 维 height scan。Black critic 的关节项有独立且显式的排列。

Wolf actor 布局为 command、IMU angular velocity、IMU projected gravity、12 腿关节位置误差、12 腿关节速度、4 个带方向符号的轮速、16 维 previous action；critic 在相同无噪声布局后追加 3 维 base linear velocity。

默认 actor running normalization 关闭，critic normalization 开启。部署必须匹配 observation scales、默认姿态、动作尺度和 previous-action 更新时机。部署侧的 clipping、关节保护和 torque safety 与策略动作语义分别配置。

## 代码结构

```text
src/alldog_mjlab/
├── robots/
│   ├── black/black_constants.py + xmls/
│   └── wolf/wolf_constants.py + xmls/
├── tasks/
│   ├── black/
│   │   ├── __init__.py
│   │   ├── env_cfg.py
│   │   ├── rl_cfg.py
│   │   ├── runner.py
│   │   └── mdp/
│   └── wolf/
│       ├── __init__.py
│       ├── env_cfg.py
│       ├── rl_cfg.py
│       ├── runner.py
│       └── mdp/
├── algorithms/him/
└── utils/
```

`env_cfg.py` 直接定义完整环境及奖励表；`rl_cfg.py` 定义 PPO/HIM 网络、算法和 runner 参数；`mdp/` 保存各机器人的自定义函数与局部配置类型。任务之间不引用另一机器人的奖励或环境配置。

Black 的 8 项和 Wolf 的 15 项奖励在各自 `env_cfg.py` 中逐项可见。每个奖励返回每环境 raw 值，统一由 RewardManager 乘以 weight 和 policy dt，不进行总奖励非负裁剪。Wolf 的 collision 按 9 个 body 聚合，feet_stumble 使用 4 个轮碰撞 geom 的世界系 net force。

`train` 默认启用项目的 EPA 数值过滤和终端 warning 控制。可用 `MJWARP_EPA_PATCH=0` 关闭过滤，`MJWARP_SUPPRESS_WARNINGS=0` 保留警告打印；结构化 overflow 指标继续保留。

当前冻结接口、轻量验证状态和已知限制见 [.ai/MIGRATION.md](.ai/MIGRATION.md)。轻量运行通过不代表训练收敛或实机部署已经验证。
