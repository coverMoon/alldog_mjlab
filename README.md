# AllDog MjLab

AllDog MjLab 是一个基于 [MjLab](https://github.com/mujocolab/mjlab) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 的四足机器人强化学习训练工程，涵盖机器人模型配置、运动任务构建、PPO / HIM 策略训练、平地到复杂地形的续训，以及 TorchScript 策略导出。

项目将机器人资产、训练任务和强化学习算法分层组织，并通过明确的 observation / action contract 与部署程序对接。当前主要支持 Black 四足机器人的速度跟踪训练；其他机器人平台正在逐步接入。

## Features

- **MjLab 原生训练流程**：基于 Entity、Manager、Actuator 和 Sensor 配置任务，使用统一的 `train` / `play` 命令。
- **Black locomotion**：提供 flat 和 rough 两类任务；rough 包含斜坡、离散障碍、上下台阶及地形课程学习。
- **PPO 与 HIM**：支持普通 PPO、HIM 历史观测与估计器，以及对应的 checkpoint 恢复流程。
- **阶段式训练**：支持同任务续训、flat → rough 完整续训，以及 PPO → HIM warm start。
- **策略回放与导出**：通过 MuJoCo native / Viser 查看策略，并导出可独立加载的 TorchScript actor。

**当前范围：** 已注册的训练任务为 `black-flat`、`black-rough`、`black-flat-him` 和 `black-rough-him`。Black rough / HIM 的长训练收敛与定量评估仍在推进中。Wolf 目前仅有待完善的 MJCF 描述，尚未注册训练任务；BlackW 也尚未迁移。

## Getting Started

### 1. 安装

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/)；正式 GPU 并行训练需要与 MjLab 兼容的 NVIDIA CUDA 环境。仓库通过 `uv.lock` 固定依赖，当前使用 **MjLab v1.6.0** 和 **RSL-RL 5.4.2**。

~~~bash
git clone https://github.com/coverMoon/alldog_mjlab.git
cd alldog_mjlab
uv sync --frozen
~~~

后续命令均在仓库根目录运行，通过 `uv run` 自动使用项目虚拟环境，无需单独创建 `train.py` 或 `play.py`。

### 2. 训练 Black flat

从随机初始化开始训练平地 PPO：

~~~bash
uv run train black-flat
~~~

训练日志和 checkpoint 默认保存在：

~~~text
logs/rsl_rl/black_velocity/
└── <timestamp>_flat/
    ├── model_*.pt
    └── ...
~~~

如果 GPU 显存不足，可以先降低并行环境数量，例如：

~~~bash
uv run train black-flat --env.scene.num-envs 1024
~~~

### 3. 从 flat 续训到 rough

推荐先训练出可以稳定运动的 flat 策略，再基于其 checkpoint 进入复杂地形：

~~~bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '.*_flat$'
~~~

这条命令会查找匹配的最新 flat run，并使用其中最新的 checkpoint。rough 训练会恢复 actor、critic、optimizer 等训练状态，同时新建 rough 地形环境。输出保存在新的 `<timestamp>_rough` run 中。

如需从指定 checkpoint 开始：

~~~bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '<flat-run>' \
    --agent.load-checkpoint 'model_500.pt'
~~~

### 4. 回放策略

指定训练 checkpoint，启动 MuJoCo 回放：

~~~bash
uv run play black-rough \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<rough-run>/model_500.pt'
~~~

默认 `--viewer auto`：有桌面显示环境时使用 native viewer，否则使用 Viser。若想在浏览器中调整速度指令并查看运行信息，可显式指定：

~~~bash
uv run play black-rough \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<rough-run>/model_500.pt' \
    --viewer viser
~~~

将示例中的 `<flat-run>`、`<rough-run>` 和 checkpoint 编号替换成实际目录或文件名。需要播放 flat 或 HIM 策略时，将 task ID 和 checkpoint 路径换成对应任务即可。

**缺省 checkpoint 自动注入**：本仓库的 `play` 是项目级入口（`alldog_mjlab/utils/play.py`，覆盖 mjlab 同名命令），显式传 `--checkpoint-file` 或 `--wandb-run-path` 时行为与 mjlab 原版一致；两者都省略时，自动按 task runner 配置（experiment / load_run regex）在 `logs/rsl_rl/` 下找该任务对应 run 目录中最新的目录内 step 最大的 `model_*.pt` 并注入：

~~~bash
uv run play wolf-flat-him --viewer viser
# -> [INFO] 未指定 --checkpoint-file：使用默认最新 checkpoint logs/rsl_rl/wolf_velocity/<...>_wolf_flat_him/model_499.pt
~~~

### 5. 导出 TorchScript 策略

从最新的 rough checkpoint 导出 actor：

~~~bash
uv run export --task-id black-rough
~~~

默认输出路径为：

~~~text
logs/rsl_rl/black_velocity/<rough-run>/exported/policy.pt
~~~

也可以明确指定来源和输出目录：

~~~bash
uv run export --task-id black-rough \
    --load-run '<rough-run>' \
    --checkpoint model_500.pt \
    --output-dir ./exported
~~~

导出程序会重新加载 TorchScript 产物进行数值校验。训练使用的 `model_*.pt` 保存训练状态；`policy.pt` 是面向推理的 actor，两者不能直接互换。

## Training Workflows

普通 PPO 的常见训练方式如下：

| 目标 | 命令 |
| --- | --- |
| 新训练 flat | `uv run train black-flat` |
| 继续 flat | `uv run train black-flat --agent.resume True` |
| 新训练 rough | `uv run train black-rough` |
| 继续 rough | `uv run train black-rough --agent.resume True` |
| flat → rough | `uv run train black-rough --agent.resume True --agent.load-run '.*_flat$'` |

不显式指定 `--agent.load-run` 时，同阶段续训默认查找对应后缀的最新 run（`_flat` 或 `_rough`）。可用 `--agent.load-checkpoint` 指定具体 checkpoint。

常用覆盖参数：

~~~bash
--env.scene.num-envs 2048       # 并行环境数量
--agent.max-iterations 1000     # 本次训练轮数；resume 时表示额外增加的轮数
--agent.save-interval 100       # checkpoint 保存间隔
--agent.seed 43                 # 随机种子
--agent.run-name experiment     # 自定义 run 后缀
--agent.logger tensorboard      # 指标记录后端（见下）
~~~

这些参数附加在 `uv run train <task-id>` 后面。默认训练参数在 `src/alldog_mjlab/tasks/velocity/black/black_config.py` 中维护，CLI 显式参数优先。

**`--agent.logger`（指标记录后端）：**

| 选项 | 行为 | 适用场景 |
| --- | --- | --- |
| `tensorboard` | 仅写本地 TensorBoard events（`{log_root}/{experiment_name}/{run}/`）；checkpoint 只存本地 | 推荐：本地训练 / 无外部服务依赖 |
| `wandb`（默认） | 指标 / 配置 / 代码状态上传 Weights & Biases，项目取 `--agent.wandb-project`（默认 `mjlab`）；checkpoint 可通过 `--agent.upload-model`（默认 True）同步上传 | 远程看板、多 run 对比 |

说明：

- 不显式传 `--agent.logger` 时用配置默认 `wandb`，需要 wandb 账号登录；离线流程请显式传 `--agent.logger tensorboard`；
- `--agent.wandb-tags` 仅在 wandb 下生效；`--agent.upload-model False` 可保留指标上传但不存储模型；
- 两种后端都写同一 run 目录结构（`logs/rsl_rl/<experiment_name>/<时间戳>_<run>/`），checkpoint 恢复与导出不受后端选择影响。

**注意：** resume 时 `--agent.max-iterations` 表示从 checkpoint 继续训练多少轮，并非最终累计 iteration 编号。

**注意：** resume 时 `--agent.max-iterations` 表示从 checkpoint 继续训练多少轮，并非最终累计 iteration 编号。

### HIM 训练

HIM 使用历史观测与估计器，训练 task 为 `black-flat-him` 和 `black-rough-him`。可以从头训练 flat HIM，也可以先用 flat PPO checkpoint 初始化：

~~~bash
# 从随机初始化训练 flat HIM
uv run train black-flat-him

# 使用已训练的 flat PPO 初始化 flat HIM（warm start）
uv run train black-flat-him \
    --agent.warm-start True \
    --agent.load-run '.*_flat$'

# 从 flat HIM 完整续训 rough HIM
uv run train black-rough-him \
    --agent.resume True \
    --agent.load-run '.*_flat_him$'
~~~

`warm-start` 用于初始化新 HIM run，不恢复原 PPO 的 optimizer 与 iteration；`resume` 用于恢复现有 HIM 训练状态，两者不可同时使用。rough HIM 不支持直接从 rough PPO warm start。

HIM 同样支持 `play` 和 TorchScript 导出：

~~~bash
uv run export --task-id black-flat-him
uv run export --task-id black-rough-him
~~~

### Wolf 训练

Wolf 轮足机器人（16 执行器：12 腿关节 + 4 轮）已注册平地与复杂地形任务
（任务实现完全独立，不依赖 Black 任务代码）：

~~~bash
# 从随机初始化训练 wolf flat PPO
uv run train wolf-flat

# 从随机初始化训练 wolf flat HIM
uv run train wolf-flat-him

# 用 wolf-flat PPO checkpoint 初始化 wolf HIM（warm start，仅接受 wolf 源）
uv run train wolf-flat-him \
    --agent.warm-start True \
    --agent.load-run '.*_wolf_flat$'

# 继续 wolf flat PPO / HIM
uv run train wolf-flat --agent.resume True

# wolf-flat PPO 续训 wolf rough（模型/optimizer 恢复；速度课程 range 保留）
uv run train wolf-rough --agent.resume True \
    --agent.load-run '.*_wolf_flat$'

# wolf-flat HIM 继续 wolf rough HIM（HIM full resume；速度课程 range 保留）
uv run train wolf-rough-him --agent.resume True \
    --agent.load-run '.*_wolf_flat_him$'

# 导出（PPO：[1,53]→[1,16]；HIM：[1,318]→[1,16]）
uv run export --task-id wolf-flat
uv run export --task-id wolf-flat-him
~~~

动作契约：16-D raw action，每腿交错 `[hip thigh calf wheel]`；腿
`q_target = q_default + 0.20*raw`，轮 `dq_target = sign*10*raw`
（符号 FL +1 / FR -1 / RL +1 / RR -1，正值 = 机身前进）。HIM actor 输入为
6 × 53 历史（部署侧 newest→oldest 展平 318 维）。速度指令由性能驱动课程从
±1 m/s 扩展到 ±4 m/s。wolf-rough 复用 flat 的完整 observation/action 契约
（actor 53-D / critic 56-D，无高度图输入），仅地形、地形课程、base_height
测量方式与越界截断不同。rough 训练收敛与曲线质量评估尚未进行。

## Tasks

| Task ID | 算法 | 场景 | 状态 |
| --- | --- | --- | --- |
| `black-flat` | PPO | 平地速度跟踪 | 已实现并完成基线验证 |
| `black-rough` | PPO | 复杂地形及地形课程 | 已实现，长期训练评估持续进行 |
| `black-flat-him` | HIM / PPO | 平地、历史观测与环境估计 | 训练链路已实现，长期效果待评估 |
| `black-rough-him` | HIM / PPO | 复杂地形、历史观测与环境估计 | 续训链路已实现，长期效果待评估 |
| `wolf-flat` | PPO | 平地速度跟踪（轮足） | 训练链路已实现，长期效果待评估 |
| `wolf-flat-him` | HIM / PPO | 平地速度跟踪（轮足、历史观测与环境估计） | 训练链路已实现，长期效果待评估 |
| `wolf-rough` | PPO | 复杂地形速度跟踪（轮足，flat 契约续训） | 已注册，训练效果评估未开始 |
| `wolf-rough-him` | HIM / PPO | 复杂地形（轮足、历史观测与环境估计） | 已注册，训练效果评估未开始 |

Black 四个任务共享 Black 的机器人模型和底层控制配置。flat 与 rough 的主要差异在地形、地形课程和相应奖励/终止设置；普通 PPO 与 HIM 使用不同的 actor 输入契约。Wolf 四个任务共享 Wolf 的机器人资产（IMU 原生观测）与轮足执行器契约，速度指令范围与 Black 独立，且不依赖 Black 任务代码（两套任务实现完全独立）。

## Configuration

项目代码按职责组织：

~~~text
src/alldog_mjlab/
├── robots/               MJCF、关节、执行器和默认姿态
│   ├── black/
│   └── wolf/
├── tasks/
│   └── velocity/
│       ├── black/        环境组装、奖励、观测、动作、地形与训练配置
│       └── wolf/         轮足机器人平地/复杂地形任务（wolf_config.py 为参数入口；
│                            实现 task local，不依赖 black/）
├── algorithms/
│   └── him/              HIM estimator、policy、storage 与 PPO 更新
└── utils/                策略导出等工具
~~~

调节 Black 训练参数，通常从以下文件开始：

- [`black_config.py`](src/alldog_mjlab/tasks/velocity/black/black_config.py)：环境数量、控制周期、PD 相关 task 参数、速度指令、奖励权重、reset、随机化、地形和训练超参数。
- [`env_cfgs.py`](src/alldog_mjlab/tasks/velocity/black/env_cfgs.py)：将机器人、Manager terms、action / observation / reward 等组装为 MjLab task。
- [`black_constants.py`](src/alldog_mjlab/robots/black/black_constants.py)：机器人资产、关节顺序、默认姿态和 actuator 配置。

普通训练参数可以按需调整；joint order、observation layout、action semantics、控制频率及 scaling 属于部署接口契约，修改时需要同步验证训练与部署端的一致性。

## Policy Export Contract

Black 策略输出 12-D raw action，按 **FL → FR → RL → RR**、每腿 **hip → thigh → calf** 排列；wolf 策略输出 16-D raw action，每腿 **hip → thigh → calf → wheel** 交错。部署侧必须显式完成模型关节顺序到策略顺序的映射。

| 策略 | TorchScript 输入 | TorchScript 输出 |
| --- | --- | --- |
| black PPO | `float32 [1, 45]`，单帧 actor observation | `float32 [1, 12]` |
| black HIM | `float32 [1, 270]`，6 × 45 历史帧展平，newest → oldest | `float32 [1, 12]` |
| black-rough PPO | `float32 [1, 45]`，同上 | `float32 [1, 12]` |
| black-rough HIM | `float32 [1, 270]`，同上 | `float32 [1, 12]` |
| wolf-flat | `float32 [1, 53]`，单帧 actor observation（IMU 原生） | `float32 [1, 16]` |
| wolf-flat-him | `float32 [1, 318]`，6 × 53 历史帧展平，newest → oldest | `float32 [1, 16]` |

Black 当前的腿部位置目标和控制周期为：

~~~text
q_target = q_default + 0.25 * raw_action
physics dt = 0.005 s
policy dt  = 0.020 s (50 Hz)
~~~

导出的策略只定义 policy inference；观测构造、关节映射、动作执行以及实机安全限位由部署端负责。训练与部署通过显式 policy I/O contract 对接，不要求依赖特定部署工程的内部实现。

## References

- [MjLab](https://github.com/mujocolab/mjlab) — 仿真与任务管理框架（本仓库锁定 v1.6.0）
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — PPO 训练框架
- [HIMLoco](https://github.com/InternRobotics/HIMLoco) — HIM 算法参考
- [quadruped_control](https://github.com/coverMoon/quadruped_control) — 与本仓库策略契约对接的运动控制工程
