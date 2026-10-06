# AllDog MjLab

AllDog MjLab 是 XJTUROBOCON 四足机器人强化学习训练仓库，基于 MjLab + RSL-RL，
从旧 Isaac Gym / legged_gym / super-dog 工程逐步迁移。当前目标架构以 MjLab 为准，
机器人与任务通过原生 Entity、Manager 和配置机制组织。

本仓库负责机器人任务、训练、策略导出和显式 deployment policy contract。
实机 runtime / backend 在独立 deployment project 中维护，通过策略契约与训练侧对接。

## Current Status

| 能力 | 状态 |
|---|---|
| Black flat PPO | Available |
| Black rough PPO | Available / baseline verified；长期收敛与定量评估尚未完成 |
| TorchScript actor export | Available |
| Black deployment policy contract | Verified；observation/action trace 与 sim2sim 已验证 |
| Black HIM（flat-him / rough-him） | Available；注册 + warm start + flat→rough full resume 已验证，长训练收敛未验证 |
| BlackW | Not yet migrated |
| Real robot backend | 在独立 deployment project 中维护，尚未实现 |

## Quick Start

完成下方安装后，在仓库根目录依次执行主流程。

训练 flat：

```bash
uv run train black-flat
```

从最新 flat checkpoint 继续训练 rough：

```bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '.*_flat$'
```

播放本地 rough checkpoint（桌面环境默认开 MuJoCo native 窗口，网页版用 `--viewer viser`，
详见下文 Playing a Policy）：

```bash
uv run play black-rough \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<rough-run>/model_500.pt' \
    --viewer viser
```

导出最新 rough actor：

```bash
uv run export --task-id black-rough
```

示例中的 `<rough-run>`、`<flat-run>`、`<run>` 和 checkpoint 文件名需替换为实际值。

## Installation

需要先安装 [uv](https://docs.astral.sh/uv/)。Python 要求为 **>= 3.12**，
仓库 `.python-version` 使用 3.12；当前 `uv.lock` 锁定 **MjLab 1.6.0 / RSL-RL 5.4.2**。
常规并行训练使用 NVIDIA GPU，框架要求见 [MjLab v1.6.0](https://github.com/mujocolab/mjlab/tree/v1.6.0)。

```bash
git clone https://github.com/coverMoon/alldog_mjlab.git
cd alldog_mjlab
uv sync
```

项目当前开发与验证基于 MjLab v1.6.0，请优先使用仓库提供的 `uv.lock` 保持环境一致，
不建议自行升级 MjLab 后直接假定兼容。安装后 `uv run` 会使用项目环境；
`train` / `play` 来自 MjLab，`export` 是本项目提供的入口。

## Training Workflow

`black_velocity` 是 checkpoint-compatible policy family，flat / rough 是两个训练阶段。
二者共享 actor、critic 和 PPO 配置，stage 决定环境、run 后缀和默认续训来源。
日志默认组织为：

```text
logs/rsl_rl/black_velocity/
├── <timestamp>_flat/
└── <timestamp>_rough/
```

| Workflow | Command |
|---|---|
| Fresh flat | `uv run train black-flat` |
| Flat → Flat | `uv run train black-flat --agent.resume True` |
| Fresh rough | `uv run train black-rough` |
| Rough → Rough | `uv run train black-rough --agent.resume True` |
| Flat → Rough | `uv run train black-rough --agent.resume True --agent.load-run '.*_flat$'` |

### 常用 CLI 覆盖参数

```bash
--env.scene.num-envs 2048          # 环境数量（默认 BLACK_CONFIG.env.train_num_envs = 4096）
--agent.max-iterations 500         # 追加 iteration 数（resume 时是“再训多少”，非绝对上限）
--agent.seed 43
--agent.save-interval 100          # checkpoint 保存间隔（另：最后一个 iteration 总是保存）
--agent.logger tensorboard
--agent.run-name mytag             # run 目录后缀 → <timestamp>_mytag
--agent.load-run / --agent.load-checkpoint
```

环境数量不影响 checkpoint 兼容性：不同 env 数训练的 run 可以互相 resume。
显式 CLI 参数优先于 `black_config.py` 中的 task 默认值。

同 stage resume 默认选最新匹配的 `*_flat` / `*_rough` run，再选最新 `model_*.pt`。
指定某个 flat run 和 checkpoint：

```bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '<flat-run>' \
    --agent.load-checkpoint 'model_500.pt'
```

Flat → Rough 是 **full resume**，继续恢复 actor / critic / optimizer / normalization /
training iteration 等训练状态，包括 checkpoint 的学习率。rough environment 会重新构造，
不会整体搬运 flat 的 simulator state 或 terrain runtime state。

**MjLab / RSL-RL resume 时，`max_iterations` 表示 checkpoint 之后额外训练的 iteration 数，
不是最终 iteration 编号。** 例如从 iter 1000 resume，`max_iterations=5000` 会再训练约
5000 iterations；对应 CLI 参数为 `--agent.max-iterations 5000`。

## Playing a Policy

使用本地训练 checkpoint 播放 flat 或 rough：

```bash
uv run play black-flat \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<flat-run>/model_500.pt'

uv run play black-rough \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<rough-run>/model_500.pt'
```

HIM task 同样可播（推理只读 actor history，不读 estimator target）：

```bash
uv run play black-flat-him \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<flat-him-run>/model_*.pt'
uv run play black-rough-him \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<rough-him-run>/model_*.pt'
```

### Viewer 选择

```bash
--viewer viser     # 浏览器网页 UI（viser，默认 http://localhost:8080；
                   #   command 滑条 / reward 面板 / checkpoint 热切换）
--viewer native    # MuJoCo 桌面窗口（无 command 控件）
--viewer auto      # 默认：检测到桌面显示环境 → native，否则 → viser
```

`auto` 在桌面会话下落到 native，因此想用网页控制速度必须显式传 `--viewer viser`。

### Play 参数

```bash
--num-envs 4                       # 环境/机器人数量（默认单 env）
--agent zero|random|trained        # policy 模式（zero/random 可做 ad-hoc 检查）
--checkpoint-file '<path>'         # 直接指定 checkpoint 文件
```

play 使用对应 task 的 play configuration，默认关闭 training DR 与 actor observation
corruption；reset events 保留。rough play 保留 terrain generator，关闭 curriculum 和
terrain 越界 truncation。推理仅加载 actor，不使用 critic。

## Inspection / Visualization

只看地形（不训练、不 play、无机器人）：

```bash
# 交互式 MuJoCo 窗口（需 DISPLAY；鼠标左键旋转 / 右键平移 / 滚轮缩放）
uv run python tests/render_black_rough.py --viewer
uv run python tests/render_black_rough.py --viewer --row 9 --col 5   # 聚焦最难的上台阶

# headless 输出图片（俯视全图 + 每列斜视图 + rough slope 三档难度）
uv run python tests/render_black_rough.py --out /tmp/render
```

网格为 10 行（难度 0.0 → 0.9）x 7 列（terrain 类型）：
`flat / smooth_slope_up / smooth_slope_down / rough_slope / discrete_obstacles /
stairs_up / stairs_down`。`tests/` 下的脚本仅用于本地检查，不提交 Git。

## Exporting a Policy

标准导出入口：

```bash
uv run export --task-id black-flat
uv run export --task-id black-rough
```

默认解析流程为 **task stage → latest matching stage run → latest `model_*.pt`
→ `<run>/exported/policy.pt`**，导出默认在 CPU 上执行。

指定来源 checkpoint：

```bash
uv run export \
    --task-id black-rough \
    --load-run '<run>' \
    --checkpoint model_500.pt
```

自定义输出目录：

```bash
uv run export \
    --task-id black-rough \
    --output-dir ~/models/black
```

| Artifact contract | 值 |
|---|---|
| Format | TorchScript，actor only |
| Input | `float32 [1, 45]` |
| Output | `float32 [1, 12]`，raw policy action |
| Filename | `policy.pt` |

Exporter 使用锁定版本的原生 runner 加载和导出，并重新加载产物验证数值等价。
训练 checkpoint `model_*.pt` 与部署 TorchScript `policy.pt` 用途不同，不能直接互换。

## Configuration

[`src/alldog_mjlab/tasks/velocity/black/black_config.py`](src/alldog_mjlab/tasks/velocity/black/black_config.py)
是当前 Black flat/rough 的主要人工训练配置入口，集中管理 environment、control、commands、
observation scales/noise、reset、termination、rewards、domain randomization、terrain、
simulation、policy network、PPO 和 runner 数值。

`env_cfgs.py` / `rl_cfg.py` 将这些数值组装成原生 MjLab / RSL-RL 配置。
Policy joint order、action order、observation layout、manager term order、sensor identity
和 entity selectors 属于 contract / wiring，不应作为普通超参数随意修改。
Observation scales、control dt 和 action scale 也影响部署契约，修改后需要重新验证兼容性。

[`src/alldog_mjlab/robots/black/`](src/alldog_mjlab/robots/black/) 负责 robot asset、default pose、
actuator、joint identity 和 robot intrinsic parameters；任务训练参数归 tasks 管理。

## Tasks

### `black-flat`

Flat-ground Black PPO locomotion task，使用 plane terrain。
Actor 为单帧 **45-D**，privileged critic 为 **259-D**，包含 terrain height scan。

### `black-rough`

在 flat 契约上加入 rough terrain generator、terrain curriculum、terrain-relative
base-height reward 和 training 下的 rough terrain safety truncation。
Terrain 为 7 类 curriculum generator（含 native box 台阶 stairs up/down，
step_height = 0.05 + 0.18 × difficulty；spawn 权重取 super-dog HEAD lineage：
flat 0.10 / slopes 0.05+0.05 / rough 0.10 / obstacles 0.20 / stairs 0.25+0.25）。
Actor contract 与 flat 一致（45-D），critic layout 也与 flat 一致（259-D）。
约 500 iteration 的 PPO baseline 已验证，长期收敛与定量评估仍待完成。

### `black-flat-him`

`black-flat` + HIM observation contract：actor 输入为 6 帧 history `[B, 6, 45]`，
推理另需 source encoder（history → velocity 估计 + latent）；训练额外使用
estimator velocity target 与 terminal successor recorder，算法为 HIMPPO。
支持 `--agent.warm-start True` 从 black-flat PPO checkpoint 初始化。

### `black-rough-him`

`black-rough` + HIM contract（同 flat-him 与 rough 的关系，环境 contract 不因 HIM 改变）。
不支持 PPO→HIM warm start；训练入口是 flat HIM checkpoint 的 full resume：
`--agent.resume True --agent.load-run '.*_flat_him$'`。
长训练收敛未验证。

Task 层组织与进一步说明见 [`src/alldog_mjlab/tasks/README.md`](src/alldog_mjlab/tasks/README.md)。

## Project Structure

```text
alldog_mjlab/
├── src/alldog_mjlab/
│   ├── robots/
│   ├── tasks/
│   ├── algorithms/
│   └── utils/
├── archive/
├── .ai/
├── pyproject.toml
└── uv.lock
```

| 目录 | 职责 |
|---|---|
| `robots/` | Asset、joint、actuator、default pose 与机器人固有参数 |
| `tasks/` | Observation、action、command、reward、termination、event/reset、DR、terrain 与 sensors |
| `algorithms/` | Algorithm-specific learning logic；当前 production Black 使用标准 PPO |
| `utils/` | 策略导出与项目级工具 |
| `archive/` | 历史原型，不参与 production task discovery |
| `.ai/` | 迁移状态、contract 与内部开发记录 |

## Black Policy Contract

Policy joint order 固定为 **FL → FR → RL → RR**，每腿 **hip → thigh → calf**。
Actor 使用单帧 45-D observation，无 history，actor running normalization 关闭；action 为 12-D。
Observation 依次为 command、base angular velocity、projected gravity、相对 default pose 的
joint position、joint velocity 和 previous action，使用任务定义的 scales。

```text
q_policy = q_default + 0.25 * raw_action
physics dt = 0.005 s
decimation = 4
policy dt = 0.02 s
policy frequency = 50 Hz
```

**MuJoCo model natural order != policy order**：当前模型腿顺序为 FL → FR → RR → RL。
Deployment observation/action 必须显式按 policy order 映射，不能直接沿用模型自然顺序。

`alldog_mjlab` 负责 training、observation/action contract、policy export 与 policy semantics。
Deployment framework 从 simulation/hardware state 构造 observation，把 policy action
映射为低层 command，并负责 runtime safety 与 hardware backend。
当前 deployment runtime 是 [coverMoon/quadruped_control](https://github.com/coverMoon/quadruped_control)；
双方通过显式契约对接，本仓库不依赖其内部代码。

## Roadmap

当前 production 已提供 Black flat/rough PPO、Black HIM（flat-him / rough-him）训练链路
（warm start + full resume 已验证）、stage-aware resume、TorchScript actor-only export
和已验证的 deployment policy contract。近期继续 HIM 长训练收敛验证、训练调参与评估；
后续计划包括 HIM exporter、BlackW flat/rough，以及更晚的任务扩展。
HIM TorchScript/ONNX exporter、BlackW 和实机 backend 当前均未在本仓库实现。

## Related Projects

| 项目 | 用途 |
|---|---|
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Framework authority，当前使用 v1.6.0 |
| [coverMoon/super-dog](https://github.com/coverMoon/super-dog) | 历史 Black / BlackW behavior 与工程参考 |
| [InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco) | 后续 HIM integration 的算法权威来源 |
| [coverMoon/rl_sar-for-super-dog](https://github.com/coverMoon/rl_sar-for-super-dog) | Legacy deployment compatibility 参考 |
| [coverMoon/quadruped_control](https://github.com/coverMoon/quadruped_control) | 当前 deployment runtime |
