# AllDog MjLab Migration Status

> 本文件记录当前实际迁移状态、已经冻结的行为 contract、已知差异、风险和下一步任务。
>
> 它不是架构规范，也不是完整开发日志。长期稳定的 source map、职责划分和迁移原则见项目 Instructions。
>
> 每次继续迁移前必须同时检查本文件和当前代码。若二者冲突，以当前代码为准，并更新本文件。

------

## 1. Current Stage

当前阶段：

```text
Black PPO deployment contract / sim2sim compatibility: COMPLETE
Black sim2real / real-backend preflight: COMPLETE（§19.13）
Black real backend v1 motor mapping / calibration contract: COMPLETE（静态；§19.14）
Black real backend v1 actuator / PD contract: COMPLETE（静态；§19.15）
Black real backend v1 IMU / orientation contract: COMPLETE（静态；§19.16）
Black real backend v1 timing / freshness / failure contract: COMPLETE（静态；§19.17）
Black RealRobotIO v1 architecture / implementation plan: COMPLETE（静态；§19.18）
Black training configuration consolidation v2: COMPLETE（behavior-neutral）
Black flat/rough shared critic height scan: COMPLETE（flat critic 72→259；§13.4）
Black actor export short CLI: COMPLETE（§19.3）
Black stage-aware run / resume / export: COMPLETE（§2 / §19.3）
Black HIM observation / history / estimator target contract: COMPLETE（§20）
Black HIM algorithm（HIMPolicy / HIMEstimator / HIMPPO）: COMPLETE（§21）
Black HIM runner integration / checkpoint / resume: COMPLETE（§22）
black-flat-him registration + PPO→HIM warm start: COMPLETE（§23）
black-rough-him registration + flat→rough HIM full resume: COMPLETE（§24）
Black rough stairs terrain（up / down）: COMPLETE（§13.5）
Black performance-based forward-speed command curriculum
+ checkpoint/resume state: COMPLETE（§4.3 / §4.4）
Wolf robot asset / joint order / actuator contract (stage 1): COMPLETE（§27）
Wolf IMU observation contract: COMPLETE（§27.5）
Wolf flat PPO / HIM / command curriculum integration: COMPLETE（§28）
Wolf Domain Randomization（逐项开关 + minimal profile）: COMPLETE（§29）
Wolf task-local independence + Wolf rough PPO/HIM: COMPLETE（§30）
Wolf flat 模板 spawn 高度 + flat 容量优化（64/256）: COMPLETE（§31）
Wolf rough 模板 spawn 高度（z=10）+ rough 容量优化（128/256）: COMPLETE（§32）
Hardware effort/current ceiling: UNCONFIRMED（实机前确认）
```

旧版 Black flat PPO baseline（critic 72）、Black rough PPO baseline（critic 259，
约 500 iteration，见 §17.2）与 rough 的 MJWarp runtime workaround
（`nconmax = 128`，见 §10）均已完成。Black PPO 部署契约、
`rl_sar` 与 `quadruped_control` 的 config / trace / rollout，以及训练侧跨 runtime
observation、actor、pre-safety `q_policy` 比较均已通过（§17.10 / §19.12）。
`quadruped_control` 的硬件位置裁剪属于已确认的 deployment safety layer；最终
`q_command` 有意不同。MJWarp GPU convex CCD 的 upstream 根因尚未修复。
当前 flat 配置已改为 259-D critic，与 rough 同 layout；该配置尚未重新长训。
旧 flat 72-D critic checkpoint 不能直接作为当前 flat/rough 的完整 PPO resume 起点。
新 flat 配置生成的 2-iteration CPU checkpoint 已在 rough runner 严格完整加载
（actor / critic / optimizer，非长期训练效果验证）。

当前尚未进入：

```text
Black real robot backend（尚未实现；quadruped_control 项目接手）
HIM exporter
BlackW migration
```

项目边界与下一候选：

```text
Black sim2real training/deployment contract: COMPLETE
quadruped_control RealRobotIO/runtime implementation: HANDOFF TO quadruped_control PROJECT
alldog_mjlab next candidate: Black command curriculum training validation
                   （§25 item 17 之后的首选；观察 vx_max progression / EMA low+high /
                   tracking ratio）; Black HIM training validation（item 16）仍排队。
```

------

## 2. Current Production Layout

当前 Python package：

```text
alldog_mjlab
```

主要路径：

```text
src/alldog_mjlab/robots/black/
src/alldog_mjlab/tasks/velocity/black/
src/alldog_mjlab/algorithms/
src/alldog_mjlab/utils/export_policy.py    actor-only TorchScript 导出 CLI（§19.3）
```

Black 当前主要 task config：

```text
src/alldog_mjlab/tasks/velocity/black/black_config.py
src/alldog_mjlab/tasks/velocity/black/env_cfgs.py
src/alldog_mjlab/tasks/velocity/black/rewards.py
src/alldog_mjlab/tasks/velocity/black/terminations.py
src/alldog_mjlab/tasks/velocity/black/terrain.py
src/alldog_mjlab/tasks/velocity/black/rl_cfg.py
```

### Configuration layout

```text
black_config.py
    Black flat/rough 单一人工训练参数入口：BLACK_CONFIG，typed sections 覆盖
    env/control/command/command_curriculum/observation/noise/reset/termination/reward/domain_rand/
    terrain/simulation/policy/algorithm/runner。train_num_envs=4096、
    play_num_envs=1；CLI 显式环境数量仍可覆盖 task 默认值。physics dt=0.005、
    decimation=4，policy dt 由二者相乘；control.action_scale=0.25。
    observation scales、control dt 和 action scale 为部署敏感字段。

env_cfgs.py
    MjLab task assembly + policy / task interface contract
    （term 顺序、selector、sensor 身份、与 native 的差异）；从 BLACK_CONFIG
    读取数值并装配 ManagerBasedRlEnvCfg

rewards.py
    Black 专用 reward math

curriculums.py
    Black 专用 stateful curriculum term（ForwardSpeedCommandCurriculum，§4.3）

curriculum_checkpoint.py
    算法无关的 command curriculum checkpoint 序列化 / restore-mode 公共层
    （同时服务 VelocityOnPolicyRunner 与 BlackHimOnPolicyRunner，§4.4）

terminations.py
    Black 专用 stateful termination math
```

`black_config.py` 只是人工数值来源，MjLab `ManagerBasedRlEnvCfg` 仍是唯一 runtime
environment config；`rl_cfg.py` 把 policy/algorithm/runner sections 转成 RSL-RL cfg。
policy joint/action/observation term order 仍由 `robots/black` 和 `env_cfgs.py` 明确绑定。

### Black stage-aware run / resume

`experiment_name = black_velocity` 继续表示共同 checkpoint-compatible policy family。
stage 默认值由 `BLACK_CONFIG.runner.flat` / `.rough` 集中管理：

```text
black-flat   run_name=flat    load_run=.*_flat$
black-rough  run_name=rough   load_run=.*_rough$
```

MjLab v1.6 原生 train 创建 `<timestamp>_<run_name>`，不新增 train/resume 入口。
同 stage 的 `--agent.resume True` 默认从该 stage 最新匹配 run/checkpoint 完整续训；
跨 stage 命令为 `uv run train black-rough --agent.resume True --agent.load-run '.*_flat$'`，
来源为 flat，目标环境与新 run 为 rough。名称/正则解析均使用 native `get_checkpoint_path()`。

Full resume 保持 MjLab / RSL-RL 默认语义：actor、critic、normalizer、optimizer、
checkpoint learning rate、iteration 完整恢复；MjLab 额外仅恢复
`env_state.common_step_counter`。command curriculum runtime state
（`env_state.command_curriculum`）由本仓库的公共层 `curriculum_checkpoint.py`
按 §4.4 restore mode 恢复（同 stage full / 跨 stage range / 旧 checkpoint none+warning；
PPO runner 为 `BlackVelocityOnPolicyRunner(mixin, VelocityOnPolicyRunner)`，
ONNX export 行为不变）。rough 环境重新构造，flat simulator、terrain levels
和其他 runtime state 不复制。native rough curriculum 仅在 counter=0 的首次 reset
跳过升级/降级；wrapper 在 load 前完成首次 reset，load 后 counter 原样恢复，后续
curriculum 仍按 target rough env 的行走距离与 command 计算，不重置 counter。
`max_iterations` 是 additional iterations，不是最终 iteration 上限。

当前两项 task 的 runtime actor 均为 MLPModel 45→512→256→128→12（无 normalization），
critic 均为 MLPModel 259→512→256→128→1（有 normalization），GaussianDistribution
与 PPO 配置一致；仅 runner `run_name/load_run` 不同。CPU config/path/native CLI
命名检查 PASS；新 flat 2-iteration checkpoint → rough 的严格完整加载 PASS。
历史 `2026-09-18_19-37-17/model_498.pt` 经实查 critic 为 259-D，当前严格完整加载
PASS：iter=498、counter=12000、学习率≈1.7086e-4，模型/normalizer/optimizer 逐值恢复。
该历史 run 保持原身份与路径，不强行归类、不参与默认 stage 匹配；显式 load-run
仍可访问。旧 72-D flat checkpoint 不满足当前 full-resume 结构要求，actor-only
export 可独立使用。CUDA 不可用，SKIPPED；本单元未验证跨 stage 长期训练收敛。

本地 migration verification：

```text
tests/check_black_flat.py
tests/check_black_rough.py
```

`tests/` 当前仅作为本地验证工具，不提交 Git。

------

## 3. Frozen Black Robot Contract

### 3.1 Policy / deployment joint order

固定顺序：

```text
FL → FR → RL → RR
```

每腿：

```text
hip → thigh → calf
```

完整顺序：

```text
FL_hip_joint
FL_thigh_joint
FL_calf_joint

FR_hip_joint
FR_thigh_joint
FR_calf_joint

RL_hip_joint
RL_thigh_joint
RL_calf_joint

RR_hip_joint
RR_thigh_joint
RR_calf_joint
```

注意：

```text
policy/deployment order != MuJoCo model natural order
```

当前 MuJoCo model natural order为：

```text
FL → FR → RR → RL
```

任何 policy、observation、deployment mapping 都不得直接依赖 model natural order。

------

### 3.2 Actuator / PD

12 个关节均使用独立 `IdealPdActuatorCfg`。

当前参数：

```text
Kp = 40.0
Kd = 1.2
effort_limit = 20.0
```

每个关节使用独立配置入口，允许后续单独调参。

旧 super-dog 中 `RL_thigh_joint Kd=1.0` 已确认是旧代码错误，不迁移。

------

### 3.3 Action

Policy action：

```text
12-D
FL3 → FR3 → RL3 → RR3
```

由四个独立 `JointPositionActionCfg` term组成：

```text
joint_pos_fl
joint_pos_fr
joint_pos_rl
joint_pos_rr
```

每腿：

```text
hip → thigh → calf
```

Action scale：

```text
0.25 rad / unit action
```

使用：

```text
use_default_offset=True
```

Action contract 不依赖 actuator/model natural order。

------

## 4. Frozen Command Contract

Black 的 command 是「**初始范围 + MjLab native sampler + performance-based
forward-speed 课程**」：

```text
generator:
    MjLab v1.6 UniformVelocityCommand
    （body-frame 速度指令，按 resampling 重采样）

initial range（课程起点 = BLACK_CONFIG.command.lin_vel_x）:
    vx   [-1, 1] m/s
    vy   [-1, 1] m/s          （训练全程固定，不参与课程）
    wz   [-π, π] rad/s        （训练全程固定，不参与课程）

resampling:
    10 s（固定，不随机）

heading command:
    disabled（rel_heading_envs = 0，ranges.heading = None）

native sampler 比例:
    standing     10%   （指令强制为 [0, 0, 0]）
    forward-only 20%   （vx ≥ 0.3 且 vy = wz = 0；standing 优先）
    world-frame   0%
    reset 初速度  0%

curriculum（train）:
    command   = ForwardSpeedCommandCurriculum（performance-based，见 §4.3）
    terrain   = 仅 black-rough / black-rough-him training（§13.3）
    play      = 无任何 curriculum（command range 固定在 initial [-1, 1]）
```

`BLACK_CONFIG.command.lin_vel_x` 现在是**课程起点**而不是终值：训练中由 command
curriculum 按策略 tracking 表现逐步扩展到 ±2 m/s（§4.3）；`lin_vel_y` / `ang_vel_z`
与 native sampler 配置训练全程不变。

Deployment contract note（本轮不修改部署侧）：observation layout / command 观测缩放
（×2.0 / ×2.0 / ×0.25）/ action / joint order / control dt 全部不变；唯一的行为变化是
**训练支持的 vx 域从初始 ±1 m/s 变得 curriculum-dependent，训练收敛后最大支持 ±2 m/s**。
部署 runtime 无需任何修改；如未来需要在部署侧显式检查 vx 上限，应读取检查点内
课程 state，而不是把 ±2 写死为机器人固有参数。

### 4.3 Performance-based forward-speed command curriculum（frozen）

语义来源：super-dog Black 后期 ``update_command_curriculum()``（已重读源码移植）；
official HIMLoco 只作概念 authority（其 env-index bucket sampler 不迁移）。
实现为 task 层 stateful class-based CurriculumTerm：

```text
文件          src/alldog_mjlab/tasks/velocity/black/curriculums.py
              class ForwardSpeedCommandCurriculum(ManagerTermBase)
注册名        cfg.curriculum["command"] → 日志前缀 Curriculum/command/*
参数唯一来源  BLACK_CONFIG.command.command_curriculum（CommandCurriculumParams）
```

冻结参数：

```text
enabled                = True（play 不注册；enabled=False 时不注册自然回退旧行为）
initial vx             = [-1.0, 1.0]
max abs vx             = 2.0（clip：vx_min >= -2.0、vx_max <= +2.0，永不越界）
step                   = 0.1（每次推进双边同时扩：vx_min -= 0.1、vx_max += 0.1）
tracking threshold     = 0.70（low-speed 组 EMA 阈值）
high threshold         = 0.60（= 0.70 - 0.10 offset）
EMA alpha              = 0.20（ema = 0.8 × old + 0.2 × group mean）
required passes        = 2（连续 2 次成功 evaluation 才扩一次 range）
buffer minimum         = 256（episode sample 数，不足不评估）
low-speed min samples  = 8
high-speed min samples = 4
low-speed lower bound  = 0.2（|vx| <= 0.2 的样本不进组，只占 buffer）
low/high split ratio   = 0.6（分界 = 0.6 × V，V = max(|vx_min|, |vx_max|)）
```

performance metric（super-dog Black 后期语义）：

```text
tracking_ratio_i = episode_tracking_reward_sum_i
                 / (episode_length_steps_i × env.step_dt × |tracking_reward_weight|)

- raw = exp(-||v_cmd_xy - v_xy||² / sigma)，理论 max 1.0；
- Episode sum 是 MjLab v1.6.0 RewardManager 保存的 raw × weight × step_dt 累计；
- weight 用 reward_manager.get_term_cfg("track_linear_velocity").weight 实时读取，
  不硬编码；
- episode sum 的读取使用 reward_manager._episode_sums —— MjLab v1.6.0 pinned
  private API（v1.6.0 无公开 episode-sum accessor，升级需核对）；
- 样本 = reset env 对应刚结束 episode 的 (|vx command|, tracking_ratio)；
  过滤 non-finite；standing/极小 vx 可进 buffer 但不进 low/high 组。
```

evaluation / 状态机（super-dog 语义）：

```text
1. buffer < 256                     → 不评估（EMA/streak/范围不动）
2. buffer >= 256 但 low/high 任一
   组样本数不足（< 8 / < 4）        → 不更新 EMA、pass_streak = 0、
                                      **buffer 保留**（不清空，继续等待样本）
3. 组数足够                          → 更新 EMA → pass rule：
                                      ema_low > 0.70 且 ema_high > 0.60
                                      → streak + 1；否则 streak = 0
4. streak >= 2                       → range 双边扩 0.1（clip ±2.0），
                                      streak = 0，progressed = 1（范围实际变化时）
5. 完成一次真正的 low/high evaluation → 清空本轮 buffer
6. 达到 [-2, 2] 后                   → 继续统计 telemetry，但不再修改 range
```

runtime state（authoritative，属于 curriculum term 实例）：

```text
vx_min / vx_max      初始 = config 初始范围
ema_low / ema_high   初始 = 0（super-dog 语义，不为加速课程改初始化）
pass_streak          初始 = 0
buffer_cmd_x / buffer_tracking_ratio   CPU float（CUDA 训练无 device mismatch）

telemetry cache（非 authoritative）：last low/high count、last low/high ratio、
evaluated / progressed 标志。TensorBoard 无 NaN：未评估时值为明确初始化值。
```

日志（CurriculumManager 自动记录，不修改 runner logging loop）：

```text
Curriculum/command/vx_min, vx_max, ema_low, ema_high, pass_streak,
buffer_count, low_count, high_count, low_ratio, high_ratio,
evaluated, progressed
```

首次 reset（``common_step_counter == 0``）：不采样、不更新 EMA、不增 streak、
不扩范围；initial range 保持 [-1, 1]。

command range 写回：直接改运行中的 ``UniformVelocityCommandCfg.ranges.lin_vel_x``
（下一次 resampling 生效；已采样的 command 不变）。``lin_vel_y`` / ``ang_vel_z``、
standing / forward-only / world-frame 比例、resampling 间隔、heading 配置
**永不**被 curriculum term 修改。

### 4.4 Checkpoint persistence 与 restore mode（frozen）

curriculum state 存入现有 ``.pt`` checkpoint（不写独立 JSON，模型与课程状态永不错位）：

```text
infos.env_state.common_step_counter          （MjLab 原生，语义不变）
infos.env_state.command_curriculum           （本单元新增）
    = {version, stage(flat|rough), vx_min, vx_max, ema_low, ema_high,
       pass_streak, buffer_cmd_x, buffer_tracking_ratio}
```

序列化 / 恢复公共层（同时服务 PPO 与 HIM，算法无关）：

```text
src/alldog_mjlab/tasks/velocity/black/curriculum_checkpoint.py
    CommandCurriculumCheckpointMixin（save 持久化 + load 恢复）
    BlackVelocityOnPolicyRunner(mixin, VelocityOnPolicyRunner)   ← black-flat / black-rough
    BlackHimOnPolicyRunner(Mixin, MjlabOnPolicyRunner)           ← black-flat-him / rough-him
    stage_from_run_name()     冻结 run-name contract -> flat | rough
                              （flat_him/rough_him 只差算法后缀；集中一个 helper）
    save 主干与 MjLab v1.6.0 MjlabOnPolicyRunner.save 逐行一致（pinned 复制；其父类
    会整体重建 infos["env_state"]，不复制无法扩展该字段）
```

restore mode（``command_curriculum_restore`` config；默认值由
``BLACK_CONFIG.runner.command_curriculum_restore`` 统一配置，PPO / HIM 共用，
CLI ``--agent.command-curriculum-restore`` 仍可临时覆盖；PPO→HIM warm start
不走该字段，固定 range）：

```text
none  —— 不恢复：初始 range [-1,1]、EMA / streak / buffer fresh
range —— 只恢复 vx range，统计 fresh（跨 stage / 跨训练条件）
full  —— 逐值恢复 range + EMA + streak + buffer（同一训练条件精确续训）

auto（默认）决策表：
    new training（无 load）            fresh（无需恢复）
    same-stage resume                  full（flat→flat / rough→rough / flat_him→flat_him / rough_him→rough_him）
    cross-stage resume                 range（flat→rough、flat-him→rough-him）
    stage 未知（手工 state）            range（保守：范围保留，统计重新开始）
    旧 checkpoint（无 curriculum state）none + 一次明确 warning（config 初始范围起步）

    PPO→HIM warm start（不走 resume load）：
        source 有 curriculum state → range（vx range 拷贝，EMA/streak/buffer fresh）
        source 无 curriculum state → warning + config 初始 [-1,1]，warm start 不失败
```

stage 判定不使用 tensor shape / 算法类型 / 模糊 regex：save 时写 ``stage``
（env cfg builder 显式给出 flat | rough），load 时与目标 stage 比较；仅 run-name
到 terrain stage 的映射使用项目冻结的 run-name contract（集中在 stage_from_run_name）。

### 4.5 What remains intentionally not migrated

```text
official HIMLoco:
    env-index high/low speed command bucket sampler（与 curriculum 耦合的那部分）

super-dog:
    terrain_probe / stand_probe / stop_probe
```

------
## 5. Frozen Actor Observation Contract

当前 PPO actor使用单帧：

```text
45-D
```

layout：

```text
[0:3]    command
[3:6]    base_ang_vel
[6:9]    projected_gravity
[9:21]   joint_pos_rel
[21:33]  joint_vel_rel
[33:45]  last_action
```

明确不包含：

```text
base_lin_vel
height_scan
history
HIM latent
privileged observations
```

Joint observation顺序严格使用：

```text
FL → FR → RL → RR
hip → thigh → calf
```

------

### 5.1 Actor observation scale

```text
command:
    vx × 2.0
    vy × 2.0
    wz × 0.25

base_ang_vel:
    × 0.25

projected_gravity:
    × 1.0

joint_pos:
    × 1.0

joint_vel:
    × 0.05

last_action:
    × 1.0
```

------

### 5.2 Actor observation noise

MjLab v1.6.0 observation pipeline：

```text
compute
→ noise
→ clip
→ scale
→ delay
→ history
```

当前使用 raw-space noise：

```text
command:
    none

base_ang_vel:
    Uniform[-0.3, 0.3]

projected_gravity:
    Uniform[-0.05, 0.05]

joint_pos:
    Uniform[-0.08, 0.08]

joint_vel:
    Uniform[-2.0, 2.0]

last_action:
    none
```

对应进入 policy 后的 noise amplitude：

```text
base_ang_vel ±0.075
projected_gravity ±0.05
joint_pos ±0.08
joint_vel ±0.1
```

------

### 5.3 Observation normalization

当前：

```text
actor.obs_normalization = False
critic.obs_normalization = True
```

Actor关闭 running normalization，以保持固定 policy input / deployment preprocessing contract。

Critic observation仍然沿用当前 MjLab privileged observation设计，尚未迁移旧 Black/HIM critic contract。

------

## 6. Frozen Termination Contract

当前 training termination（play 关闭全部碰撞类终止）：

```text
black-flat   train : time_out + illegal_contact + stuck
black-flat   play  : time_out + stuck（illegal_contact 已移除；time_out 因 play
                      episode 极长实际不触发）
black-rough  train : time_out + illegal_contact + stuck + out_of_terrain_bounds
black-rough  play  : time_out + stuck（illegal_contact / out_of_terrain_bounds
                      均已移除）
```

### 6.1 Timeout

```text
episode_length = 20.0 s
```

`time_out` 保持 MjLab truncation语义。

------

### 6.2 Illegal contact

当以下任一 body 与 terrain发生：

```text
contact force magnitude > 1.0 N
```

时 termination。触发 body 集合由
`BLACK_CONFIG.termination.illegal_contact_bodies` 配置（legged_gym 式
``termination_contact_names`` 等价项），冻结默认为：

```text
trunk

FL_thigh
FR_thigh
RL_thigh
RR_thigh
```

不包含：

```text
hip
calf
foot
```

使用 MjLab native：

```text
ContactSensor
+
mdp.illegal_contact
```

Contact history：

```text
history_length = 4
```

对应：

```text
physics dt 0.005 s
×
decimation 4
=
control dt 0.02 s
```

当前已删除：

```text
fell_over / bad_orientation 70°
```

flat task 不使用 orientation-based fall termination。
`out_of_terrain_bounds` 在 common contract 里移除，由 §6.4 仅对 black-rough training
重新追加。

### 6.3 Stuck

```text
command xy norm > 0.2
progress_speed < 0.05
grace > 1.0 s
continuous timer > 4.0 s
```

`progress_speed` 为 body-frame root 线速度在 commanded planar direction 上的投影；
progress 恢复或 move command 失效时计时立即归零（连续计时）。

实现：

```text
src/alldog_mjlab/tasks/velocity/black/terminations.py
class StuckTermination(ManagerTermBase)
```

per-env timer（秒）由 class term 自己持有，经 `TerminationManager.reset()` 清零；
不使用 event，不写入 env / runner。

### 6.4 out_of_terrain_bounds（仅 black-rough training）

使用 MjLab v1.6.0 native ``mjlab.tasks.velocity.mdp.out_of_terrain_bounds``，不写 custom
OOB 实现（native 已正确处理 effective grid shape / curriculum 模式 / border_width）。

```text
cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=mdp.out_of_terrain_bounds,      # params 为空：用 native default margin
    time_out=True,
)
```

边界公式（native）：

```text
num_rows, num_cols = terrain.terrain_origins.shape[:2]      # effective grid（非 cfg 字段）
half_x = 0.5 x (num_rows x size[0]) + border_width
half_y = 0.5 x (num_cols x size[1]) + border_width
limit  = max(0, half - margin)
out    = |root_x_w| > limit_x  OR  |root_y_w| > limit_y      # 严格 >，非 >=
```

当前 Black rough 数值（由 runtime terrain config 推导，非硬编码；§13.5 加入
stairs up/down 后 terrain 类型数 5 → 7）：

```text
effective grid   10 x 7（curriculum 模式一个 terrain 一列）
patch            8 x 8 m
border_width     20 m
margin           0.3 m（native default，未显式传入）
half_x / half_y  60.0 / 48.0 m
limit_x / limit_y  59.7 / 47.7 m
train  注册（末尾追加）
play   移除
flat   不注册（也不依赖 always-False）
```

`time_out=True` 表示这是 **truncation** 而非 terminal failure：OOB 是有限生成 map 造成的
artificial truncation，不是机器人自身的 physical failure，因此 MjLab / RSL-RL 会把它作为
truncated 传给 PPO 并正确 bootstrap value（测试已断言
``termination_manager.time_outs`` 为 True 而 ``terminated`` 为 False）。

Intentional framework difference：legacy super-dog ``check_termination()`` 只有 contact
failure 与 episode time out，没有 global terrain OOB；MjLab Black rough v1 把它作为有限
生成 terrain 的 safety truncation，**不是** legacy-equivalent。

------

## 7. Frozen Root Reset Contract

每次 reset：

### Root pose

```text
position = default position + env origin

default root position:
(0.0, 0.0, 0.45)

orientation = default orientation
```

不随机：

```text
x
y
z
roll
pitch
yaw
```

即：

```text
pose_range = {}
```

### Root velocity

6-D independent uniform：

```text
linear x ∈ [-0.5, 0.5] m/s
linear y ∈ [-0.5, 0.5] m/s
linear z ∈ [-0.5, 0.5] m/s

angular x ∈ [-0.5, 0.5] rad/s
angular y ∈ [-0.5, 0.5] rad/s
angular z ∈ [-0.5, 0.5] rad/s
```

使用 MjLab native：

```text
reset_root_state_uniform
```

------

## 8. Frozen Joint Reset Contract

使用三个互不重叠的 native：

```text
reset_joints_by_offset
```

events：

```text
reset_hip_joints
reset_thigh_joints
reset_calf_joints
```

### Hip

```text
position offset = [0.0, 0.0]
joint velocity = 0
```

因此 hip保持 default pose。

### Thigh

```text
position offset ∈ [-0.4007, +0.4007] rad
joint velocity = 0
```

完整保留旧 Black `default × U[0.5, 1.5]` 的 thigh support。

### Calf

```text
position offset ∈ [-0.5945, +0.5945] rad
joint velocity = 0
```

这是一个有意的 framework adaptation。

旧 Black等效 calf offset：

```text
±0.7635 rad
```

其部分 support超出当前 MjLab soft joint limits。

没有采用：

```text
旧范围 + clamp
```

因为会在 soft-limit boundary产生概率质量堆积。

当前设计保持：

```text
default pose = distribution mean
左右腿对称
continuous uniform distribution
全部 support 位于 soft limits 内
```

------

## 9. Current PPO Configuration

当前 PPO仍使用 MjLab / RSL-RL runner。

Actor / critic network：

```text
512
256
128
```

activation：

```text
ELU
```

主要 PPO参数当前未作为 Black legacy behavior migration重点修改。

当前 HIM算法目录存在，但不是当前 migration stage。

------

## 10. Current Intentional Framework Differences

以下差异当前明确保留，不要在无关任务中顺手修改。

### Encoder bias

encoder bias domain randomization 已冻结为 DR contract 的一项（见 §12）。

旧 Black没有完全对应的同类机制，因此这是 framework-native 行为，不是 exact migration。

------

### Calf initial reset distribution

旧：

```text
default ±0.7635
```

当前：

```text
default ±0.5945
```

原因见 joint reset contract。

这是有意改变，不是遗漏。

------

### MjLab termination timing

MjLab termination manager读取部分 derived physics quantities时，可能相对最终 physics state落后一个 physics substep。

当前接受 MjLab framework-native timing。

不要为了旧 Isaac Gym timing在 termination内部额外：

```text
sim.forward()
sim.sense()
```

### MJWarp contact capacity（nconmax）

```text
MjLab v1.6 velocity baseline:  cfg.sim.nconmax = 35
Black flat（train / play）:     35（保持 baseline）
Black rough（train / play）:    128
```

原因：Black rough 的多接触状态已在 MJWarp GPU convex narrowphase CCD 中触发
capacity-related CUDA runtime fault。固定 bad simulator state 下，`nconmax = 35` 可确定性
复现（4/4），`>= 48` 不再触发；长 rollout 在 35 下 6 次里 5 次崩溃（崩在哪步在 ~2600–
3500 间浮动），128 / 256 均稳定。项目选择 128 作为保守余量。

- 这是 **project-side capacity workaround**，不是 legacy Black behavior contract；
- **不改变** reward / observation / action / terrain / policy contract：除该 field 外，
  rough 的 config 与改动前逐字段一致，flat（train / play）完全不变；
- **不是** upstream MJWarp 根因修复，也不声称 128 是理论最小正确值；
- 证据（固定 state 复现、sanitizer、长 rollout）见 §17.2；
- 不要把该值传播到 flat。

### MJWarp constraint capacity（njmax）

```text
MjLab v1.6 velocity baseline:  cfg.sim.njmax = 1500
Black 统一候选（flat / rough / PPO / HIM 共用）: 256
```

依据：black-rough-him 实测（check_black_sim_capacity，3000 env / 800 physics
substeps / 2.4M world-substep samples）：max nefc = 80、p99.9 = 52，overflow = NO。

结构性下限（MuJoCo-Warp 3.11 put_data 硬性要求 njmax >= 模板 spawn state 的
mjd.nefc）：flat seed = 144、rough seed = 220（约 12 friction_dof + 4 joint limit +
spawn pose contact 的 pyramidal rows；Black nv=18<=32 → dense，tile 16 对齐）。
原候选 128 < 144，无法构建任何 Black env，已否决。256 = seed floor ~1.16x、
实测 runtime max ~3.2x。

状态：**capacity candidate implemented；long stress / VRAM / training validation
pending（用户手动验证；验证口径 = overflow bit 观察为 NO + 训练不收敛崩溃）**。
不要在验证完成前把 256 写成最终冻结值。

```
check_black_sim_capacity.py（tests/，不提交）用于测量；本轮同时修复该工具的
CONTACTS / WORLD mean 统计 bug：histogram.sum() 是 (world, substep) 样本数而非
contacts 数，mean 改用直接求和累计器；修复前出现 mean=1.000 / p50=2 的矛盾输出。
```

------

## 11. Frozen Reward Contract

### 11.0 Reward authority

```text
core reward structure / math:
    InternRobotics/HIMLoco official Go1 baseline
    （legged_gym/legged_gym/envs/go1/go1_config.py + envs/base/legged_robot.py）

Black-specific numeric / robot semantics:
    Black legacy（机器人固有量，如 base height target）

super-dog 2026-07-03 lineage (68f1c1c):
    optional shaping / historical tuning reference / sim2real 诊断参考，
    不再是「所有 Black reward 必须迁移」的 authority
```

super-dog 中未被本 baseline 采用的自定义 shaping 仍是有效的历史知识，可在需要时
按训练问题重新引入，但不属于 v1。见 §11.3。

### 11.1 Final reward table（10 项）

`term / func / weight / params` 的完整 contract，dict 顺序即 logging 与调参表顺序：

```text
term                    func                          weight
track_linear_velocity   track_linear_velocity_xy      +1.0
track_angular_velocity  track_angular_velocity_z      +0.5
lin_vel_z               vertical_linear_velocity_l2   -2.0
body_ang_vel            angular_velocity_xy_l2        -0.05
upright                 flat_orientation_l2 (native)  -0.2
base_height             base_height_l2_flat           -1.0
dof_acc                 dof_acc_l2                    -2.5e-7
joint_power             joint_power_l1                -2e-5
action_rate_l2          action_rate_l2 (native)       -0.01
smoothness              action_acc_l2 (native)        -0.01
```

```text
tracking sigma      = 0.25
base height target  = 0.43
```

没有 zero-weight placeholder；reward 函数只返回 raw magnitude，`dt` 缩放由
`RewardManager`（`scale_rewards_by_dt=True`）统一处理。

### 11.2 Per-term semantics

```text
track_linear_velocity
    raw = exp(-Σ(v_cmd_xy - v_xy)² / sigma)        body-frame root lin vel
track_angular_velocity
    raw = exp(-(w_cmd_z - w_z)² / sigma)           body-frame root ang vel
lin_vel_z
    raw = v_z²                                      body-frame root
body_ang_vel
    raw = ω_x² + ω_y²                               body-frame root
upright
    raw = g_x² + g_y²                               body-frame projected gravity
base_height
    raw = (base_height - 0.43)²
    flat : base_height = root_link_pos_w.z               （world-z 语义）
    rough: base_height = mean(terrain_scan 中央 35 ray)  （local terrain-relative，见 §11.4）
dof_acc
    raw = Σ((q̇_prev - q̇) / step_dt)²               control step 有限差分
joint_power
    raw = Σ|q̇| · |τ|                               τ 取 qfrc_actuator（关节空间）
action_rate_l2
    raw = Σ(a_t - a_{t-1})²
smoothness
    raw = Σ(a_t - 2a_{t-1} + a_{t-2})²
```

实现归属：

```text
native（与 HIMLoco 严格等价）:
    flat_orientation_l2 / action_rate_l2 / action_acc_l2
task-local（native 不等价或缺失）:
    track_linear_velocity_xy / track_angular_velocity_z /
    vertical_linear_velocity_l2 / angular_velocity_xy_l2 /
    base_height_l2_flat / base_height_l2_terrain（rough） /
    joint_power_l1 / dof_acc_l2（stateful class term）
```

不再使用的实现：`orientation_l1`（Black 后期 shaping，非 HIMLoco 公式）、
`base_height_l1_flat`（HIMLoco 为平方惩罚）。

`upright` 与 native exp 版 `mdp.upright` 不是同一 contract：后者是
`exp(-Σg_xy²/std²)` 的正奖励（upright 时 raw = 1），当前 contract 的 L2 惩罚在
upright 时 raw = 0。

`dof_acc` 使用 MuJoCo 瞬时 `qacc` 的 native `joint_acc_l2` 与 HIMLoco 的
control-step 有限差分不等价，因此用 stateful class term 保存上一 step 的 q̇；
reset 时把缓存置为当前 q̇（首个 step 差分为 0）。

### 11.3 Not part of Black flat v1 baseline

以下既不在表中，也不再用 zero-weight 占位（避免「看起来还在但不确定该不该用」）：

```text
MjLab-only shaping（原 MjLab velocity baseline）:
    pose / angular_momentum / dof_pos_limits / air_time /
    foot_clearance / foot_swing_height / foot_slip / soft_landing

super-dog 自定义 shaping（未采用，来源仍可在 super-dog 查看）:
    hip_pos / feet_spacing / raibert / stand rewards（stand_still /
    stand_torque_balance / stand_feet_force_balance）/ collision /
    foot_impact_vel / gait phase shaping（trot）/ all_joint_pos /
    progress / feet_air_time / feet_stumble / foot_slip /
    termination reward / torques / dof_vel / dof_pos_limits /
    dof_vel_limits / torque_limits / terrain-adaptive reward shaping
```

foot_clearance 特别说明：HIMLoco 官方 Go1 有 `foot_clearance = -0.01`，但三家公式
互不相同（HIMLoco：body-frame foot z 误差² × body-frame 横向速度；MjLab native：
terrain-relative 高度误差 × world-frame 横向速度；Black 后期：phase / terrain
adaptive），因此本轮有意不采用任何一种。这是 baseline simplification，不是遗漏。

采用这些项需要新的 behavior unit 与明确动机。

### 11.4 Black rough base-height reward

Black rough 的 ``base_height`` 与 flat 用**同一个 reward key / weight / L2 kernel**，
只把 base height 的**测量方式**从 world z 换成 local terrain-relative clearance：

```text
                       func                          base_height
flat   base_height_l2_flat      root_link_pos_w.z（world-z）
rough  base_height_l2_terrain   mean(terrain_scan 中央 footprint 的 raw 高度)
```

``raw`` 取自 native ``mjlab.envs.mdp.height_scan()``（**不是** critic ObservationManager
里已 scale = 0.2 的那份），即每条 ray 的 ``trunk_z - terrain_hit_z``（offset 0）。
footprint 为 ``terrain_scan`` 的中央区域：

```text
x ∈ [-0.3, +0.3]   7 values
y ∈ [-0.2, +0.2]   5 values
35 rays（从 187 条 native ray 中按 mask 选出；索引由 pattern 实际 offsets 推导，
不手写 magic indices）
```

reward：

```text
raw = (mean(raw_footprint) - 0.43)²        weight = -1.0
```

无 clip / scale / ×5 / height noise。target 0.43 与 reset root z 0.45 的差异与 flat 相同
（见 §7）。其余 9 项 reward 的 func / weight / params 完全不变，``upright`` 仍是 native
``flat_orientation_l2``（**不是** terrain-normal 语义）。

legacy 对照（**intentional difference，不是 exact reproduction**）：

```text
legacy BlackEnv（black_env.py ``_init_base_height_points`` + base ``_get_base_heights``）
    footprint 11 x 9 = 99 点：x ∈ [-0.30, 0.30] step 0.06、y ∈ [-0.18, 0.18] step 0.045
    （0.60 x 0.36 m），随 robot yaw 旋转
    高度取 heightfield 3-cell min：min(h[px,py], h[px+1,py], h[px,py+1])
    base_height = mean(root_z - terrain_height)
    reward = |base_height - target|              ← L1 kernel

Black rough v1
    35 条 native ray 的直接命中值（0.1 m 网格，与 legacy 的 0.06 / 0.045 不同）
    kernel 保持 L2（与 flat 一致，不恢复 legacy L1）
    不含 legacy 的 3-cell min 采样
```

关键 invariance（已由 ``tests/check_black_rough.py`` 验证）：terrain 与 root 同时升高
``+0.10 m`` 时 local clearance 不变 → reward 不变（``flat`` 的 world-z 语义在同一情形下
从 ``1.0e-4`` 增到 ``1.21e-2``）。因此 rough 在坡 / 坑上的 ``base_height`` 惩罚不再因
world z 偏移而系统性偏大（实测：同批 env 的 world-z 惩罚均值在 slope 列达 ``0.19``，
terrain-relative 仅 ``4e-4``）。

------

## 12. Frozen Domain Randomization Contract

Black flat v1 的 DR 只包含下面 6 项。其余候选项明确 deferred（见本节末）。

```text
term            func (mjlab v1.6)        mode      operation  数值
foot_friction   dr.geom_friction        startup   abs        0.2 ~ 1.25（同一 env 四足共享）
payload_mass    dr.body_mass            startup   add        trunk -1 ~ +2 kg
base_com        dr.body_com_offset      startup   add        trunk xyz ±0.05 m
pd_gains        dr.pd_gains             reset     scale      Kp / Kd 0.9 ~ 1.1
encoder_bias    dr.encoder_bias         startup   —          ±0.015 rad
push_robot      mdp.push_by_setting_velocity  interval  —    每 16 s，root xy Δv ±1 m/s
```

细节：

```text
foot_friction
    selector = 四个 *_foot_collision geom
    shared_random = True（同一 env 内四只脚同一采样，不同 env 独立）
    只改切向摩擦（mjlab 默认 axis 0），不改 terrain friction

payload_mass
    selector = trunk（Black 主要机身质量所在，nominal 5.7042 kg）
    只加质量不改惯量（legacy payload 语义一致；mjlab 会就此给 UserWarning）

base_com
    相对 nominal COM 的 offset（不是 absolute COM）

pd_gains
    覆盖全部 12 个 IdealPd actuator；nominal Kp 40 / Kd 1.2 → 运行时 Kp 36~44、Kd 1.08~1.32

encoder_bias
    固定 encoder calibration bias，只影响带 bias 的 joint position observation，
    不改变物理 qpos；episode reset 不重采样

push_robot
    只扰动 root xy；z / roll / pitch / yaw 的 Δ 严格为 0
```

### 12.1 Intentional framework differences

```text
PD gains
    legacy HIMLoco：一个 env 共享一个 Kp factor / 一个 Kd factor
    MjLab native ：每个 actuator target 独立采样
    本轮接受该差异，不为 shared scalar 写 custom DR

push
    legacy HIMLoco：直接设置随机 root xy velocity
    MjLab native ：向当前 root velocity 增加随机 increment
    本轮接受 native 语义，不写 custom push
```

### 12.2 play 模式

`play=True` 为 nominal physics：上面 6 个 DR event 整组移除（push 也在其中），
只保留 reset events（`reset_base` / `reset_hip_joints` / `reset_thigh_joints` /
`reset_calf_joints`），并继续关闭 actor corruption。

train / play 的区别仅剩：DR、actor observation corruption、episode 长度、
碰撞类终止（play 无 illegal_contact）；command
contract 两侧完全相同（见 §4）。

### 12.3 Deferred DR（不在 v1）

```text
link mass                （官方 HIMLoco randomize_link_mass = False）
inertia / pseudo inertia
motor strength           （effort_limits 改 saturation boundary，与 legacy
                          τ_out = factor × τ_computed 在未饱和时行为不同）
external force disturbance（官方有 ±30 N / 8 s，本 v1 先用 push）
action delay             （legacy 为 control-step action queue，1 lag = 20 ms；
                          MjLab 为 actuator physics-step delay，1 lag = 5 ms，
                          与 sampling cadence / deployment 对应关系需一并决定）
restitution              （保持无 restitution DR）
```

initial joint-state variation 不再叠加官方 HIMLoco 的 `initial_joint_pos_range`：
已由 reset contract（`BLACK_CONFIG.reset.joint_position`）覆盖。

motor strength / action delay 分别在 sim2real contract 阶段处理。

------

## 13. Frozen Black Rough Terrain Contract

Black rough v1 的 terrain 由 MjLab v1.6.0 native terrain generator 生成
（`TerrainEntity` + `TerrainGeneratorCfg` curriculum 模式），只新增一个 task-local
sub-terrain primitive（rough slope），不引入第二套 terrain framework。

数值在 `BLACK_CONFIG.terrain.*`，terrain 数学在 `tasks/velocity/black/terrain.py`，装配在
`env_cfgs.black_rough_env_cfg()`（= `black_flat_env_cfg()` + rough terrain 覆盖，
不复制 flat 配置）。

### 13.1 Legacy 来源与映射

legacy Black（super-dog）terrain 来自 `legged_gym/utils/terrain.py`
（`Terrain.curiculum()` + `make_terrain()`）叠加 Isaac Gym `terrain_utils`
（`HIMLoco/isaacgym/python/isaacgym/terrain_utils.py`）；数值取 `68f1c1c`
（2026-07-03 lineage）的 `black_config.py`。HEAD 的 stair-mixed proportions
（`[0.1, 0.1, 0.1, 0.25, 0.25, 0.2, ...]`）不采用，与 §11.0 的 reward authority
是同一条 lineage 决策。

```text
legacy（68f1c1c）                      Black rough v1（MjLab v1.6.0）
terrain_length / width = 8.0 / 8.0     generator.size = (8.0, 8.0)
horizontal_scale = 0.1                 sub-terrain horizontal_scale = 0.1
vertical_scale = 0.005                 sub-terrain vertical_scale = 0.005
num_rows = 10                          generator.num_rows = 10
num_cols = 20                          curriculum 模式忽略；列数 = terrain 类型数 = 5
difficulty = row / num_rows            difficulty_range = (0.0, 0.9)（按 row/(num_rows-1) 插值）
curriculum = True                      generator.curriculum = True + cfg.curriculum["terrain_levels"]
max_init_terrain_level = 5             TerrainEntityCfg.max_init_terrain_level = 5
border_size = 25                       generator.border_width = 20.0（native preset 值）
pyramid platform_size = 3.0            platform_width = 3.0
```

terrain 类型与 spawn 权重（legacy `terrain_proportions`）冻结为：

```text
flat                 0.20   BoxFlatTerrainCfg
smooth_slope_up      0.15   HfPyramidSlopedTerrainCfg(inverted=False)
smooth_slope_down    0.15   HfPyramidSlopedTerrainCfg(inverted=True)
rough_slope          0.30   BlackRoughSlopeTerrainCfg（task-local）
discrete_obstacles   0.20   HfDiscreteObstaclesTerrainCfg(mode="choice")
```

难度语义（d = row difficulty ∈ {0.0, 0.1, ..., 0.9}，不含 1.0）：

```text
slope            = 0.7 * d        （smooth 与 rough 相同；max 0.63）
rough noise      = ±(0.015 + 0.1 d)，step 0.005，downsample 0.2（双线性）
obstacle height  = 0.06 + 0.2 d   （choice 模式：±h 与 ±h/2 混合坑与凸起）
stair step height = 0.05 + 0.18 d （up / down 同式，见 §13.5）
```

未迁移（本轮明确不含）：stairs / wave / stepping stones / gap / bridge / wall。

### 13.2 Intentional framework differences

```text
1. 列语义：legacy 用列编码 proportion（20 列）；MjLab curriculum 模式一个 terrain
   一列（5 列），proportion 变成 env spawn 权重，因此实际网格是 10 x 5。
2. border：legacy 的 border 是 heightfield 的一部分（25 m 平面）；MjLab 的
   border_width 是 z = 0 的 flat apron + 1 m 裙边，取 native rough preset 的 20 m。
3. rough slope 组装：MjLab 没有 slope + noise 的 native composition，因此
   BlackRoughSlopeTerrainCfg 组合 native 的 slope 数学与 native 的 uniform noise
   数学（两个 int16 高度场相加），复用 native 的 hfield / color_by_height 构造。
4. uniform noise 插值：MjLab native preset 用 RectBivariateSpline 默认三次样条，
   会在采样点之间 overshoot 超出 ±amplitude；legacy 用 interp2d(kind='linear')。
   Black rough v1 取 kx = ky = 1（双线性）= legacy 语义，使 ±amplitude 契约可验证。
5. plateau 截断：native 与 legacy 一致地在 platform 角点高度截断（plateau 实际略大于
   platform_width），配置的 slope 只体现在 plateau 之外的 ramp 段。
6. env 分配：env 按 proportion 分配到 (row, col)，多个 env 可共享同一 patch
   （native 文档语义；legacy 同样共享）。
7. play：MjLab native rough task 在 play 下把 generator 切成 random 模式 + 5 x 5 小网格；
   Black rough v1 的 play 保留与训练相同的 generator 布局与 spawn 比例（便于按训练
   分布评估），只把 curriculum term 置空。
8. out_of_terrain_bounds 不在 common termination 中（见 §6.4），仅 black-rough
   training 末尾追加；terrain_scan 已在 13.4 接入 rough critic（actor 仍 45 维）。
9. rough slope 高度场保留 legacy 的 absolute zero：geom z offset = elevation_min ×
   vertical_scale，因此其物理表面严格等于 raw heightfield × vertical_scale
   （slope 分量在 patch 边缘为 0，但 rough noise 覆盖整个 patch，因此实际边缘
   不保证严格 z = 0，相邻 patch 可存在与 legacy 一致的小噪声高度不连续；
   native 的 HfDiscreteObstaclesTerrainCfg 用同一 absolute-zero 手法）。
   spawn origin z 同样按 legacy
   ``add_terrain_to_map()`` 语义，取 patch 中心 ±1 m 区域的最大 raw terrain height
   （不是全局最大值，也不是 elevation range）；rough slope 的 plateau 带噪声，因此
   该项的 spawn 可高于其 XY 处局部表面，上限为 2 × amplitude（d = 0.9 时约 0.21 m，
   实测 worst 0.17 m），spawn 永不低于所在 patch 的表面（ray-cast 逐格验证）。
10. obstacle height 用 int() 截断（native 实现）：d = 0.7 得到 39 units = 0.195 m
   而不是精确 0.2 m，且 ±h/2 在奇数 units 下略不对称（-0.1 / +0.095）。
```

### 13.3 Terrain curriculum

```text
cfg.curriculum.keys() == {"terrain_levels", "command"}
func = mjlab.tasks.velocity.mdp.terrain_levels_vel（native）
params.command_name = "twist"
command_vel 已移除；"command" 是本单元新增的 forward-speed curriculum（§4.3），
与 terrain_levels 共存、互不覆盖（command 只改 ranges.lin_vel_x）。
```

推进 / 回退公式由 native 实现，与 legacy `_update_terrain_curriculum()` 一致：
walked distance > size[0] / 2 升一级；walked distance < ||command_xy|| x
max_episode_length_s x 0.5 降一级；达到 num_rows 时随机新 level；首次 reset
（common_step_counter == 0）不改 level，因此 `max_init_terrain_level` 生效
（`randint(0, max_init_terrain_level + 1)`，inclusive，与 legacy 相同）。

### 13.4 Terrain scan 与 critic privileged height

flat 与 rough 共用 MjLab v1.6 native `terrain_scan` sensor（`RayCastSensorCfg`），
均把 frame 绑到 `robot/trunk`（与 native rough task 相同）：

```text
name                 terrain_scan（flat / rough 各恰好一个）
frame                robot 的 trunk body
ray_alignment        "yaw"（随 yaw 旋转，不随 roll / pitch 倾斜；ray 恒为 world-down）
pattern              GridPatternCfg(size=(1.6, 1.0), resolution=0.1) = 17 x 11 = 187 rays
max_distance         5.0 m
exclude_parent_body  True
include_geom_groups  (0,)（仅 terrain）
```

`terrain_scan` 的唯一 consumer 是 critic 的 187 维 `height_scan` term；actor 不含它。

observation 维度 contract：

```text
black-flat   actor 45 / critic 259 = 原 privileged 72 + height_scan 187
black-rough  actor 45 / critic 259 = 原 privileged 72 + height_scan 187
```

critic term 顺序被显式冻结（不依赖 native `critic_terms = {**actor_terms, ...}` 的 dict 顺序）：

```text
base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions,
command, foot_height, foot_air_time, foot_contact, foot_contact_forces,
height_scan            ← flat / rough 共用，追加在最后
```

height_scan 数值语义为 native `mjlab.envs.mdp.height_scan()`：

```text
raw   = sensor frame z - terrain hit z   （offset = 0；ray miss 时取 max_distance）
scale = 1 / max_distance = 0.2
noise = 无
clip  = 无
```

flat plane 的 CPU reset/rollout 检查确认 187 条 ray 均命中 z=0，critic 最后
187 维逐值等于 raw height scan × 0.2；rough 原有 scan 与 terrain 检查继续通过。
flat 的 `base_height` reward 仍按 world-z 计算，rough 仍按 local terrain clearance。

Intentional difference（**不是 legacy 238-D / HIM privileged observation migration**）：

```text
legacy（super-dog black_env.py）
    heights = clip(root_z - 0.5 - measured_heights, -1, 1) * 5.0
    采样 x ∈ [-0.8, 0.8] step 0.1、y ∈ [-0.5, 0.5] step 0.1（187 点，与 MjLab 同网格）
    height noise raw scale = 0.1
    privileged = 45 + base_lin_vel 3 + external disturbance 3 + heights 187 = 238

Black flat / rough PPO 当前配置
    MjLab native height_scan 语义（offset 0、scale 0.2、无 noise / clip）
    不含 external disturbance 分量；45 + 3 + 187 的 238-D layout 属于后续 HIM
    observation contract，不在本 baseline 内
```

### 13.5 stairs（up / down，2026-10 stair unit 加入）

用户确认的两个决策：几何用 native box stairs；权重用 super-dog HEAD lineage 分布。

legacy 来源（super-dog `legged_gym/utils/terrain.py` `make_terrain()`，HEAD lineage）：

```text
step_height = 0.05 + 0.18 × difficulty       # d=0 → 0.05 m，d=0.9 → 0.212 m
up / down   ：stairs block 内按 choice 分方向（HEAD 版本 up/down 各 0.25）
step_width  ：恒 0.3 m（legacy 中的"宽度课程学习"插值是 dead code：
              current_step_width 算完未使用，调用写死 step_width=0.3）
platform    3.0 m
几何        heightfield 金字塔台阶（0.1 m 网格 / 0.005 m 量化，台缘 1 格斜坡）
```

MjLab v1.6.0 只有 mesh box 台阶（无 heightfield 台阶 primitive），采用 native：

```text
stairs_up     BoxPyramidStairsTerrainCfg
stairs_down   BoxInvertedPyramidStairsTerrainCfg
step_height_range = (0.05, 0.23)
    → h(d) = 0.05 + d × 0.18，d ∈ [0, 0.9] 与 legacy 公式逐值一致
step_width = 0.3 / platform_width = 3.0（同 legacy）
```

spawn 语义：up 的 origin z = (num_steps+1) × step_height（顶层 platform，实测
0.45 → 1.908 m 随难度递增）；down 的 origin z = -(num_steps+1) × step_height
（底部 platform，实测 -0.45 → -1.908 m）。legacy 的 add_terrain_to_map 取中心
区域最大 raw 高度，对 up/down 台阶同样落在顶层 / 底部 platform，等价。

Intentional framework difference（不声称 exact reproduction）：

```text
1. native 是真实竖直台阶沿的 mesh box；legacy heightfield 台缘是 0.1 m 网格上
   的 1 格斜坡。box 台阶对 policy 略更严格。
2. native 顶层 platform 比 legacy 多一层（(num_steps+1) 步 vs legacy 8 步）。
3. legacy 的"宽度课程学习"插值（0.3 → 0.2 m）是 dead code，不迁移。
```

不变量：terrain curriculum（terrain_levels_vel）、height_scan / critic 187 维、
base_height footprint reward、OOB（effective grid 10 x 7 → limit 59.7 / 47.7）、
command / action / reward / DR / termination 全部不变；nconmax = 128 保持。

------

## 14. In Progress

```text
Black flat reward:                 COMPLETE（§11）
Black flat DR:                     COMPLETE（§12）
Black flat command:                COMPLETE（§4）
Black flat final PPO verification: COMPLETE（§17.1）
Black rough terrain generator:     COMPLETE（§13）
Black rough terrain scan / critic privileged height: COMPLETE（§13.4）
Black rough base-height reward:  COMPLETE（§11.4）
Black rough out_of_terrain_bounds: COMPLETE（§6.4）
Black rough PPO sanity / baseline training: COMPLETE（§17.2）
Black rough MJWarp contact capacity workaround: COMPLETE（§10 / §17.2）
Black actor-only TorchScript export:  COMPLETE（§19.3 / §17.3）
Black legacy rl_sar 45-D deployment config: COMPLETE（§19.6 / §17.4）
Black sim2sim observation / action trace: COMPLETE（§19.7 / §17.5）
Black rl_sar MuJoCo locomotion rollout: COMPLETE（§19.8 / §17.6）
Black quadruped_control 45-D deployment config: COMPLETE（§19.9 / §17.7）
Black quadruped_control observation/action/torque trace: COMPLETE（§19.10 / §17.8）
Black quadruped_control MuJoCo locomotion rollout: COMPLETE（§19.11 / §17.9）
Black cross-runtime policy/safety contract: COMPLETE（§19.12 / §17.10）
Black PPO deployment / sim2sim compatibility: COMPLETE（§19）
Black HIM observation / history / estimator target contract: COMPLETE（§20 / §17.12）
Black HIM algorithm（HIMPolicy / HIMEstimator / HIMPPO / HIMRolloutStorage）: COMPLETE（§21 / §17.13）
Black HIM runner integration / checkpoint / resume: COMPLETE（§22 / §17.14）
black-flat-him registration + PPO→HIM warm start: COMPLETE（§23 / §17.15）
black-rough-him registration + flat→rough HIM full resume: COMPLETE（§24 / §17.16）
Black rough stairs terrain（up / down, native box）: COMPLETE（§13.5）
Black forward-speed command curriculum + checkpoint state: COMPLETE（§4.3 / §4.4）
Wolf robot asset / joint order / actuator contract (stage 1): COMPLETE（§27）
Wolf IMU observation contract: COMPLETE（§27.5）
Wolf flat PPO / HIM / command curriculum integration: COMPLETE（§28）
Wolf Domain Randomization（逐项开关 + minimal profile）: COMPLETE（§29）
Wolf task-local independence + rough PPO/HIM registration: COMPLETE（§30）
Wolf flat 模板 spawn 高度 + flat 容量优化: COMPLETE（§31）
```

command 已冻结为固定范围 + native sampler，且不再有任何 curriculum（§4）。
train / play 的 command contract 完全相同。

部署契约 / sim2sim 兼容（§19）已完成：`rl_sar` 的 config、observation/action trace、
locomotion rollout 均 PASS（§19.6~§19.8）；`quadruped_control` 的 config、
observation/action/torque trace、locomotion rollout 均 PASS（§19.9~§19.11）；
跨 runtime 的 observation、actor、pre-safety `q_policy` 均 PASS（§19.12）。
`quadruped_control` 的硬件位置裁剪在正常 rollout 中低频触发，属于有意的
deployment safety layer，不能描述为 inactive guard，也不要求最终 `q_command` 三边相同。
rough 的 sanity / baseline 训练属于 pipeline / baseline verification，**不是** long-run
convergence 结论。

next decision：

```text
Black command curriculum training validation（§4.3 已实现未长训验证）—
    先跑 black-flat（PPO 或 HIM）观察 vx_max progression / Curriculum/command/ema_low
    / ema_high / tracking ratio / 训练稳定性；随后 black-rough range-resume continuation。
Black HIM training validation — flat HIM 收敛 + flat→rough HIM 训练行为
（registration / resume 机制已全部完成；“算法能跑” ≠ “算法训练有效”，排在课程验证之后）
Wolf rough PPO/HIM training validation（§30 注册完成、未训练验证）—
    先跑 wolf-flat（PPO 或 HIM）基线，确认行为与迁移前一致；
    随后 wolf-flat → wolf-rough resume / warm start 路线与 rough 课程推进。
Wolf flat 容量长训监控（§31 候选值 64/256）— 真实训练下确认 overflow 恒为
    NO；若触发按 §31.3 约定上调并在 §31 登记实测下限。
```

Black real robot backend / sim2real contract 仍等待用户决定，不自动开始实现。
进入实机前仍须单独决定 inference deadline / 线程配置与 torque 语义（§19.12）。
`black-flat-him` / `black-rough-him` 均已注册；flat→rough HIM full resume 已验证。
ONNX metadata 归属见 §18.1 / §19.4。

------

## 15. Deferred Black Flat Work

尚未迁移/冻结：

### Reward

Black flat v1 reward baseline 已完成并冻结（10 项，见 §11）。不再有「待迁移 reward」。

super-dog 的自定义 shaping 与 MjLab-only 项均未采用（见 §11.3）；若日后因训练问题
需要重新引入，必须作为新的 behavior unit 提出，先说明动机与失败现象。

------

### Domain Randomization

Black flat v1 的 DR 已完成并冻结（6 项，见 §12）。

仍未迁移（见 §12.3）：

```text
link mass
inertia
motor strength
external force disturbance
action delay
restitution
```

不要在 reward 或其他任务中顺手改 DR。

------

### Command Curriculum

Black 的 performance-based forward-speed command curriculum 已随本单元迁移并冻结
（§4.3 / §4.4）：super-dog Black 后期 buffer / EMA / pass streak 状态机 + 10 类
telemetry 日志 + checkpoint restore（none/range/full）。official HIMLoco 的 env-index
bucket sampler 仍不迁移（§4.5）。

------

### Critic Observation

flat / rough critic 均在原 MjLab privileged observation 72 维之后追加 187 维
terrain height scan（共 259 维，见 §13.4）；flat plane 也保留 raycast。
旧 Black/HIM privileged critic layout（238-D）尚未迁移。
这次只改 critic 输入；actor 45-D 与部署接口不变。新 flat checkpoint 向 rough
完整续训的 checkpoint 加载已完成小规模 CPU smoke；训练收敛需后续验证。

------

## 16. Explicitly Not Started

以下均未开始，不得提前宣称支持：

```text
Black command curriculum training validation（§4.3 已实现，未做长训练效果验证）
Black HIM 长训练收敛验证（flat HIM 收敛 + flat→rough HIM 训练有效性）

Black real robot backend / sim2real
    （ONNX metadata 归属见 §18.1 / §19.4）

HIM ONNX exporter（HIM TorchScript 导出已完成，见 §19.3；当前部署只消费 TorchScript）

blackw-flat

blackw-rough
```

HIM task-side observation / history / estimator target contract（§20）、HIM 模型 /
HIMPPO 算法（§21）、`MjlabOnPolicyRunner` 接入与 checkpoint / resume（§22）、
`black-flat-him` 注册与 PPO→HIM warm start（§23）、`black-rough-him` 注册与
flat→rough HIM full resume（§24）均已完成。rough stairs terrain（§13.5）已完成。

------

## 17. Local Verification Baseline

本地：

```text
tests/check_black_flat.py        flat 的 migration verification tool
tests/check_black_rough.py       rough 的 terrain / terrain scan verification tool
tests/check_black_rough_him.py   rough HIM 注册 + flat→rough HIM full resume verification
tests/check_black_him.py         HIM task-side contract（Unit 1）
tests/check_black_him_algo.py    HIM 算法（Unit 2）
tests/check_black_him_runner.py  HIM runner / checkpoint（Unit 3）
tests/check_black_him_warm_start.py  flat HIM 注册 + warm start（Unit 4）
tests/check_black_command_curriculum.py  forward-speed command curriculum（§4.3 / §4.4）
tests/render_black_rough.py      rough terrain 的可视化渲染（人工检查用）
```

作为 migration verification tool。

`tests/check_black_rough.py` 覆盖（见 §13 / §13.4 / §13.5）：flat 仍为 plane / 仅
command curriculum，与 rough 共用 terrain_scan 和 critic layout；rough generator 的
size / num_rows / difficulty_range / border / max_init / 7 类 sub-terrain（含 native box
stairs up/down 的 step_height_range / step_width / platform）与 proportion；curriculum
含 terrain_levels + command；逐行难度
0.0 → 0.9；slope = 0.7d、rough noise ±(0.015 + 0.1d) / step 0.005 / downsample 0.2、
obstacle height 0.06 + 0.2d；50 个 spawn origin 的 ray-cast 落面检查；terrain_scan
sensor 唯一性与 frame / alignment / grid 187 rays / max_distance；actor 45-D 与
critic 259-D（末 187 维 = height_scan，scale 0.2、无 noise）的 config 与 runtime 断言；
ray miss = 0 与 scan 数值 sanity；teleport 探针区分 flat / 双 slope / rough slope /
obstacle 的 scan 形状；base_height reward 的 flat（world-z）/ rough（terrain-relative）
config 差异与其余 9 项不变量；rough 的 35 ray footprint 索引 contract（x ∈ [-0.3, 0.3] /
y ∈ [-0.2, 0.2]，index = i_y × 17 + i_x）与数值 invariance（flat 等价 / raised-terrain
不变 / slope / too high-low 对称 / footprint 外 ray 不参与）+ 逐 terrain 实测表；
termination contract（flat train/play 与 rough play 不含 OOB、rough train 末尾恰好一个
native OOB、time_out=True、params 为空）与 OOB runtime（由 runtime terrain config 推导的
effective 10 x 5 grid / 60.0 / 40.0 / 59.7 / 39.7 、严格 > 边界 probe、
``time_outs`` True 而 ``terminated`` False 的 truncation 语义、OOB reset 后 curriculum
level 合法且无 NaN、plane 恒 False、正常 rollout OOB fires = 0）；
stairs 列的 spawn z 随难度单调（up 递增 / down 递减负值）与 spawn 落面；
少量 env 的 zero / random rollout smoke。

当前应持续覆盖：

```text
Black asset compile

nq / nv / nu

default pose

per-joint PD

effort limit

action dimension/order/mapping

command contract（初始范围 / native sampler 比例 / performance-based command curriculum
（§4.3）；旧 time-based curriculum 移除的 regression：step counter 推到旧 stage 后 range 不变）

45-D actor observation

actor observation term order

actor joint order

actor scaling/noise

reward table（10 项 term / func / weight / params，无 zero-weight 占位）

tracking reward formula / invariance / dt scaling

base motion stability penalty（v_z² / ω_xy²）formula / invariance / 与 tracking 的解耦

orientation penalty（L2）formula / 单轴 tilt / 对称性 / yaw invariance / native exp 语义差异

base height penalty（L2）formula / target-above-below / XY·orientation·velocity invariance / 无地形依赖

regularization 组（dof_acc 有限差分 / joint_power / action_rate / smoothness）解析值与 dt 缩放

domain randomization（friction 共享性 / payload / COM / PD 范围与 reset 重采样 /
encoder bias 只影响 observation / push 只扰动 xy / play 无 DR）

last_action mapping

contact sensors

illegal-contact termination

stuck termination

root reset

joint reset

joint-reset soft-limit support

partial reset

finite zero/random rollout

PPO short training

parameter update

checkpoint save/reload

inference

actor-only TorchScript 导出（[1,45] -> [1,12]、与 actor 的 deterministic forward 数值等价、
独立 torch.jit.load）
```

新增 behavior unit必须增加对应 contract test，但不得弱化已有 smoke。

### 17.1 Final PPO verification run record

Black flat PPO 的最终验证在 2026-09-17 完成，verdict 见本小节末尾。

```text
code state        HEAD 61626dc（production 无改动）
task              black-flat
training command  uv run train black-flat --env.scene.num-envs 4096 \
                      --agent.max-iterations 500 --agent.logger tensorboard \
                      --agent.run-name sanity500
device            NVIDIA RTX 4060 Laptop 8 GB（cuda:0，单卡）
num_envs          4096
iterations        500（= max_iterations，正常跑到最后一次 save）
steps_per_env     24
transitions       4096 × 24 × 500 ≈ 4.92e7
checkpoint        logs/rsl_rl/black_velocity/2026-09-17_10-35-28_sanity500/model_499.pt
日志              <run_dir>/events.out.tfevents.*    （tensorboard logger）
```

训练曲线（从 run 的 event 文件实测，首值 → 末段）：

```text
Train/mean_reward                    -0.68  → 22.49
Train/mean_episode_length             23.7  → 1000（= 20 s 满）
Episode_Reward/track_linear_velocity  0.002 → 0.889（raw ≈ 0.89）
Episode_Reward/track_angular_velocity 0.001 → 0.411（raw ≈ 0.82）
Episode_Termination/illegal_contact    37.8 → 0.0
Episode_Termination/time_out            4.2 → 3.09（末段终止 ≈100% 为走满 20 s）
Episode_Termination/stuck                 0 → 0
Loss/entropy                          16.96 → 1.46
Policy/mean_std                        0.99 → 0.275
Perf/total_fps                              ≈ 96k
```

独立进程 checkpoint 重载 + 固定 command 短评估（play contract：无 DR / 无 corruption，
16 envs，每档 5 s，确定性策略）：

```text
command                abs err (vx / vy / wz)      root z   终止
stand                  0.001 / 0.001 / 0.003       0.452    0
forward   [0.5,0,0]    0.036 / 0.018 / 0.036       0.455    0
backward  [-0.5,0,0]   0.050 / 0.033 / 0.052       0.434    0
lat left  [0,0.5,0]    0.012 / 0.130 / 0.042       0.449    0
lat right [0,-0.5,0]   0.022 / 0.060 / 0.088       0.428    0
yaw left  [0,0,1.0]    0.016 / 0.028 / 0.088       0.441    0
yaw right [0,0,-1.0]   0.032 / 0.037 / 0.138       0.451    0
combined  [0.5,.3,.5]  0.027 / 0.037 / 0.044       0.453    0

8 档 × 16 envs × 5 s 内 0 次终止（无 illegal_contact / stuck）
```

play 定性（用户在 viser viewer 中手动改变 command 观察）：

```text
stand / forward / backward / lateral / yaw / combined 均正常响应
未触发 illegal_contact / stuck
已知弱点：轻微拖脚（foot dragging）
```

verdict：

```text
Black flat PPO pipeline:              VERIFIED
Black flat locomotion baseline:       VERIFIED（基于 500-iteration run）
10 000-iteration 长程收敛性:           NOT VERIFIED（按用户决定不跑满）
```

即结论是「训练 pipeline 能稳定跑通并产生基本可用 flat locomotion policy」，
不是最终性能结论：rough terrain / 强扰动恢复 / 高速 / 完美 trot / sim2real 均未验证。

拖脚不作为 reward 配置缺陷处理：v1 baseline 有意不含 foot_clearance（见 §11.3），
若需要抬脚高度约束，须作为新的 behavior unit 提出（见 §15）。

标准验证：

```bash
uv run python tests/check_black_flat.py --device cpu
```

CUDA可用时：

```bash
uv run python tests/check_black_flat.py --device cuda
```

测试结果必须明确写：

```text
CPU PASS/FAIL
CUDA PASS/FAIL/SKIPPED
```

### 17.2 Black rough PPO baseline 与 MJWarp capacity workaround 验证记录

rough sanity / baseline run（由用户手动运行，agent 未代跑）：

```text
logs/rsl_rl/black_velocity/2026-09-18_19-24-21_sanity200/    run_name sanity200，至 model_199.pt
logs/rsl_rl/black_velocity/2026-09-18_19-37-17/              resume 续训，至 model_498.pt（共约 500 iteration）
```

该 run 的记录配置（`params/agent.yaml` / `params/env.yaml`）：`num_steps_per_env 24`、
`max_iterations 300`、`save_interval 50`、`resume true`；`num_envs 4096`、
`terrain_type generator`、`episode_length_s 20.0`。artifact 完整（model_498.pt + tfevents + .onnx），
末次记录（step 498）：`Train/mean_reward` 13.39、`Train/mean_episode_length` 914.4（上限 1000）、
`Episode_Termination` illegal_contact 0.92 / stuck 0 / time_out 3.54 / out_of_terrain_bounds 0。
checkpoint 在 play 下可完成基本 locomotion。

范围：这是 **pipeline / baseline verification**（约 500 iteration），**不是** long-run
convergence 结论（未做长训练，也未做 8-command 定量评估）。

MJWarp capacity 验证（固定 bad simulator state，`tests/repro_black_rough_multiccd.py`）：

```text
同一 bad state + 1 次 state restore + 1 次 sim.step()
    nconmax = 35     → 确定性 crash（ccd_kernel_builder，CUDA memory-access fault）
    nconmax >= 48    → clean（48 / 64 / 96 / 128 / 256）
生产 rough cfg（nconmax = 128，无 CLI override）    → clean
长 rollout（5000 control steps，model_498，CUDA graph OFF，nconmax = 128） → 无 crash
```

诊断结论：CUDA graph 与 MULTICCD 都不是根因；无 NaN / Inf；compute-sanitizer 指向 CCD
kernel 的 capacity-related boundary access；upstream MJWarp 根因尚未修复。

### 17.3 Black PPO actor-only TorchScript 导出验证记录

```text
code state        HEAD eda3e82 + 新增 src/alldog_mjlab/utils/export_policy.py（未提交）
task              black-rough（生产 CLI）/ black-flat（check_black_flat）
checkpoint        logs/rsl_rl/black_velocity/2026-09-18_19-37-17/model_498.pt（iter 498）
export command    当时使用：uv run python -m alldog_mjlab.utils.export_policy \
                      --task-id black-rough --checkpoint <model_498.pt> \
                      --output <policy.pt> --device cpu
exported module   actor-only TorchScript：input float32 [1,45] / output float32 [1,12]
```

数值等价（reference = 刚加载 checkpoint 的 actor deterministic forward，atol 1e-6 / rtol 1e-5）：

```text
play 环境真实一帧   max abs diff 0.0
全零 [1,45]        max abs diff 0.0
linspace(-1,1,45)  max abs diff 0.0
```

独立加载（不依赖 runner 对象）：`torch.jit.load()` + `eval()` 后可推理，输出 [1,12] float32 有限。
上述命令是历史验证记录；当前 exporter CLI 参数见 §19.3。

```text
uv run python tests/check_black_flat.py --device cpu    PASS（含导出检查）
uv run python tests/check_black_flat.py --device cuda   PASS（含导出检查）
```

### 17.4 legacy rl_sar 45-D deployment config 验证记录

```text
rl_sar HEAD         cd46b93（coverMoon/rl_sar-for-super-dog，变更前 clean）
新增（untracked）   src/rl_sar/policy/black/mjlab_ppo/{config.yaml, black_ppo_actor.pt}
未修改              policy_switch.yaml / base.yaml / 任何 C++ 源码
```

验证按顺序执行：

```text
1. config 静态 contract    num_observations 45 / observations_history [] / 12 个 action /
                           default pose / scale / PD / torque / joint_mapping 全部匹配
2. TorchScript 独立加载     torch 2.14：zeros(1,45) -> (1,12) float32 finite
                           libtorch 2.0.1+cpu（runtime 实际链接版本）：同样 PASS，
                           zeros 输出与训练侧逐位一致
3. build                    bash build.sh rl_sar → PASS（46.2 s，仅警告）
4. minimal runtime load     rl_sim + black/mjlab_ppo，debug_key 注入 0 → 1：
                           fsm_state 稳定保持在 RLFSMStateRL_Locomotion，
                           runtime_status.model_name = black_ppo_actor.pt，
                           InitRL() failed = 0 次，无异常 / 维度错误
负向对照                    不存在的 config name → [ERROR] InitRL() failed: bad file，
                           证明该路径能显式暴露加载失败
文档启动方式                bash run_rl_sim_debug.sh black mjlab_ppo 同样 PASS
                           （fsm_state / model_name 同上，无 InitRL failed）
```

该次尚未做 MuJoCo backend rollout、observation/action trace 数值对比或运动表现评价；
后续结果见 §17.5 / §17.6 / §17.10。

### 17.5 Black PPO sim2sim observation/action trace 验证记录

```text
MuJoCo backend   /home/windnotebook/PROJECT/Dog/real_robot/black 的 mujoco_runner（ROS 2）
MuJoCo 启动      ros2 launch mujoco_runner mujoco.launch.py rname:=black scene:=flat \
                     render:=false real_time:=true publish_gap_model:=false
rl_sar 启动      rl_sim --ros-args -p robot_name:=black -p policy_config:=mjlab_ppo
（非零 command） navigation mode + /cmd_vel（键盘命令会被 RunModel 的 joy_timeout 清零）
trace            2 次 session，共 2829 个 policy step（1324 / 1505），连续无缺号
instrumentation  rl_sar 侧 env-gated（RL_SAR_TRACE）临时 trace，验证后已整体回退，
                 git diff 为空并已重新 build（回退后 smoke：InitRL OK / 0 ERROR）
reference        独立 numpy 实现，从 raw simulator state（quat / gyro / command / q / dq /
                 previous action）按 deployment contract 重算 45-D；joint reorder 由显式
                 joint_mapping 完成，不以 rl_sar 处理后的张量为输入
```

```text
obs.command      ≤ 2.0e-9        raw_action（Python torch 2.14 vs libtorch 2.0.1） ≤ 9.6e-7
obs.ang_vel      ≤ 5.0e-9        scaled_residual（0.25 * action）                  ≤ 2.4e-7
obs.gravity      ≤ 2.2e-7        q_target（default + 0.25 * action）              ≤ 2.9e-7
obs.q_rel        ≤ 8.0e-8        obs.full[45]                                    ≤ 2.3e-7
obs.dq           ≤ 7.0e-8        clip_obs = 100 触发                            false
obs.last_action   0.0            （trace 内 max |obs| = 4.40）
```

```text
previous_action timing   2828 次连续转移均为 prev_action_t == raw_action_(t-1)，
                         step 1 为 zeros(12)；不存在 == 当前 action 的 step
状态覆盖                 A zero command 站立 784 帧；B non-zero command 2045 帧
                         （vx ∈ [-2,2] / vy ∈ [0,0.9] / wz ∈ [0,3]）；
                         C tilt > 5° 476 帧（max roll 8.35° / max pitch 13.6° /
                         max |gyro| 11.96 rad/s）；未出现大角度持续倾倒
```

结论：**PASS**（observation / action / q_target 全部在浮点容差内一致，
joint mapping / gravity / ang_vel frame / previous_action timing 均无 contract 冲突）。

### 17.6 Black PPO rl_sar MuJoCo locomotion rollout 验证记录

```text
backend      real_robot/black mujoco_runner（scene=flat, real_time, publish_odom=true）
policy       black/mjlab_ppo（TorchScript，50 Hz）；low-level / sim dt = 0.005 s（200 Hz）
command      navigation mode + /cmd_vel（键盘 command 会被 joy_timeout 清零）
方法         4 个 session / 22 个 command 段；每段 3 s 过渡 + 9 s 测量窗口
             线速度用 odom pose 中心差分（world→body），角速度用 IMU gyro（body frame）
             扭矩用 rl_sar 计算值（临时 env-gated trace，验证后已回退并重新 build）
```

```text
command              cmd             measured          判定
zero（×4 session）    0,0,0          0.000             PASS 稳定站立（roll≤3.6° pitch≤3.5° z 0.478）
vx +0.2              0.2            0.000            standing fixed point
vx +0.3              0.3            0.175 / 0.182    欠跟踪
vx +0.6              0.6            0.548            PASS
vx +1.0              1.0            0.800            PASS（38% 帧 tau>20）
vx -0.2 / -0.3       -0.2 / -0.3    0.000 / -0.033   standing fixed point
vx -0.6              -0.6           -0.548           PASS
vy +0.3 / -0.3       0.3 / -0.3     0.144 / -0.176   部分（48% / 59%）
vy +0.6 / -0.6       0.6 / -0.6     0.505 / -0.554   PASS（84% / 92%）
wz ±0.5 / ±1.0（纯）  0.5/1.0        0.003~0.006       FAIL（站立时不旋转）
wz 0.5 + vx 0.2/0.3  0.5            0.133 / 0.197    yaw 被解锁，仍欠跟踪
wz 0.5 + vy 0.2      0.5            0.202            yaw 被解锁
(0.5,0.2,0.5)        三者           0.400/0.179/0.334 PASS（66~80%）
(0.5,-0.2,-0.5)      三者           0.508/-0.150/-0.289 PASS
```

全程无 NaN / Inf、无摔倒、roll ≤ 6.4°、pitch ≤ 5.6°、base z ∈ [0.441, 0.485]。

```text
torque saturation（rl_sar ComputeOutput 计算值，33.5 N·m 诊断张量裁剪；非实机命令限幅）
全部帧 9858： |tau|>20 任一关节 2.2% / 关节样本 0.2% / 从未达 33.5（max 27.79 N·m）
zero command： 0%
locomotion：  2.4%（集中于 FL_calf / FR_calf，max 25.62 / 27.79）
最高：vx=+1.0 段 38.2% 帧存在 >20
```

结论：**PASS（基本 locomotion 能力成立）**，但记录两项 limitation：
（L1）低速/静止下存在 standing fixed point（详见 §19.8）；
（L2）关节范围 asset 差异（部署 ±10 rad vs 训练软限，实测 calf 超出 0.09~0.13 rad）。

### 17.7 quadruped_control 45-D deployment config 验证记录

```text
quadruped_control HEAD  7a36118（变更前 clean）
新增                    configs/policies/black/mjlab_ppo.yaml
                        assets/policies/black/mjlab_ppo/black_ppo_actor.pt
                        （md5 e7318ea9019cf27f2d204565659cf0cd，与 alldog_mjlab 导出产物逐字节一致）
修改                    assets/policies/black/SOURCE.md（补 mjlab_ppo 来源与 md5）
未修改                  policy_switch.yaml / 任何 runtime 代码 / flat.yaml / obstacle.yaml
build                   ./scripts/build.sh --target motion → PASS，ctest 8/8 passed
```

严格走正式运行时链路（`/tmp/verify_mjlab_ppo_quadruped.cpp`，链接 .build/rl 的
quadruped::config / motion / torch_policy）：

```text
load_robot_model         12 关节，顺序 FL/FR/RL/RR ⟨hip/thigh/calf⟩，与 policy 顺序一致
load_rl_config           observation 45 / history_frames [0] / input 45 / action 12
                         policy_dof_indices 恒等 / default pose / kp 40 / kd 1.2 / action_scale 0.25
负例                     history_frames=[0,1] + input=45 → validate_rl_config 拒绝
RlController::create     PASS
build_observation        45 维；与独立 Python 参考实现 max_abs_diff 2.4e-8
insert + inference_input dimension 45（不是 270），且与当前观测逐元素相同；45 维之后无残留
TorchPolicy::create      PASS（含 3 次 warmup forward）
forward                  output 12 维 finite；与 Python torch 2.14 torch.jit.load max_abs_diff 1.8e-7
                         零输入输出与训练侧记录逐位一致（0.134997 / -0.423780 / ...）
convert_actions          q_target = default + 0.25 * action（max_abs_diff 5e-9）、kp/kd 40/1.2
                         未触发 position limit / max_position_jump；|raw action|max = 1.87 ≪ action_clip 100
```

app 级启动 smoke（`quadruped_mujoco_sim --policy-switch-config` 指向 /tmp 下的
symlink switch 配置，仓库内 policy_switch.yaml 未改）：进程正常运行，无配置/加载/维度错误，
即已覆盖 config parse → TorchPolicy::create（warmup forward）→ attach_policy →
RlController::create。后续正式 app rollout 结果见 §17.9。

### 17.8 quadruped_control observation/action/target/torque trace 验证记录

```text
quadruped_control HEAD   a0970dd（变更前 clean，本单元零改动）
方式                     /tmp/trace_quadruped_mjlab_ppo.cpp：用公开接口自己驱动仿真
                         2 ms 物理 / 5 ms 控制（与 tests/mujoco 同构）
                         包装 RobotIO 记录 MotionRuntime 真正 submit 的 CommandFrame
                         包装 Policy 记录每次推理的 45-D 输入 / 12-D 输出
                         io.raw_data()（公开）读 data.ctrl / actuator_force / qfrc_actuator
trace                    3206 个控制周期 / 751 个 policy step（≈16 s）；rl_locomotion Running
```

```text
obs.full[45] 与独立 numpy 参考      max_abs_diff 3.2e-8（各 block ≤ 2.4e-8）
history [0] 单帧                    input_dim 恒为 45，非 270
previous_action                     750 次连续转移均 == a_(t-1)，step 1 为 zeros
raw_action（C++ libtorch vs Python） max_abs_diff 4.8e-7
q_target = clamp(default+0.25*a)    max_abs_diff 5.0e-9
                                    （position limit 命中 3 帧 / 1 个关节 FL_calf；jump limit 0）
torque 链                           tau_raw 重算 7.3e-7；
                                    tau_cmd == clamp(tau_raw, ±max_effort) 7.3e-7；
                                    MuJoCo actuator_force == tau_cmd 0.0；
                                    MuJoCo qfrc_actuator == tau_cmd 0.0（无第二级 clamp）
```

```text
torque（|tau_raw|，N·m）  RL 各阶段 max ≤ 16.6，>20 = 0.00%，被 max_effort 截断 = 0.00%
                          仅 startup（GetUp/Stand，固定 80/3 控制器）达 34.31、>20 占 2.43% 周期
                          逐关节：thigh max 23.69（贴 23.7 但未超）、calf max 34.31（均发生在 startup）
关节范围                training MJCF == RobotModel == quadruped_control MuJoCo（逐项相同）
```

结论：**PASS**（policy I/O 与 torque 链全部对齐；各仿真执行器配置上界的差异不影响本轮 policy 轨迹，
因为 RL 阶段从未超过 20 N·m）。

------

### 17.9 quadruped_control Black PPO MuJoCo locomotion rollout 验证记录

```text
quadruped_control HEAD     a0970dd（本单元 production 零改动；插桩临时 env-gated，已完全回退）
方式                       quadruped_mujoco_sim 正式 app（MotionRuntime + MuJoCo backend +
                           TorchPolicy + TerminalInput 键盘命令通道），pseudo-TTY 自动化
场景                       flat：assets/robots/black/mujoco/scene.xml
                           ⚠ app 默认场景是 scene_terrain.xml（rough 地形），必须显式 --scene
会话                       6 个短会话（每会话 2-4 个命令段，RL ≤ ~30 s）
trace                      167422 个物理步（2 ms）/ 109458 个 RL cycle / 10949 次策略推理
                           全程 rl_locomotion Running，无 fall / 无 NaN / 无 error_message
```

```text
检验项                            结果
previous_action                  10943/10943 连续转移 == a_(t-1)，step 1 为 zeros
推理耗时（仅真实推理帧）          mean 1.91 ms，p50 0.43 ms，p99 4.83 ms，max 12.85 ms
                                  > 20 ms 占 0.000%（10949 次）
base 高度                         mean 0.425 ~ 0.470 m，全局 min 0.422 m
roll / pitch                      |roll| ≤ 5.3 deg，|pitch| ≤ 5.7 deg
力矩 |tau_raw|                     全局 max 20.93 N·m（B3_fwd_1p0 的 calf）
                                  > 20 N·m 占样本 0.059%；> 23.7 / 33.5 / 59.25 均 0.000%
                                  被 max_effort 截断 0.00%（全部帧未触到力矩上限）
硬件关节位置裁剪                  已触发：|q_command - q_policy| > 0.5 rad 的样本 0 个；
                                  > 0.05 rad 仅 B3 段 83 个样本（max 0.131 rad）；jump 限位 0 次
关节范围违反（相对训练限）        仅 B3 段 max 0.024 rad / 164 个样本（MuJoCo 软约束穿透），其余 0
```

运行时约束发现（环境时序问题，不是策略 / 契约问题）：

```text
motion_runtime.cpp:821    单次推理 > 20 ms 即 fail → 行为从 rl_locomotion 掉回 Passive
默认多线程                偶发 22 ms 卡顿，命中一次即整段作废（6 个会话中曾 4 次失败）
OMP_NUM_THREADS=1         长尾消失（见上表），本单元全部有效数据均在此设置下采集
```

结论：**PASS**（工程级：站立稳定、方向正确、无跌倒 / NaN / 持续饱和、契约无冲突），
两条已知限制见 §19.11。

------

### 17.10 Black PPO 训练 / 部署跨 runtime contract 验证记录（COMPLETE）

上一单元采集、此次仅重新分类，未重跑：MjLab flat deterministic play 1000 帧，
从真实 rollout 选 30 个带来源标记的 canonical raw-state fixture（站立、前进、组合动态、
低速后退、纯 yaw）。独立 reference 从 command / wxyz quaternion / body angular velocity /
policy 顺序 q、dq / previous action 构造 45-D；各 simulator 的 live trajectory **未**按时间逐帧相减。

```text
MjLab live observation vs independent reference    max_abs_diff 5.96e-8（1000 帧）
MjLab previous_action                             995 次连续转移均为 a_(t-1)
同一 45-D 输入：MjLab actor vs Python TS             max_abs_diff 4.77e-7
同一 45-D 输入：MjLab actor vs rl_sar LibTorch       max_abs_diff 4.77e-7
同一 45-D 输入：MjLab actor vs quadruped TorchPolicy max_abs_diff 4.77e-7
同一 45-D 输入：rl_sar vs quadruped_control          max_abs_diff 0
policy joint order                                 FL/FR/RL/RR，hip/thigh/calf
policy period                                      三边 0.02 s / 50 Hz
同一 raw action 的 pre-safety q_policy              default + 0.25 * action，三边一致
```

最初按“三边最终 `q_target` 必须相同”的 strict criterion 判 FAIL：MjLab 正常前进
rollout 的 1000 个 policy frame 中，25 帧的 `FL_calf_joint` 的 `q_policy` 超出位置限，
`quadruped_control` 最大一次将 `-0.81150 rad` 裁为 `-0.85000 rad`，差 `0.03850 rad`
（约 `2.2°`）。用户随后确认该位置限经过真实硬件验证；正确的比较边界是安全处理前的
`q_policy = q_default + 0.25 * raw_action`。该值三边一致；裁剪后的 `q_command`
有意不同。MjLab 实际 target 对独立 `q_policy` 参考最大差为 0；`rl_sar` 的
`ComputeOutput()` 与 `quadruped_control` 裁剪前计算均使用同一公式。30 个代表性
fixture 未触发裁剪，但完整 1000 帧证明位置安全层在正常
policy 分布中**低频生效**。`max_position_jump = 1.0 rad` 独立于位置限；相邻 target
变化最大约 `0.208 rad`，本轮未触发 jump limiter。

重新分类：observation PASS；actor PASS；policy action / pre-safety `q_policy` PASS；
`quadruped_control` 硬件位置安全层 PASS；final commanded target INTENTIONALLY DIFFERENT；
Black PPO sim2sim compatibility **PASS**。低速后退静止与纯 yaw 不旋转在 MjLab play
也复现，是三边共享的 policy 行为。原始量测保留，未将 25 帧描述为 inactive guard。

### 17.11 dcb21311 trunk-root MJCF structural regression（PASS）

`dcb21311` 删除无功能的 dummy `base`，将 `trunk` 提升为 floating root；
此前的 500-iteration 训练、部署 rollout 和跨 runtime fixture 均采自旧 asset。
以 `a214092` 和 `dcb21311` 两版 XML 分别经 MjLab v1.6.0 Entity/MuJoCo 编译比较：
`nq/nv/nu=19/18/12`、12 个 joint 顺序/轴/范围/地址、12 个 actuator 绑定及
`±20 N·m` 范围、default keyframe（root `[0,0,0.45]`，含全部 joint q/ctrl）
完全一致；仅 `nbody 21→20`、`ngeom 50→49`。`trunk` 自身质量 `5.7042 kg`
及惯量不变，整机质量 `13.247181→13.247180 kg`（仅去除 dummy 的 `1e-6 kg`），
default-pose COM 最大差 `7.71e-9 m`、对齐广义坐标后的质量矩阵最大差 `1e-6`。
trunk、IMU 和四足 world pose 差为零；传感器绑定不变，静态值差 ≤`7.1e-16`，
相同非零 qvel 下 gyro 差 0、accelerometer 差 `4.8e-8`。
当前 flat/rough 环境中的 trunk/thigh contact、四足碰撞、terrain-scan、viewer、
payload/COM selectors 均能解析。`tests/check_black_flat.py --device cpu` 与
`tests/check_black_rough.py --device cpu` 均 PASS。冻结的 rough `model_498.pt`
在当时因 critic 输入 259-D 不能整包加载到旧 flat 72-D critic；使用 MjLab/RSL-RL 原生
actor-only checkpoint load，在 black-flat play 环境以 zero 与 `vx=0.6` 各走
100 policy step：actor 45→12、无 NaN/重置/立即摔倒，root 最低约 0.440/0.449 m。
**Classification：behavior-neutral structural asset fix，regression PASS。**
本次没有重跑历史 500-iteration 训练或跨 runtime 全量 trace。

------

### 17.12 Black HIM observation / history / estimator target contract（PASS）

新增 `src/alldog_mjlab/tasks/velocity/black/him.py` 与
`black_flat_him_env_cfg` / `black_rough_him_env_cfg`（`env_cfgs.py`），
并在 §20 冻结 HIM task-side contract。
本地验证工具：`tests/check_black_him.py`（不提交 Git）。

结果：

```text
CPU  PASS
CUDA PASS（RTX 4060 Laptop 8 GiB，sim.use_cuda_graph = True）
```

覆盖：普通 `black-flat` actor 45 / critic 259 不变；HIM actor `[B, 6, 45]`，当前帧
与从 raw state 重建的 45-D 一致；MjLab 内部 oldest → newest，
`flip(1).flatten(1)` 得到 newest → oldest canonical；full / partial reset 后 history
全帧 backfill 且未 reset env 不被污染；`estimator_velocity` `[B, 3]` =
`root_link_lin_vel_b × 2.0` 且与 IMU velocimeter 一致；target encoder input
= 42 + 3 = 45 且不等于 critic slice；terminal transition 用 `auto_reset=False` 的
terminal observation 作权威值，play 配置逐值一致，training 配置 velocity 逐值一致、
frame / velocity 均不等于 post-reset observation。

既有回归：`tests/check_black_flat.py --device cpu` 与
`tests/check_black_rough.py --device cpu` 均 PASS。
本单元未重跑长训练，也未实现 HIM estimator / HIMPPO / exporter。

------

### 17.13 Black HIM algorithm（model + HIMPPO）on RSL-RL 5.4.2（PASS）

新增 `src/alldog_mjlab/algorithms/him/{spec,estimator,policy,storage,ppo}.py`，
删除 legacy `actor_critic.py` / `runner.py`。本地验证工具：
`tests/check_black_him_algo.py`（不提交 Git）。

结果（Black flat HIM env，8 envs，16 policy steps，episode 0.2 s）：

```text
CPU  PASS
CUDA PASS（RTX 4060 Laptop 8 GiB）
```

关键验证：

```text
spec               45 / 6 / 270 / 3 / 3 / 16 / 12；target_dim 45；actor_input_dim 64
history adapter     canonical == history.flip(1).flatten(1)，且与 task-side helper 一致
policy forward      latent L2 norm = 1；action / mean / std / entropy / log_prob shape 正确
estimator loss      synthetic 与真实 rollout batch 均 finite；sinkhorn 无 NaN/Inf
optimizer 分账      PPO 17 params（actor MLP 9 + critic 8）；estimator 13 params；交集空
gradient boundary   PPO actor backward 后 encoder grad == None；estimator backward 后
                    encoder / target / prototype grad 非零（实测 ~40 / ~0.04 / ~0.007）
storage             next_estimator_input [16,8,45]、next_estimator_velocity [16,8,3]，
                    与 capture 的 successor observation 逐步一致
terminal override   rollout 中 1 个 terminal step（8 rows）逐值与 Unit 1 terminal pre-reset
                    数据一致，且 != post-reset observation
one real update     value 0.06 / surrogate -0.05 / entropy 17.02 / estimation 0.46 /
                    swap 0.108 均 finite；estimator / policy / critic 参数确实变化
checkpoint state    save() 包含 estimator_optimizer_state_dict；新实例 strict load smoke PASS
                    （完整 round-trip / resume 留待 Unit 3）
```

既有回归：`tests/check_black_flat.py --device cpu` PASS；`tests/check_black_him.py
--device cpu` PASS。
本单元未注册 task、未做 warm start、未验证 checkpoint round-trip、未实现 exporter、
未做长训练 / 收敛验证。

------

### 17.14 Black HIM runner / checkpoint / resume（PASS）

新增 `src/alldog_mjlab/tasks/velocity/black/him_rl_cfg.py`，共享 refactor 在
`rl_cfg.py`（`black_ppo_algorithm_kwargs` / `black_critic_model_cfg` /
`black_actor_distribution_cfg`），`black_config.py` 增加 HIM-only `him` section 与
`flat_him` / `rough_him` run 名。本地验证工具：`tests/check_black_him_runner.py`
（不提交 Git）。

结果（Black flat HIM env，8 envs，12 steps/rollout，episode 0.2 s）：

```text
CPU  PASS
CUDA PASS（RTX 4060 Laptop 8 GiB）
```

关键验证：

```text
config layering    12 项 PPO 超参数 / critic cfg / actor dims 与普通 PPO 逐值一致，
                   仅 HIM 独有字段（class、encoder、latent、prototype、estimator lr）单独声明
construction       MjlabOnPolicyRunner → HIMPPO → HIMPolicy + MLPModel + HIMRolloutStorage
obs routing        obs_groups actor=[actor] / critic=[critic]；estimator_velocity 不进入两者
module ownership   actor 22 params（estimator 13 + mlp 8 + std 1）无重复注册；
                   estimator.state_dict() 带前缀后是 actor.state_dict() 子集；无 estimator_state_dict 重复
runner learn       learn(2) 真实跑通 rollout/update；loss = value 0.014 / surrogate -0.068 /
                   entropy 17.02 / estimation 0.356 / swap 0.108，均 finite
checkpoint         lr 3e-4（模拟 KL 调整值）、iter 1、common_step_counter 24、
                   critic normalizer 逐一恢复
optimizer state    PPO 51 个 / estimator 39 个 Adam tensor（step / exp_avg / exp_avg_sq）逐值一致
inference parity   固定 obs：action [4,12] / critic / source encoder + latent 均一致；
                   仅给 actor group 也能得到同一 action
shared source enc  load 后 estimator optimizer 参数与 inference 读取的 source encoder 同一对象；
                   estimator step 后 inference 输出同步变化
resume             load 后 learn(2)：首个 logged iter = 1（从恢复值继续），
                   iter 1→2；PPO step 和 680→1360，estimator step 和 533→1053；loss finite
flat/rough         actor [6,45] / critic [259] / estimator_velocity [3]、HIMSpec、
                   model state shapes 完全一致
```

既有回归：`tests/check_black_flat.py --device cpu`（含 `rl_cfg.py` refactor 后的 PPO
config）PASS；`tests/check_black_him.py --device cpu`、`tests/check_black_him_algo.py
--device cpu` 均 PASS。
本单元未注册 task、未做 warm start、未做 flat→rough resume、未实现 exporter、未做长训练。

------

### 17.15 black-flat-him registration + PPO→HIM warm start（PASS）

新增 `src/alldog_mjlab/tasks/velocity/black/him_runner.py`（`BlackHimOnPolicyRunner`）与
`src/alldog_mjlab/algorithms/him/warm_start.py`（`warm_start_from_ppo_actor`），
`HimRslRlOnPolicyRunnerCfg` 新增 `warm_start`，注册 `black-flat-him`。
本地验证工具：`tests/check_black_him_warm_start.py`（不提交 Git）。

结果：

```text
CPU  PASS
CUDA PASS（RTX 4060 Laptop 8 GiB）
CLI  PASS（CPU：PPO / random HIM / warm-start HIM / HIM resume；CUDA：warm-start HIM）
```

关键验证（真实 `black-flat` PPO checkpoint，由短 PPO runner 产生）：

```text
actor mapping     first layer [512,64]：[:, :45]==source，[:, 45:]==0，bias==source；
                  mlp.2/4/6 与 distribution.std_param 逐值一致
parity            synthetic max_abs_error 0.0；real env max_abs_error 0.0；
                  history variation max_abs_error 0.0
critic parity     fixed 259-D critic max_abs_error 0.0（含 normalizer buffer）
fresh state       iteration 0 / PPO + estimator optimizer state 空 / lr 1e-3 /
                  common_step_counter 0
estimator fresh   estimator 13 个 key 迁移前后完全一致；source 无 estimator key
warm update       learn(1)：value / surrogate / entropy / estimation / swap 均 finite，
                  actor / critic / estimator 参数均变化
random init       black-flat-him 不传 warm start 也能短训练
HIM resume        warm-start HIM 保存后 fresh HIM runner strict load + 继续 1 iteration 通过
negative          resume+warm_start / HIM checkpoint 作为 source / input dim / hidden /
                  missing critic 均 fail loudly 并给出具体字段
```

CLI（`uv run train`，`--log-root` 隔离）命令与结果：

```text
black-flat（短 PPO，产生合法 PPO checkpoint）                        exit 0
black-flat-him（随机初始化）                                        exit 0
black-flat-him --agent.warm-start True --agent.load-run '.*_flat$'  exit 0；
    [INFO] PPO→HIM warm start ... actor_input 45→64 (copied 45, zero 19), critic keys 12
black-flat-him --agent.resume True --agent.load-run '<him run>'     exit 0
black-flat-him --agent.warm-start True --agent.resume True          ValueError（互斥）
CUDA：black-flat-him --agent.warm-start True ...                     exit 0
```

既有回归：`tests/check_black_flat.py` / `check_black_rough.py` / `check_black_him.py` /
`check_black_him_algo.py` / `check_black_him_runner.py` 均 `--device cpu` PASS。
本单元未注册 `black-rough-him`、未做 flat→rough resume、未实现 exporter、未做长训练。

------

### 17.16 Unit 5 / stairs 验证记录（2026-10-06）

```text
tests/check_black_rough_him.py --device cpu        PASS
    registry / task contract / command 一致 / schema 一致 / full-resume state parity
    （双 optimizer step/exp_avg/exp_avg_sq 逐 tensor）/ model parity max_abs_error=0.0 /
    rough env identity / iteration 续接（1→3）/ losses finite / 参数变化 /
    terrain curriculum term 执行 / warm-start 显式拒绝
tests/check_black_rough_him.py --device cuda       PASS
tests/check_black_flat.py    cpu + cuda            PASS
tests/check_black_rough.py   cpu + cuda            PASS（7 类 terrain + stairs 数值）
tests/check_black_him.py / check_black_him_algo.py /
tests/check_black_him_runner.py / check_black_him_warm_start.py（cpu）   PASS
CLI（CUDA）: uv run train black-flat-him 2-iter smoke →
             uv run train black-rough-him --agent.resume True
             --agent.load-run '.*_flat_him$' 2-iter smoke        PASS（iter 1→2 续接）
CLI 负例: uv run train black-rough-him --agent.warm-start True →
             ValueError "PPO→HIM warm start currently supported only for
             black-flat-him. ..."                                 PASS
```

注：smoke checkpoint 仅 2 iteration（随机初始化），不能站立 / 行走属预期，
不是配置缺陷（play viewer 观察到的摔倒 / 频繁 reset 已由诊断确认：action 非 0、
qvel 正常、illegal_contact ≈0.65 s 一次）。

------

## 18. Known Risks

当前需要持续注意：

1. MuJoCo / mjwarp contact sensor在深度 penetration 情况下观察过 `found` 存在但 force为0的现象。正常落地/趴地时 force可正常达到明显大于1 N。目前 illegal-contact threshold继续保持1 N，后续根据训练日志判断是否需要处理。由于 trunk 与地面的接触力只在某个高度区间可靠，本地验证的强制触地 probe 会从浅到深扫几个 root 高度取首个触发，而不是固定单一高度。
2. 当前 actor contract已经固定，但 critic仍属于 MjLab baseline，后续 HIM阶段不能将其误认为旧 Black privileged observation。
3. 旧 `algorithms/him/` prototype 的 Black-specific hardcode（`HIMActorCritic(270,238,45,12)`、`HIMRunner`、ONNX dummy 270）已在 §21 单元删除并替换为 spec-driven 实现；生产 HIM 路径不再依赖固定维度。后续 unit 不得重新引入。
4. Legacy Black reward 存在两个版本：当前 `black_config.py` / `black_env.py`（HEAD，含 2026-07-15 的 `1f344d9` 覆盖式同步）与 2026-06-29~07-03 的日志 lineage。Black flat v1 的 reward authority 不再取二者之一：核心公式改用 InternRobotics/HIMLoco 官方 Go1 baseline，机器人数值取 Black intrinsic（见 §11.0）；`68f1c1c` 只作为 optional shaping / 历史调参 / sim2real 诊断参考。`1f344d9` 不作为迁移依据（它同时改动 reward / command / DR / terrain / termination / PPO 且无对应决策记录），这是迁移 source decision，不是对该提交作者意图的事实断言。`super-dog` 的 shaping 项若日后需要启用，须先确认取哪一版。
5. 训练每次 save 都会打印 `[WARN] ONNX export failed (training continues): 'joint_pos'`：`.pt` 与训练均不受影响，但导出的 onnx 缺少部署 metadata。现象 / 分析 / 结论与处理时机见 §18.1，不在当前 Black flat PPO 阶段处理。
6. 最终验证观察到的策略弱点：轻微拖脚（foot dragging）。v1 baseline 有意不含 foot_clearance（§11.3），因此这不是配置错误；若需要抬脚高度约束，须作为新的 behavior unit 提出（§15）。另：最终验证只跑到 500 iteration（用户决定不跑满 10 000），因此长程收敛性未验证（§17.1）。

### 18.1 Policy export / ONNX metadata

现象：

Black flat PPO 训练中每次 save 都会打印

```text
[WARN] ONNX export failed (training continues): 'joint_pos'
```

500-iteration sanity run 中 11/11 次出现（从第一次 `model_0.pt` 起）。

分析：

`mjlab/tasks/velocity/rl/runner.py` 的 `VelocityOnPolicyRunner.save()` 顺序为
`super().save()`（.pt）→ `export_policy_to_onnx()`（.onnx）→ `get_base_metadata()`
→ `attach_metadata_to_onnx()`，失败在第三步：
`mjlab/rl/exporter_utils.py:47` 硬编码

```python
joint_action = env.action_manager.get_term("joint_pos")
```

而 Black 的 action 契约是四个分腿 term（`joint_pos_fl` / `_fr` / `_rl` / `_rr`，见 §3.3），
不存在名为 `joint_pos` 的 term，该行抛 `KeyError('joint_pos')`，被 `except` 兜住只打 WARN。

本地已直接复现（只构建 env，不训练）：

```text
action terms     : ['joint_pos_fl', 'joint_pos_fr', 'joint_pos_rl', 'joint_pos_rr']
action total dim : 12
metadata FAILED  : KeyError('joint_pos')
出错行            : exporter_utils.py, line 47, in get_base_metadata
```

MjLab 官方 velocity 任务使用单个 `"joint_pos"` term（`velocity_env_cfg.py:168`），
因此上游不会遇到该错误；这是 Black 分腿 action term 契约（有意为之）与 MjLab
导出工具约定不一致的结果，不是环境或依赖问题。

影响范围：

```text
训练             不受影响（WARN 之后继续训练，run 能跑到最后一次 save）
checkpoint       不受影响（.pt 在失败步骤之前已经写完）
play / 评估      不受影响（play 只读 .pt，不读 onnx）
.onnx 文件       会生成且结构合法（onnx.checker 通过，inputs=['obs'], outputs=['actions']）
.onnx metadata   metadata_props 为空
```

缺失的字段即 `get_base_metadata()` 本应写入的内容：

```text
joint_names / joint_stiffness / joint_damping / default_joint_pos / action_scale
command_names
observation_names / observation_terms_scale / observation_terms_clip
observation_terms_flatten_history_dim / observation_terms_history_length
```

即：网络权重在，但「策略输出如何映射到真机」的元数据不在。

结论：

- 当前 Black flat PPO 阶段不需要处理；不要因为看到该 WARN 而改动 action term 命名
  （会触碰 §3.3 已冻结的 policy 顺序契约，且 MjLab v1.6.0 的 `preserve_order` 不能可靠
  重排 JOINT action 目标）。
- 该问题归属 Black sim2real / deployment contract 阶段，必须解决并验证，二选一：
  (a) 由 Black / 部署侧提供自己的导出与 metadata（不改 action term 名，也不改 MjLab 源码）；
  (b) 改成单个 `joint_pos` term（需重新验证 action 顺序契约，风险更高）。
- 无论取哪种方案，部署契约都必须显式给出「policy 顺序 FL→FR→RL→RR ↔ MuJoCo 模型顺序」
  的映射（见 §3.1），因此这不是额外负担，而是部署阶段本来就需要的产物。
- 验收条件：导出路径不再出现该 WARN，或明确采用「onnx + 独立映射表」方案并证明能完整
  还原 joint 顺序 / action_scale / default pose。
- 部署侧现状（见 §19.4）：两个目标运行时都用 TorchScript + 显式配置，因此该 metadata 问题
  **不是**当前部署的主要阻塞项；部署阶段同样不要为了它重命名 action term。

------

## 19. Frozen Black Deployment / Sim2Sim Contract

阶段状态：

```text
Black flat PPO baseline:               COMPLETE
Black rough PPO baseline:              COMPLETE
Black rough MJWarp runtime workaround: COMPLETE
Black PPO deployment contract / sim2sim compatibility: COMPLETE
next decision: Black real robot backend / sim2real contract
```

部署参考实现：

```text
legacy:                 N-W-wolf/rl_sar-black-W
                        本地：/home/windnotebook/PROJECT/Dog/rl_sar
                        （remote: coverMoon/rl_sar-for-super-dog）
主要未来运行环境:        N-W-wolf/quadruped_control
                        本地：/home/windnotebook/PROJECT/Dog/quadruped_control
```

历史记录修正：本文件与 `AGENTS.md` §3.1 曾把 legacy 本地路径写成
`/home/windnotebook/PROJECT/Dog/real_robot`。该路径实际是 `coverMoon/real_robot`（ROS 2
底层通信 / real_runner / serial / robot_description），**不含** `rl_sar`、policy config
或 TorchScript runtime；真正的 `rl_sar` checkout 在
`/home/windnotebook/PROJECT/Dog/rl_sar`。`AGENTS.md` 的同一处错误已修正：
§3.1 为 legacy RL deployment runtime（`rl_sar`），§3.2 为 legacy real-robot
通信 / 硬件层（`real_robot`），原 §3.2 顺延为 §3.3。

`quadruped_control` 当前状态：simulation-side backend 可用；real robot backend 尚未实现。

### 19.1 当前 Black PPO policy / deployment safety contract

policy / deployment joint order：`FL -> FR -> RL -> RR`，每腿 `hip -> thigh -> calf`
（冻结于 §3.1）。

actor observation：**45 维单帧**（冻结于 §5）：

```text
[0:3]   command
[3:6]   base angular velocity
[6:9]   projected gravity
[9:21]  joint position relative to default
[21:33] joint velocity
[33:45] previous action
```

observation scale：

```text
command             [2.0, 2.0, 0.25]
angular velocity    0.25
projected gravity   1.0
joint position      1.0
joint velocity      0.05
previous action     1.0
```

actor normalization：**disabled**（§5.3）。

Policy action：12 维 position residual。训练与部署必须在 safety processing **之前**
得到相同的 `q_policy`：

```text
raw_action[12]
q_policy = q_default + 0.25 * raw_action
```

Deployment safety 是随后独立的一层，不属于 exported policy semantics，也不要求
MjLab action manager 复刻。用户已确认下表位置限经过真实硬件验证，因此
`quadruped_control` 在发送底层命令前有意执行：

```text
q_command = clamp(q_policy, q_hw_min, q_hw_max)

             hip          thigh          calf
FL / RL      [-0.5, 0.5]  [-1.2, 1.6]   [-2.5, -0.85]
FR / RR      [-0.5, 0.5]  [-1.6, 1.2]   [ 0.85, 2.5]
```

该硬件位置裁剪属于 deployment runtime responsibility。`rl_sar` 当前没有等价的
显式 policy-stage hardware position clamp；旧 `real_runner` 也未找到等价的显式
hardware joint-position clamp（实机链路分析见 §19.13，固件保护仍待确认）。
跨 runtime 要求 `q_policy` 一致，并验证 safety transform 明确且正确；
**不要求最终 `q_command` 三边相同**（§19.12）。

Black default pose：

```text
FL: [0.0,  0.8014, -1.527]
FR: [0.0, -0.8014,  1.527]
RL: [0.0,  0.8014, -1.527]
RR: [0.0, -0.8014,  1.527]
```

RL PD：`Kp = 40` / `Kd = 1.2`（§3.2）。policy 周期：`0.02 s` = 50 Hz。

注意：当前 actor 输入是 **45 维单帧**，**不是**未来 HIM 的 270 维 history 输入。

### 19.2 Deployment compatibility findings

legacy `rl_sar` 与 `quadruped_control` 使用同一套 Black policy 语义：

```text
FL FR RL RR policy joint order
45 维单帧 observation 定义
command / angular velocity / gravity / joint position / joint velocity / previous action 顺序
一致的 observation scales
action scale 0.25
一致的 Black default pose
Kp 40 / Kd 1.2
50 Hz policy rate
```

历史 HIM 部署配置使用 `45 维 x 6 帧 history = 270 维` 输入，**这不是当前 PPO contract**。
因此当前 PPO 部署配置必须使用单帧 45 维；270 维 history contract 属于未来的 HIM 部署，
届时单独恢复。

### 19.3 Model-format gap → actor-only TorchScript 导出（COMPLETE）

两个部署运行时当前都通过 `torch::jit::load()` 加载 **TorchScript** 模型；而 MjLab 训练
checkpoint（如 `model_498.pt`）是 RSL-RL 训练 checkpoint，**不能**直接作为可部署 TorchScript
模块。

导出路径已实现并验证：

```text
实现              src/alldog_mjlab/utils/export_policy.py
checkpoint 加载   MjLab runner 的 load(..., load_cfg={"actor": True}, strict=True)
导出              RSL-RL 5.4.2 原生 runner.export_policy_to_jit()
                  （PPO：MLPModel.as_jit；HIM：HIMPolicy.as_jit，见下）
环境              task registry 的 play cfg（num_envs = 1），维度取自 env / HIMSpec 而非 checkpoint
```

当前标准入口：`uv run export --task-id <task>`（PPO / HIM task 均支持）。由
`load_rl_cfg(task_id)` 取得 `experiment_name`，在 `logs/rsl_rl/<experiment_name>`
下直接使用 MjLab v1.6 `get_checkpoint_path()` 按 task runner 的 stage 默认
`load_run`（flat 为 `.*_flat$`，rough 为 `.*_rough$`，HIM 为 `.*_flat_him$` /
`.*_rough_him$`）选择最新匹配 run 与 checkpoint（默认 `model_.*.pt`），输出到
`<run>/exported/policy.pt`。可用 `--load-run`、`--checkpoint` 指定名称或正则，
用 `--output-dir` 改输出目录；文件名始终为 `policy.pt`，设备默认 CPU。
Black flat / rough 当前共用 `black_velocity` 日志目录，
因此在共同 root 中分别按 stage 选择；历史无后缀 run 仅通过显式 `--load-run` 访问。
无对应 stage run 时明确报无匹配，不回退到 legacy run。

导出模块的 deployment contract（冻结）：

```text
PPO  input  float32 [1, 45]   actor 单帧 observation
     output float32 [1, 12]   policy action

HIM  input  float32 [1, 270]  canonical history（frame-major，newest → oldest，
                              与任务侧 him.canonical_history() 同语义，§20.1）
     output float32 [1, 12]   policy action
```

`_HimActorJit`（algorithms/him/policy.py）为 HIMPolicy 的脚本化包装：输入
canonical [B, 270]，current frame 取前 45 列（canonical 中最新帧在最前），source
encoder 直接消费全部 270 列，输出 velocity + L2 normalized latent，与 current frame 拼接后经 actor MLP →
deterministic action；与推理路径 `get_latent()` 数值
等价（3 组 probe max_abs_diff = 0）。history / canonical 维度由 HIMSpec 运行时
读取（不硬编码 6 / 270）。HIM ONNX 导出未实现（当前部署只消费 TorchScript）。

验收已满足：同一 observation 下与 checkpoint actor 的 deterministic forward 在
atol 1e-6 / rtol 1e-5 内一致（实测三组 probe 的 max abs diff 均为 0.0，见 §17.3）。
导出侧不做任何额外 normalization / scaling（actor normalization 为 disabled，§5.3）。

sim2sim 侧 observation、actor 与 pre-safety `q_policy` 已验证（§17.10 / §19.12）；
部署位置安全层单独判定，不并入训练 action transform。

### 19.4 ONNX metadata（非当前部署阻塞项）

MjLab v1.6 velocity exporter 的 metadata 假设只有一个 action term `joint_pos`，而 Black 使用
四个显式 term（`joint_pos_fl` / `joint_pos_fr` / `joint_pos_rl` / `joint_pos_rr`），因此
metadata 导出会报已知的 `joint_pos` 查找告警（见 §18.1）。

两个目标部署运行时都用 **TorchScript + 显式配置**，所以 ONNX metadata **不是**当前部署的
主要阻塞项。**不要**为了迁就 stock exporter 而重命名 Black action term（会触碰 §3.3 冻结契约）。

### 19.5 Intended deployment migration order

```text
1. Freeze Black PPO deployment contract.                              （完成，§19.1）
2. Export actor-only TorchScript from an existing MjLab checkpoint.   （完成，§19.3）
3. Verify exported policy numerically against MjLab actor.            （完成，§17.3）
4. Add a 45-D PPO deployment config for legacy rl_sar.                （完成，§19.6）
5. Run legacy rl_sar sim2sim.                                         （完成，§17.5 / §17.6 /
                                                                       §19.7 / §19.8）
6. Add the corresponding 45-D PPO config for quadruped_control.       （完成，§19.9 / §17.7）
7. Run quadruped_control MuJoCo sim2sim.                             （完成，§19.11 / §17.9）
8. Compare training/deployment observation, actor, pre-safety q_policy, and safety.
                                                                      （完成，§17.10 / §19.12）
9. Decide the real robot backend / sim2real contract.                 （next decision，未开始）
```

本阶段**不要**开始 HIM integration。

### 19.6 legacy rl_sar Black PPO 45-D deployment config（COMPLETE）

```text
仓库              /home/windnotebook/PROJECT/Dog/rl_sar（coverMoon/rl_sar-for-super-dog）
config 目录       src/rl_sar/policy/black/mjlab_ppo/
顶层 key          black/mjlab_ppo
模型文件          black_ppo_actor.pt（从 §19.3 导出产物直接复制，未重新导出）
md5               e7318ea9019cf27f2d204565659cf0cd（与训练侧导出文件逐字节一致）
启动方式          ./run_rl_sim_debug.sh black mjlab_ppo
```

观测是 **45 维单帧**（`observations_history: []`，不是 HIM 的 `[0..5]` / 270 维）：

```text
commands(3) + ang_vel(3, body-frame) + gravity_vec(3) + dof_pos(12) + dof_vel(12) + actions(12)
```

部署侧数值：`action_scale = 0.25`、`rl_kp = 40` / `rl_kd = 1.2`、
`torque_limits = 33.5`、`default_dof_pos = [0.0, 0.8014, -1.527, ...]`、
`joint_mapping = [3,4,5, 0,1,2, 9,10,11, 6,7,8]`（外部顺序 → policy 顺序，保留不变）。

`policy_dof_indices = [0..11]` 是显式 identity：`ReadYamlRL()` 校验其为 12 个合法下标，
`SelectDofColumns()` / `ComputeOutput()` 在 identity 下与“未配置”逐元素等价
（`wheel_indices = []`，mask 分支不执行；`num_of_dofs = 12`，action_dim 不变）。

已验证：`bash build.sh rl_sar` PASS（shebang 不在第 1 行，需用 `bash build.sh`）；
`rl_sim` + `ReadYamlRL("black/mjlab_ppo")` + `torch::jit::load()` PASS——进入
`RLFSMStateRL_Locomotion` 后持续保持（无 `InitRL() failed`），`runtime_status` 的
`model_name` 为 `black_ppo_actor.pt`（见 §17.4）。runtime 链接的 libtorch 为 2.0.1+cpu，
可加载 torch 2.14 导出的 TorchScript 且输出与训练侧逐位一致。

已知差异（属部署侧，不在本单元解决）：

```text
clip_obs = 100              legacy runtime 强制 clamp，训练侧无此 clip（observation 未触发）
command range               base.yaml command_limits = [3.0, 1.0, 3.0]，训练命令范围 vx/vy ∈ [-1,1]、wz ∈ [-pi,pi]
torque semantics            rl_sar ComputeOutput 计算张量裁剪 33.5 N·m vs 训练 actuator effort_limit 20 N·m
action clip                 有意不配置 clip_actions_lower/upper → Forward() 直接返回 actor action
policy_switch.yaml          mjlab_ppo 未加入 policy_config_cycle，只能显式指定 config name 启动
```

### 19.7 sim2sim observation / action contract（COMPLETE）

数据链、reference 方法与逐步数值结果见 §17.5。已有的有效结论：

```text
joint order      policy = FL, FR, RL, RR（每腿 hip/thigh/calf），与 §3.1 / §5 一致
joint mapping    MuJoCo physical (FL,FR,RL,RR) --mujoco_runner.motor_mapping--> ROS msg
                 (FR,FL,RR,RL) --rl_sar.joint_mapping--> policy (FL,FR,RL,RR)
                 两者是同一个数组、方向相反，且该置换是对合（P∘P = I），
                 因此复合后 policy 顺序 == physical 顺序；名字与数值双重验证
gravity          wxyz quaternion + quat_apply_inverse(quat, [0,0,-1])（两侧形式等价）
ang_vel frame    body-frame（backend `<gyro site="imu">`，imu site 在 trunk 原点）
action           q_target = default_dof_pos + 0.25 * raw_action，无额外 action clip
previous_action  obs 使用上一 policy step 的 action a_(t-1)
clip_obs         100，在本轮 trace 中从未触发
```

操作注意（非 contract）：`rl_sim` 的键盘 command 会被 `RunModel()` 的 joy_timeout 清零
（无 `/joy` 时），sim2sim 需要 navigation mode + `/cmd_vel` 才能给出非零 command。

MuJoCo backend 实际位置：`/home/windnotebook/PROJECT/Dog/real_robot/black/src/mujoco_runner`
（`rl_sar/README.md` 里的 `~/PROJECT/RoboCon/Dog/black_mujoco` 已过期）。

### 19.8 rl_sar MuJoCo locomotion rollout 行为（COMPLETE）

已验证的 command 表与扭矩统计见 §17.6。当前有效结论：

```text
站立          zero command 稳定（4 个 session、每个 19 s，无漂移、无振荡、无饱和）
前进 / 后退   |v| >= 0.6 可跟踪（0.548 / -0.548）；|v| <= 0.3 基本不产生运动
横向          ±0.3 部分（48~59%）、±0.6 可跟踪（84~92%），横向与 yaw 存在耦合
组合          (0.5, ±0.2, ±0.5) 均可跟踪（线速度 66~101%，yaw 66~80%）
yaw           纯 yaw（零线速度）不产生旋转；叠加任一非零线速度后 yaw 恢复（0.13~0.20 rad/s）
稳定性        无摔倒 / 无 NaN / 无控制发散；roll ≤ 6.4°、pitch ≤ 5.6°、z ∈ [0.441, 0.485]
```

（L1）standing fixed point：低速或零线速度下策略会落入一个静止平衡点（action std < 0.03、
q std < 0.002 rad、|tau| < 6），此时即使 cmd 非零也不产生运动。已确认这不是 runtime /
contract 失败（数据链 §19.7 已 PASS、FSM 仍在 Locomotion、无 torque saturation），
而是策略在部署 sim 中的行为：推测其步态需要非零线速度激发（训练采样器含 10% standing）。
因此本轮把「纯 yaw 不旋转」归类为 policy behavior limitation，**不是** contract failure。

当前已知 framework difference（本轮实测，未修正）：

```text
关节范围      训练 MJCF：hip ±0.5 / thigh [-1.2,1.6] / calf [-2.5,-0.85] 或 [0.85,2.5]（hard limit）
              部署 MJCF：全部 ±10 rad ⇒ 更宽松；实测 calf 实际角度超出训练范围 0.09~0.13 rad
torque        训练 actuator effort_limit 20 / rl_sar ComputeOutput 计算张量裁剪 33.5 / 旧部署 MuJoCo actuatorfrcrange ±20
              实测 rl_sar 计算值最大 27.79（从未达 33.5），>20 占 2.2% 帧 ⇒ 非持续性瓶颈
质量 / 惯量    总质量 13.0025 kg（部署）vs 13.2472 kg（训练），trunk 惯量差 3~6%
足底摩擦       两侧一致（sliding 1.0）；训练侧仅在 DR 中随机化
PD / dt        一致（Kp 40 / Kd 1.2、policy 50 Hz、low-level & sim 200 Hz）
执行链         部署侧多出 3 ms motor delay 与 encoder delay buffer（backend 行为）
command 通道   rl_sar 键盘 command 会被 joy_timeout 清零，sim2sim 需 navigation mode + /cmd_vel
odom twist     backend 的 /odom twist 用 mj_objectVelocity(flg_local=1)，实测是 body **inertial**
              frame（body_iquat）而非 base frame；本轮线速度改用 pose 差分，未依赖该字段
```

### 19.9 quadruped_control Black PPO 45-D config（COMPLETE）

```text
config            configs/policies/black/mjlab_ppo.yaml（新增，不覆盖 HIM-era flat.yaml）
模型              assets/policies/black/mjlab_ppo/black_ppo_actor.pt
                  md5 e7318ea9019cf27f2d204565659cf0cd（来自 §19.3 导出产物，未重新导出）
observation       45 维单帧：history_frames [0] ⇒ inference_input_dimension 45（不是 270）
observation_order commands / angular_velocity / projected_gravity /
                  joint_position_error / joint_velocity / previous_action（loader 硬校验）
关节              RobotModel 顺序 FL/FR/RL/RR hip-thigh-calf，policy_dof_indices 恒等
scales            command [2,2,0.25] / ang_vel 0.25 / q_rel 1.0 / dq 0.05 / gravity & last_action 1.0
default pose      [0.0, 0.8014, -1.527, -0.0, -0.8014, 1.527, ...]（不是 controller stand pose）
action            q_target = default + 0.25 * action；action_clip 100 为 schema 必填项，
                  实测 |a|max ≈ 1.9~3.2，属不会触发的 legacy safety guard
PD                policy kp 40 / kd 1.2；controller/FSM stand 仍为 80 / 3（职责分离，未混用）
启动              默认 policy_switch.yaml 未加入 mjlab_ppo，仍为 flat/obstacle；
                  需要显式加载时用 --policy-switch-config 指向包含 mjlab_ppo 的 switch 配置
```

torque/effort 配置差异（本单元按指示只报告，未改架构；实机语义见 §19.15）：

```text
RlConfig 没有 torque_limits 字段；当前 MuJoCo backend 使用 RobotModel.joints[].limits.max_effort：
hip/thigh 23.7 N·m、calf 59.25 N·m（近期提交 6bb7f1d 同步自 blackW 配置），
与 MuJoCo actuator ctrlrange 取交集。这是配置的 joint-side 仿真输出上界，
不是已验证的真实硬件力矩上限，也不等同 rl_sar 的 ComputeOutput 张量裁剪。
⇒ rl_sar 的 33.5 N·m 计算张量裁剪在当前 schema 下**无法按 policy 表达**，
  本单元未扩展架构、未添加会被静默忽略的 torque_limits 字段。
```

### 19.10 quadruped_control observation / action / target / torque contract（COMPLETE）

trace 方法与逐步数值见 §17.8。当前有效结论：

```text
observation       45 维单帧，与独立参考 max 3.2e-8；history [0] 确实只取当前帧
joint order       RobotModel FL/FR/RL/RR hip-thigh-calf；MuJoCo 模型原生顺序是
                  FL,FR,RR,RL，backend 按名字映射，policy 侧无感知
ang_vel frame     body frame（StateFrame.imu.angular_velocity 直接进 obs×0.25）
gravity           wxyz 四元数 + 与训练侧等价的投影重力公式（实测 3.0e-8）
previous_action   a_(t-1)（750 次连续转移无偏差）
action            q_policy = default + 0.25 * action；
                  q_command = clamp(q_policy, 经真实硬件验证的关节位置限)
                  无额外 action clip（clip=100 不触发）；max_position_jump=1.0 未触发
torque            tau_raw = kp*(q_command-q) + kd*(dq_target-dq) + ff
                  → clamp 到 [max(ctrl_min,-max_effort), min(ctrl_max,max_effort)]
                  → MuJoCo 无额外 clamp（actuator_force == qfrc_actuator == data.ctrl）
                  实测 RL 阶段 |tau_raw| ≤ 16.6 N·m，从未触到 23.7/59.25/33.5/20
关节范围          training / RobotModel / MuJoCo 三者一致（hip ±0.5、thigh [-1.2,1.6] 或
                  [-1.6,1.2]、calf [-2.5,-0.85] 或 [0.85,2.5]）
```

当前 torque 语义与另外两个 runtime 的差异（仅记录，未决定）：

```text
training            20 N·m 全场统一（mjlab effort_limit）
rl_sar              ComputeOutput 张量裁剪 33.5 N·m；旧 MuJoCo actuator 另限 ±20
quadruped_control   无 per-policy torque clamp；MuJoCo backend 按 RobotModel 配置的 joint-side max_effort
                    23.7（hip/thigh）/ 59.25（calf）与 actuator ctrlrange 取交集
```

------

### 19.11 quadruped_control MuJoCo locomotion rollout 行为（COMPLETE）

trace 方法与数值见 §17.9。flat 场景、body 系命令、稳态窗口（每段末 8 s）实测：

```text
命令                    实测 (vx, vy, wz)             误差
(0.0, 0.0, 0.0)         (0.000, 0.000, 0.002)         ~0              稳定站立
(+0.2, 0, 0)            (+0.054, +0.006, -0.005)      +0.146
(+0.3, 0, 0)            (+0.187, +0.012, +0.019)      +0.113
(+0.6, 0, 0)            (+0.530, +0.028, +0.063)      +0.070
(+1.0, 0, 0)            (+0.775, +0.027, +0.108)      +0.225
(-0.2, 0, 0)            (+0.000, -0.000, -0.000)      -0.200          静止（固定点）
(-0.3, 0, 0)            (+0.000, -0.000, +0.001)      -0.300          静止（固定点）
(-0.6, 0, 0)            (-0.468, +0.029, -0.132)      -0.132
(0, +0.3, 0)            (+0.018, +0.174, +0.145)      +0.126
(0, -0.3, 0)            (+0.004, -0.143, +0.016)      -0.157
(0, +0.6, 0)            (+0.010, +0.547, +0.351)      +0.053
(0, -0.6, 0)            (+0.020, -0.509, -0.053)      -0.091
(0, 0, ±0.5 / ±1.0)     实测 yaw rate ≤ 0.03 rad/s     ±0.48 / ±0.99   不旋转（固定点）
(+0.5, +0.2, +0.5)      (+0.423, +0.203, +0.414)      (+0.077, -0.003, +0.086)
(+0.5, -0.2, -0.5)      (+0.507, -0.139, -0.317)      (-0.007, -0.061, -0.183)
```

```text
L1 低速 / 静止固定点（policy 行为，与 rl_sar 结论一致）
   |v| ≤ 0.3 m/s 的后退命令完全不产生位移；纯 yaw 命令（±0.5 / ±1.0 rad/s）不旋转。
   只要存在非零线速度分量，yaw 即正常（F1 / F2 实测 wz 0.414 / -0.317）。
   该状态下 raw action std 0.011~0.029、q std ≈ 0.004 rad、|tau| ≤ 5.94 N·m，
   是稳定静止点而非发散。判定：训练侧策略固有行为，不是 deployment contract 冲突。

L2 跟踪精度（工程级可接受）
   前进低速段欠速（+0.2 → 0.054、+0.3 → 0.187），+1.0 实测 0.775；
   横向 ±0.3 欠速约 50%，±0.6 基本跟得上；后退 -0.6 实测 -0.468。
   方向（符号 / 轴）从未出错，未出现 rl_sar 那次明显的侧向漂移放大。
```

```text
torque 语义（本 runtime）   无 per-policy torque clamp；MuJoCo 配置上界 23.7（hip/thigh）/ 59.25（calf）；
                            本轮 RL 阶段 max 20.93 N·m，0.00% 触发截断
硬件关节位置裁剪            与训练 MJCF 数值一致，但这是独立的 deployment safety layer；
                            正常 rollout 中已低频触发（> 0.5 rad 的样本为 0，并不代表未触发），
                            完整跨 runtime 统计见 §17.10 / §19.12
```

结论：**PASS**（engineering-level）。L1 属策略行为、L2 属跟踪精度，均不构成本阶段阻塞。

------

### 19.12 跨 runtime policy / deployment safety contract（COMPLETE）

验收分层（数值见 §17.10）：

```text
Layer 1  policy I/O
         raw state → observation[45] → deterministic actor[12]
         → q_policy = q_default + 0.25 * raw_action
         三边同一输入下必须一致；不修改训练 action、default pose 或 joint order。

Layer 2  deployment safety
         quadruped_control: q_command = clamp(q_policy, hardware-validated joint limits)
         rl_sar:            q_command = q_policy（当前无等价的显式 policy-stage 位置裁剪）
         MjLab:             不定义部署侧最终 q_command；action manager 无需复刻硬件裁剪。
```

Cross-runtime acceptance：observation 一致、actor 输出一致、pre-safety `q_policy`
一致、部署安全变换明确且计算正确，并且不静默改变 policy observation / action 定义。
`quadruped_control` 的硬件位置限见 §19.1。正常 policy 分布的 1000 帧中，25 帧
（全部 `FL_calf_joint`）触发位置裁剪，最大 `0.03850 rad`；这是**active but
low-frequency deployment safety layer**。`max_position_jump = 1.0 rad` 在同批
rollout 中未触发。最终低层 `q_command` 三边有意不同，不能声称全部低层命令相同。

```text
Observation contract:                    PASS
Actor inference:                        PASS
Policy action / pre-safety q_policy:     PASS
quadruped_control hardware position safety: PASS
Final commanded target across runtimes: INTENTIONALLY DIFFERENT
Overall Black PPO sim2sim compatibility: PASS
```

部署里程碑：`rl_sar` config / trace / locomotion rollout PASS；
`quadruped_control` config / observation-action-torque trace / locomotion rollout PASS；
cross-runtime observation / actor / pre-safety `q_policy` PASS。实机链路静态检查见
§19.13；位置安全已归属 deployment layer，effort/torque 语义见 §19.15：
training actuator `20 N·m`；`rl_sar` ComputeOutput 张量裁剪 `33.5 N·m`，
旧 MuJoCo actuator `±20 N·m`；`quadruped_control` MuJoCo 的 RobotModel
joint-side max_effort 为 hip/thigh `23.7 N·m`、calf `59.25 N·m`。
真实硬件最终 effort/current ceiling 仍未确认。
`quadruped_control` 单次推理超过 20 ms 会触发 watchdog；目前
`OMP_NUM_THREADS=1` 避免了已观察到的长尾失败，但不是最终实时方案。

motor mapping / calibration 静态契约见 §19.14；当前下一决策见 §1。

### 19.13 Black sim2real / real-backend preflight（分析完成，backend 未实现）

四源静态核对：`alldog_mjlab` 定义 45-D 单帧 observation、12-D FL→FR→RL→RR action、
`q_policy = q_default + 0.25 * raw_action`、50 Hz policy、Kp 40/Kd 1.2 与训练侧
20 N·m effort limit。`rl_sar` 的实机 ROS 命令把 policy index 映射到 message index
`[3,4,5,0,1,2,9,10,11,6,7,8]`；`real_robot` 按 message index `3*leg+j`
送至 `/dev/leg_{leg}` 的 motor ID `3*leg+j`。物理端口到实际腿的接线未由源码证明，
接线及逐关节方向必须在接实机前确认。`quadruped_control` RobotModel 顺序与 policy
相同；其 real RobotIO 尚未实现。

旧实机反馈/命令换算使用 hip/thigh 6.33、calf 15.825 传动比；calf 反号。
以 `real_robot_motor_index i`、`s=+1`（hip/thigh）或 `-1`（calf）、`G` 为传动比、
`O_i=off_set_i-straight_i+calf_correction_i`（`off_set_i=-round((raw_start_i-creep_i)/2π)*2π`，
calf correction 在 `real_robot_motor_index` 2/8（软件映射 FR/RR calf）为
`-46.66°*15.825`、`real_robot_motor_index` 5/11（FL/RL calf）为
`+46.66°*15.825`）记：
`q_joint=s*(motor.Pos+O_i)/G`；反向 `motor.Pos=s*G*q_command-O_i`；
`motor.W=s*G*dq`、`motor.T=s*tau_ff/G`、`motor.Kp=joint.Kp/G²`、
`motor.Kd=joint.Kd/G²`。标定文件缺失时旧代码会使用零数组继续初始化，不能视为
已验证的安全行为。旧 Black RL FSM 实际发送 position/dq/Kp/Kd 与 **tau=0**，
而 `rl_sar` `ComputeOutput` 算出的 ±33.5 N·m torque 只入诊断队列；
它不是已证明的实机输出限幅。电机 mode=1 FOC 闭环，最终 joint effort 由电机侧
PD/固件决定；旧实机 hip/thigh/calf 实际最大输出值和固件保护范围未查明。

旧 `real_runner` 没有找到显式 hardware joint-position clamp 或收到命令后的过期处理；
其 30° 倾角、0.5 s IMU 超时及 8π motor-position jump 的安全 latch 在当前代码中因
`ENABLE_LATCHED_SAFETY_PROTECTION=false` 不生效（8π 检查仍会退出本次命令更新），
serial 失败后的 latch 也被注释。
这不改变用户确认的硬件位置限：`quadruped_control` 的 position clamp 是必须保留的
deployment safety layer。旧 IMU 是 AB5465，按 `(x,-y,-z)` 转换 gyro/accel，
默认用 VQF 6D 的软件变换后 sensor→重力对齐 frame 姿态 quaternion（wxyz）供 policy；
用户已确认当前实体 Black 沿用该已验证的 IMU 安装与配置，见 §19.16。
`rl_sar` joystick 0.3 s
超时归零，navigation `/cmd_vel` 路径未见等价
超时。新框架命令有 10 ms 有效期，控制周期 5 ms、policy 每四周期一次；
单次推理 >20 ms 会失败并转 Passive（Disabled）。`OMP_NUM_THREADS=1` 在前次
rollout 消除了观察到的 >20 ms 长尾，但并非实机实时性证明。

**接 real backend 前的 blocking decisions：**物理 bus/motor/zero/sign 实测和标定
失败策略；real RobotIO 的 position/effort safety ownership 与电机 PD/firmware
实际 torque 上限（不能把 20、33.5、23.7/59.25 直接等同）；
freshness/watchdog 与安全模式的接口语义见 §19.17，数值和固件行为待 bring-up。
旧 ROS topic/SerialPack 线程结构、MuJoCo 2 ms 与训练
5 ms physics dt、policy default 与 stand pose 的小差异，不要求复制为新架构。

motor mapping / calibration 静态契约已在 §19.14 冻结；实体接线仍待确认。
本节仅完成分析，未实现或验证实机 backend，也未启动 HIM/BlackW。

### 19.14 Black real backend v1 motor mapping / calibration contract（静态冻结）

canonical `policy_index = RobotModel_joint_index`：FL→FR→RL→RR，每腿 hip→thigh→calf。
旧实机软件链（依 policy index 0–11 排列）：

```text
rl_sar_message_index = real_robot_motor_index:
    [3,4,5, 0,1,2, 9,10,11, 6,7,8]
real_robot_leg_index = real_robot_motor_index // 3
leg_local_joint_index = real_robot_motor_index % 3
device = /dev/leg_{real_robot_leg_index}
SDK request motor ID = real_robot_motor_index（实际接线/电机身份未由源码证明）
```

`real_robot` hip/thigh `G=6.33, s=+1`，四个 calf `G=15.825, s=-1`；
feedback 和 command 两个方向均已由 `serial_packages.hpp` 核对。
旧标定文件 `./src/real_robot/real_runner/motor_calibration.conf`（相对运行目录）按
`real_robot_motor_index` 保存两行、各 12 个 rotor-side rad：第一行 `straight_position_`，
第二行 `creep_position_`；读取时只顺序提取 24 个数，不检查行边界。
首次成功读取电机反馈后，以该电机上电转子位置
`raw_start_i` 计算 `off_set_i=-std::round((raw_start_i-creep_i)/(2π))*2π`。
令 `O_i=off_set_i-straight_i+calf_correction_i`，`calf_correction_i` 为 motor-side rad：
FL/RL calf `+46.66°*15.825`（对应 `real_robot_motor_index` 5/11），
FR/RR calf `-46.66°*15.825`（对应 `real_robot_motor_index` 2/8）；其机械原因未由源码证明。
hip/thigh correction 为零。新 backend 的 calibration 必须按显式 joint name 绑定；
旧数组文件只能经核对 `real_robot_motor_index`→joint name 后导入。

```text
feedback: q_joint=s*(motor.Pos+O_i)/G; dq_joint=s*motor.W/G
command:  motor.Pos=s*G*q_command-O_i; motor.W=s*G*dq_command
          motor.K_P=Kp_joint/G²; motor.K_W=Kd_joint/G²
          motor.T=s*tau_ff_joint/G
```

`motor.Pos` / `motor.W` 为 rotor-side rad / rad/s，canonical joint q/dq 为 rad / rad/s。
增益和前馈公式对应理想、无损传动下的 joint-space MIT impedance；只冻结坐标换算，
**不**据此冻结固件内部控制或最终 torque limit。对 12 关节的限位端点及随机合法状态，
独立 `/tmp/black_motor_mapping_contract.py` 以多组整数圈数验证了 q/dq 数学 round trip：
最大误差 `6.66e-16 / 8.88e-16`；模拟 float32 电机字段后约 `3e-7`。
SDK 二进制打包/固件及实体电机精度未验证。硬件位置限沿用 RobotModel 的已验证值；
各 motor 端点须用**当次有效** `O_i` 计算 `s*G*q_min-O_i`、`s*G*q_max-O_i`，
calf 的 motor 顺序可能与 joint q 顺序相反，不可把 q_min 端点当作 motor_min。

旧 `motor_zero` 构造时忽略 `load_calibration_file()` 返回值：缺文件继续使用零数组，
字段不足可能留下部分已读值；文件无版本/关节名，未验证 finite，也不拒绝多余字段。
**v1 决策：**缺失、无效、关节数量错误、非有限值、版本不支持或身份不匹配时
calibration **FAIL CLOSED**：RobotIO/hardware enable 失败，RL 不得进入；禁止零偏移
回退。RealRobotIO 对 MotionRuntime 只暴露 canonical RobotModel 顺序、校准后的 q/dq，
并接收同序 joint-space CommandFrame；gear、sign、offset、bus、motor ID 和 SDK
packet 转换均由 backend 负责，不进入 policy/RL 层。

**HARDWARE CONFIRMATION TODO：**逐一确认 `/dev/leg_0`～`/dev/leg_3` 的实体腿，
每端口三个实际 motor ID 与 hip/thigh/calf 的对应、每个电机正转对应的 canonical
q 正方向、该实体机器人的标定值与校准身份。源码只能证明软件请求的 index/ID，
不能证明物理接线。本节未实现 real backend；torque limit、IMU、通信时序、watchdog
和 E-stop 留在后续单元。

### 19.15 Black real backend v1 actuator / PD / effort contract（静态冻结）

**训练物理假设，不属于 exported policy I/O：**Black 的 12 个 `IdealPdActuatorCfg`
均为 joint-space `Kp=40`、`Kd=1.2`、`effort_limit=20 N·m`。
项目锁定的 MjLab v1.6.0 `JointPositionAction` 只写 position target；未设置
velocity/effort action target（清状态时为零）。`pd_actuator.py` 计算
`tau_raw=40*(q_target-q)+1.2*(0-dq)+tau_ff`，再逐关节裁剪到 `[-20,20]`；
`utils/spec.py:create_motor_actuator` 同时把 MuJoCo motor 的 control/force range
设为 `[-20,20]`。20 是训练 actuator 的计算输出/仿真执行器上限，
不是导出 actor 的 action 维度、比例或真实电机的已验证限流值。

**旧部署：**`rl_sar` Black PPO 配置的 33.5 N·m 只裁剪 `ComputeOutput` 的
`rl_kp*(q_target-q)-rl_kd*dq` 计算张量。该张量进入诊断队列/可选 CSV；Black
`fsm.hpp` 的 RL 状态从 position/velocity 队列发送 joint-space `q_target`、
`dq_target=0`、`Kp=40`、`Kd=1.2`、`tau=0`，没有把该 33.5 裁剪值送到
`motor_command.tau`。`TorqueProtect` 调用被注释。因此 **33.5 不是已验证的
真实硬件力矩限幅**；旧 MuJoCo 仿真另有 ±20 N·m actuator 限制。

**旧实机命令：**`real_robot` 的 `SerialPack` 将上述 joint-space 命令转为
`mode=1`（FOC 闭环）的 `motor.Pos/W/K_P/K_W/T`，按 §19.14 的 `G/s/O_i`：

```text
motor.Pos = s*G*q_command-O_i       motor.W = s*G*dq_command
motor.K_P = Kp_joint/G²             motor.K_W = Kd_joint/G²
motor.T   = s*tau_ff_joint/G
```

Black RL 中 `dq_command=0`、`tau_ff_joint=0`，仍有 position/velocity error
引起的电机侧阻抗力矩；`motor.T=0` **不代表实际输出力矩为零**。理想无损传动下，
电机侧 PD 输出乘 `s*G` 正好还原
`Kp_joint*(q_command-q)+Kd_joint*(dq_command-dq)+tau_ff_joint`。
独立 `/tmp/black_pd_coordinate_check.py` 对 `G=6.33,s=+1` 与
`G=15.825,s=-1` 各 1000 组随机目标、状态、偏移、前馈检验，最大误差
`8.53e-14 N·m`。这是坐标公式检验，不证明固件 PD 实现或电机效率。

**当前新框架：**`CommandFrame::JointCommand` 的 position [rad]、velocity
[rad/s]、Kp [N·m/rad]、Kd [N·m·s/rad]、feedforward_effort [joint N·m]
均为 joint-space；`MotionRuntime` 对 RL 命令设 `JointImpedance`、
`feedforward_effort=0`，并在提交前对 position 作硬件范围裁剪。
MuJoCo backend 以这些 joint-space 字段计算 PD+前馈，再裁剪到
`RobotModel.max_effort` 与 MuJoCo actuator control range 的交集。
`RobotModel.max_effort` 同时用于 `CommandFrame` 前馈字段校验；**这种
MuJoCo 软件裁剪尚未在 real RobotIO 中实现**。当前 black 配置中的
hip/thigh `23.7 N·m`、calf `59.25 N·m` 是 joint-side 配置上界，
与 blackW 配置及仿真 actuator range 一致；源码不能证明 Black/BlackW 实物
电机完全相同，或这些值是连续、峰值、固件保护阈值。

**RealRobotIO v1 已冻结的命令语义：**接收 canonical RobotModel 顺序的
joint-space `q_command`（正常 Black PPO 路径为
`clamp(q_default+0.25*raw_action, hardware position limits)`；现有
`max_position_jump` guard 若触发还会进一步限制相邻目标变化）、
`dq_command=0`、`Kp=40`、`Kd=1.2`、`tau_ff=0`，再由 backend 按 §19.14
进行标定、sign/gear 和 SDK 字段转换。接口保留 joint-space `tau_ff` 语义，
Black PPO v1 使用零值。policy 不输出 torque action，也不承担硬件力矩保护；
位置安全先于电机阻抗命令。`RobotModel.max_effort` 当前是配置的 joint-space
上界，不能仅凭此声称真实固件会把**隐式 PD 输出**限到该值；
RealRobotIO/电机固件必须承担 SDK 字段校验、故障处理及经硬件确认的
effort/current 安全机制。是否可在 SDK/固件中限制总 PD effort，或需另设
软件保护及其数值，**NEEDS HARDWARE DECISION**，不能靠裁剪 `tau_ff=0`
实现。SDK 头文件标注的 `motor.T ±127.99 N·m` 是转子侧命令字段范围，
`K_P/K_W 0–25.599` 是增益字段范围，均不是已证实的物理输出上限。
源码可识别 `GO_M8010_6` 型号、故障标志（过热/过流等）和预编译 SDK，
但不能证明旧 Black 实际 joint effort/current ceiling：**SOURCE INSUFFICIENT**。

已有 sim2sim 证据只显示：旧 `rl_sar` 计算值最大约 27.79 N·m；新框架
平地 rollout 的 `tau_raw` 最大约 20.93 N·m，未命中 23.7/59.25 裁剪。
这不构成实机安全证明。实机前仍需核对实体电机型号/固件、运行电流/力矩
阈值、阻抗控制内部限幅与故障行为，并决定 real backend effort safety 数值及
执行位置；本节未修改 RobotModel 或实现 backend。IMU 静态契约见 §19.16，
时序与 watchdog 静态契约见 §19.17。

### 19.16 Black real backend v1 IMU / orientation contract（静态冻结）

**Policy 与训练侧：**Black PPO 45-D actor 只取 `0.25 * ω_B` 和
`g_B=R_WB^T[0,0,-1]`；`B` 是 canonical trunk/root-link body frame，
`R_WB` 将 body 向量旋至 world，姿态 quaternion 为 `wxyz`。MjLab v1.6.0
`base_ang_vel` 取 `root_link_ang_vel_b`，`projected_gravity` 取
`quat_apply_inverse(root_link_quat_w, gravity_vec_w)`，其中
`gravity_vec_w=[0,0,-1]`。actor 不使用 raw accelerometer、base linear velocity、
absolute yaw、magnetometer 或 world position。当前 MJCF 的 trunk IMU site
相对 trunk 为 identity；训练 observation 取 root-link 状态，并非直接读取
gyro/accelerometer sensor。

**旧实机软件链：**AB5465 的 CRC 有效包按小端 float 读取 Euler角、gyro、accel。
`real_runner` 对 gyro 施加 `diag(1,-1,-1)` 且按 raw 为 deg/s 的假定转为 rad/s，
对 accel 施加相同轴旋转但不作单位缩放；该矩阵是右手系绕 X 轴 180° 的
proper rotation。驱动假定 accel 数值可按 m/s² 送入 VQF，原始设备单位及
是否为 specific force **SOURCE INSUFFICIENT**。默认 VQF 用固定 2 ms
采样周期输入 gyro/accel，不使用 magnetometer，输出 6D `wxyz` quaternion：
软件变换后 sensor frame → 重力对齐、yaw 原点任意的惯性 frame；内部启用
gyro bias 估计。VQF 输入无效时旧代码退回由 raw Euler
`(roll,-pitch,-yaw)` 生成的姿态，仍会发布该帧；发布 ROS `Imu` 时
`frame_id=imu_link`、时间戳为主机 `now()`，没有原始 sensor timestamp。
用户已确认当前实体 Black 的 AB5465 安装与配置延续此 legacy 实机路径；
Black v1 将该轴变换、gyro deg/s→rad/s 及 VQF 6D 输出作为 body-frame
部署依据，不增加另一项 Black 专用安装旋转。

`rl_sar` 从 `/_lowState/imu` 直接复制 ROS orientation（xyzw 字段重排为
内部 wxyz）及 angular velocity，不再作轴变换或归一化；
obs[3:6] 为 `0.25 * incoming_gyro`，obs[6:9] 为
`QuatRotateInverse(incoming_quat,[0,0,-1])`。旧链默认信任驱动已给出
body-frame gyro 与 body→重力对齐 world 的 orientation；实体 frame 一致性
已由上述实体配置确认。VQF 6D 的 absolute yaw 不可观测；只要 roll/pitch 与 frame
一致，yaw 原点/漂移本身不直接进入重力投影，但不能据此忽略 estimator
或 frame 错误。既有 sim2sim 数值 trace 已验证 policy 侧投影公式
（§17.10）；实体安装依据是本次用户确认，而非该仿真 trace。

**新框架边界：**`StateFrame::ImuState` 已定义 body→world `wxyz`、
body-frame angular velocity [rad/s]、body-frame acceleration 字段 [m/s²]、
`age_ns` 与 `valid`；MuJoCo backend 从 trunk identity IMU site 的
framequat/gyro/accelerometer 填充。`RlController` 使用 orientation 计算
projected gravity，使用 angular velocity 构造 actor observation；
RealRobotIO 不输出 policy 专用 gravity 或 45-D observation，也不施加 0.25
观测缩放。若传感器 frame `S` 与机体 `B` 有固定安装旋转 `R_BS`（S→B），
backend 应给出 `ω_B=R_BS ω_S`、`a_B=R_BS a_S`、
`R_WB=R_WS R_BS^T`；当前 Black v1 的 `R_BS` 按已确认的 legacy
`diag(1,-1,-1)`，不额外叠加安装变换。

**有效性与待确认项：**RealRobotIO v1 必须只发布有限、已初始化、
近单位且对应同一坐标系的姿态/角速度/加速度，必要时处理小幅数值漂移，
不得把无效 quaternion 静默替换为 identity；过期/无效数据标为 invalid，
RL 不得消费。当前 `RlController` 检查 `imu.valid` 和 quaternion
`|norm²-1|≤1e-4`，`StateFrame` 通用校验检查有限数值与非负 `age_ns`，
但没有 IMU age 上限。旧驱动没有可靠的 orientation-ready 闸门；
freshness 与故障转移接口见 §19.17，具体收敛判据和数值阈值待 bring-up。
该 acceleration 字段在真实驱动与 MuJoCo accelerometer 间是否都表示
specific force 尚未由当前源码统一证明；Black PPO 不消费该字段。
AB5465 原始 accel 物理语义、timestamp 质量、启动收敛、丢包、振动与 bias
表现仍需 bring-up 核查；这些不是 IMU 安装/frame blocker。
本节只冻结接口语义，
未实现 real backend 或作实机验证。

### 19.17 Black real backend v1 timing / freshness / watchdog / failure contract（静态冻结）

**节拍与时钟。**MjLab v1.6.0 velocity cfg 为 `timestep=5 ms`、`decimation=4`；
`ManagerBasedRlEnv.step()` 在四次 physics substep 反复 apply 同一 action target，
故 Black PPO 每 20 ms（50 Hz）产生新关节目标。实机只须保持此 policy 语义，
不得把训练 physics thread 当成硬件线程设计。当前 `quadruped_control` Black
`control_period=5 ms`、`command_validity=10 ms`；`MotionRuntime` 在每个控制周期
`read_latest → validate → FSM/RL → submit`，`kRlDecimation=4`，首次进入 RL 立即推理，
其余 3 周期保留最新 `rl_command_` 并重新提交 joint impedance 命令。
`motiond` 由新 StateFrame 事件及其 timestamp 调度 5 ms 更新，无状态时以
`steady_clock` 约 5 ms 触发失败检查；MuJoCo app 按仿真时间调度控制，物理线程
以 `steady_clock::sleep_until` 节流。实机 StateFrame、CommandFrame 和
BaseCommand 的 timestamp、age、expiry 必须处于同一 monotonic 时钟域；
policy 更新、低层命令刷新、IMU 采样、各电机串口轮询是四种独立节拍。

| 当前调度阶段 | 目标周期 | 在当前路径中的工作 |
|---|---:|---|
| control update | 5 ms | RobotIO read、StateFrame validation、FSM/obs（仅推理帧）、CommandFrame submit；`motiond` 需新状态事件 |
| policy update | 20 ms | 在上述同步 control update 内 observation build、Torch forward、action conversion；`forward >20 ms` 才在完成后判失败 |
| backend apply | MuJoCo 物理 2 ms；实机待定 | 应用最新有效命令或独立安全回退；不得把 20 ms policy 间隔当成命令过期阈值 |

这张预算描述当前调用链，不证明 policy 帧总耗时小于 5 ms；
当前一次推理会占用同一个 control update，实机需隔离低层命令刷新。

旧 `rl_sar` Black 的 `dt=5 ms,decimation=4` 实际由两个独立 `LoopFunc` 线程
分别执行 `RobotControl` 和 `RunModel`；后者是 20 ms 周期的 steady-clock
执行时长扣减再 sleep，不是 ROS 回调或计数器保证 50 Hz，超时不补偿。
旧 `real_runner` 外层交换/ROS joint 发布以 `rclcpp::Rate(1/0.005)` 目标 200 Hz；
四条腿各有 `steady_clock::sleep_until(start+2 ms)` 的串口线程目标 500 Hz，
IMU 为异步串口收包，VQF 固定 2 ms 样本 dt。均只是**目标节拍**，
实际总线、传感器及 ROS 发布频率 SOURCE INSUFFICIENT，未见各样本可靠采样时戳。

**状态与命令新鲜度。**`StateFrame.header.timestamp_ns` 是完整快照生成时间；
每个 `JointState.age_ns` 与 `ImuState.age_ns` 表示距各自最后成功更新的时间，
由 backend 按同一 monotonic 时钟填写，不由 MotionRuntime 猜测。
`valid/online/error_code` 也由 backend 从采样完整性、通信和电机故障填充；
不存在独立 base-state age 字段。MuJoCo backend 全部 age=0/valid=true，
没有硬件时延模型。当前 core validation 只检查 age 非负、数值 finite，
`MotionRuntime` 检查 joint online/valid 和 executor `ControlEnabled`，
`RlController` 检查 IMU valid、quaternion `|norm²-1|≤1e-4`；
**尚无 joint/IMU age 上限，也不检查 `error_code` 非零**。
`RemoteRobotIO` 的 500 ms heartbeat 仅是进程存活判据，不能充当采样新鲜度；
它在共享内存状态槽短暂不可读时可能返回同一旧帧。`motiond` 只在新帧版本到来时
执行正常更新；实机必须另设 wall-clock freshness/watchdog，即使没有新帧也会触发。
一帧可组合不同时间的 IMU 与各关节样本，但 backend 必须拒绝超龄、无效、
时间倒退或超过允许采样偏差的快照；无需强求硬件同时采样。IMU 和每关节
age 阈值、最大样本间偏差、orientation-ready 判据须在 bring-up 测量后配置，
明显短于旧 0.5 s（该值相当于 25 次 policy 更新或 100 次控制周期），
不能照搬。单包丢失仅在上一个样本仍满足全部 freshness 条件时可继续。

`MotionRuntime` 每个**成功控制周期**以本周期 timestamp 生成新 CommandFrame，
`expires_at=timestamp+10 ms`；10 ms 是执行侧每帧有效期，不是 policy action
20 ms 的寿命。两次推理之间每 5 ms 重新提交相同目标，故正常运行不会在 20 ms
间隔内过期。backend 必须在**应用**时按 monotonic now 再验 expiry、
session/sequence 和数值；过期不得重放。MuJoCo backend 过期即执行器 `ctrl=0`，
`RemoteRobotIO::submit` 当前以帧自身 timestamp 作校验，不能替代执行侧检查。
policy 最后有效目标也必须有独立、有限的 hold 寿命，不能靠不断刷新
CommandFrame 无限延长。当前 `BaseCommand` 过期时 `RlController` 把速度观测归零，
不退出 RL；ROS gateway 默认 `cmd_vel` 200 ms、Joy 250 ms freshness，
仿真 Terminal 每步刷新测试命令（100 ms expiry）。旧 `rl_sar` joystick 300 ms
归零，navigation `/cmd_vel` 未见等价超时。高层速度命令失效仅 ZERO HIGH-LEVEL
COMMAND；joint/IMU 失效则属于状态完整性故障，不得仅清零速度后继续 RL。

**当前失败路径及实机要求。**`TorchPolicy` 用 `steady_clock` 测量同步 forward，
内部设 Torch intra/inter-op 各 1 线程并预热 3 次；输出要求 CPU float32
`[1,12]` 且全 finite。异常、shape/type/device/NaN/Inf 均使 `inference.ok=false`；
`RlController` 再检 action 维度和 finite，绝不对 NaN clamp。
`kRlInferenceDeadlineNs=20 ms` 是显式常量：一次 forward **完成后**若耗时
`>20 ms`，即使 action 有效也调用 `fail_active_motion`，退 `MotionMode::Passive`，
同周期在状态可用时提交各关节 `ControlMode::Disabled`；没有可用状态时不能提交。
推理本身在 5 ms `MotionRuntime::update()` 调用内同步执行，故一次较慢推理
也会阻塞本轮读/写及后续调度，20 ms 检查不能代替独立执行侧 watchdog。
旧 `rl_sar` / `real_runner` 无等价 inference deadline、输出 finite gate 或
可靠旧命令失效处理；serial `sendRecv` 失败只打印，仍把 `localState` 拷入共享反馈。
旧 30°/0.5 s/8π safety latch 因 `ENABLE_LATCHED_SAFETY_PROTECTION=false`
不生效；8π 分支会跳过本次命令更新，serial 失败 latch 被注释。

实机 v1 将**单次有效但迟到的推理**和单包丢失视为可恢复候选：
仅在所有状态仍 fresh、无电机故障、链路可写且上一 target 有效时，
允许 BOUNDED HOLD 上一已校验 `q_command`、`dq=0`、原 Kp/Kd、`tau_ff=0`；
最多跨一次 20 ms policy 更新，下一更新仍失败便退出 RL。此 hold 不能
复用过期 CommandFrame，低层须发带新时间戳的命令，且不更新 previous_action。
**当前同步 MotionRuntime 没有该路径，real backend 不可直接沿用**；
执行侧独立调度与有限 hold 的架构见 §19.18，完成相应 implementation units
之前不得启用实机主动控制。
policy 异常、非有限/错维输出、非有限 state、无效姿态、标定缺失、critical motor
fault、持续超龄/通信失败及 E-stop 不允许继续 RL 或 hold 旧 RL 目标。

SAFE HOLD 指退出 RL 后执行侧能通信且阻尼模式已被硬件确认时，
发送 `ControlMode::Damping`：position 无效、`dq_target=0,Kp=0,Kd=经验证的安全阻尼,
tau_ff=0`，不再使用 policy target。旧 Black Passive 实际为 mode=1、Kp=0、
Kd=3 joint-side、tau=0；旧安全分支拟用 joint-side Kd=6 但上述 latch 已关闭，
二者不能直接作为新实机阻尼定值。DISABLED 是另一种命令：
`ControlMode::Disabled`、无 position/velocity/PD/torque 主动输出；MuJoCo
实现为零 torque。真实 SDK 应如何编码 motor mode/zero gain/zero torque、
失联后电机固件会怎样动作，均需 bring-up 确认；**不能将 Disabled
自动解释为物理断电或安全支撑**。当前 `GetDown` 可在有 rest pose、
状态和执行侧安全条件满足时从 RL 受控趴下；`Stand` 请求从 Running 被拒绝，
`EnterPassive` 立即 Disabled。正常退出优先用 GetDown；若需 RL→Stand，
须单独定义过渡，不得声称现有代码已支持。
SAFE HOLD 与 DISABLED 最终切换条件、阻尼值及掉电/跌倒风险需实体试验。
外部 E-stop 覆盖 RL、Stand、SAFE HOLD 和 Disabled，须能由独立硬件安全通道
使电机进入已确认状态，具体实现不在本单元。

| Failure | Legacy 当前行为 | quadruped_control 当前 | Real backend v1 决策 |
|---|---|---|---|
| 高层速度命令超时 | Joy 300 ms 归零；navigation 无等价检查 | BaseCommand 过期归零；gateway cmd_vel 200 ms / Joy 250 ms | ZERO HIGH-LEVEL COMMAND，保持 RL |
| 单个 IMU 包丢失 | 保留上次值；0.5 s latch 关闭 | 仿真无丢包模型；无 IMU age 上限 | 状态仍 fresh 才继续；否则 EXIT RL → SAFE HOLD |
| IMU 持续 stale | latch 关闭；旧数据可继续发布 | age 不查 | EXIT RL → SAFE HOLD；无法执行则 DISABLED/硬件保护 |
| 无效 quaternion | VQF 无效输入可退回 Euler 并发布 | 非单位/非有限退 Passive/Disabled | EXIT RL → SAFE HOLD，禁止使用该帧 |
| 单个电机包丢失 | serial 失败日志后可能复制旧/无效反馈 | 仿真无丢包模型；age 不查 | 状态仍 fresh 才继续；否则 EXIT RL → SAFE HOLD |
| joint state 持续 stale / 非有限 | 无可靠 per-motor freshness gate | online/valid 或 finite 失败退 Passive；age 不查 | EXIT RL → SAFE HOLD；不能可信阻尼则硬件保护 |
| 推理有效但 >20 ms | 无显式 deadline | 立即 Passive/Disabled，同步阻塞 | BOUNDED HOLD 至多一次 policy 更新；再次失约退出 RL |
| 推理异常 / shape 错误 / NaN / Inf | 无完整 gate | 立即 Passive/Disabled | EXIT RL → SAFE HOLD，禁止使用输出 |
| 标定无效 | 旧 offset 缺失仍可能继续 | 无实体标定输入 | FAIL CLOSED BEFORE ENABLE |
| serial TX 失败 | `sendRecv` 失败仅日志 | 无实机链路 | 立即停止信任写入成功；EXIT RL，失联依赖独立硬件 watchdog |
| serial RX 失败 | 旧/无效反馈可能被拷入共享状态 | 无实机链路 | 上一采样仍 fresh 方可短暂容忍；超龄 EXIT RL → SAFE HOLD/硬件保护 |
| 电机 fault | 故障码未系统传播 | JointState.error_code 有字段但 runtime 未检查 | EXIT RL → SAFE HOLD 或硬件保护；critical fault 锁存 |
| 用户正常退出 RL | FSM 可转 GetUp/Passive 等 | GetDown 可受控；Stand 请求从 Running 拒绝；EnterPassive 直接 Disabled | 优先 GetDown；RL→Stand 需另定义；显式 Passive 需实机评估 |
| 外部 E-stop | 旧代码不能证明独立 E-stop | SafetyState 有 EmergencyStop，但仅抽象状态 | 立即退出 RL，由独立硬件安全通道覆盖全部软件模式 |

**进入 RL 与归属。**仅在标定有效、全关节 fresh/online/valid、IMU 姿态完成
初始化且 fresh、无 critical fault、执行侧允许主动控制、policy 已预热、
有效 BaseCommand 与 Stand/受控过渡条件满足时准入。激活 policy 会清零
previous_action、history 和目标，第一帧 previous_action=0，立即推理；
第一目标仍受 `max_position_jump=1.0 rad` 与硬件 position clamp 约束。
RealRobotIO 负责硬件采集、每关节/IMU timestamp 与 age、标定、fault、命令
编码及执行侧失效保护；MotionRuntime 负责 5/20 ms 调度、freshness gate、
模式/失败转换；执行侧负责独立于 motiond 的有限 hold 与最终 watchdog；
RlController 只处理 canonical state→obs→actor→
安全裁剪后的 joint target，不感知串口与 ROS 丢包。

**剩余实现前关口。**无状态帧时不能依赖 MotionRuntime 发送 SAFE HOLD，
实机执行侧须独立监督 CommandFrame 失效、`motiond` 停滞及串口通信失效。
同步推理阻塞 5 ms 控制路径与所需 BOUNDED HOLD 的隔离方案见 §19.18。
joint/IMU stale 阈值、样本偏差、hold 寿命
（上界已定为一次 policy 更新）、阻尼值、firmware mode 和 fault 恢复条件
均为 **BRING-UP TUNING / HARDWARE DECISION**，未作实机验证。
既有 rollout 在 `OMP_NUM_THREADS=1` 下 10949 次推理 max 12.85 ms，
但当前 `TorchPolicy` 源码另在创建时设置 Torch 线程数为 1 并预热；
这些仅是已测试配置及降低长尾的措施，不是 5/20 ms 实时保证。

### 19.18 Black RealRobotIO v1 architecture / implementation plan（静态冻结，未实现）

**进程与责任。**沿用 `ros2_gateway → motiond(MotionRuntime + RL + Torch)
→ IPC → real_backendd → RealRobotIO → 四条电机总线 + AB5465 IMU`。
`alldog_mjlab` 仍仅通过 policy deployment contract 对接，不依赖部署工程。
`real_backendd` 是独立 IPC owner，负责 session/heartbeat、CommandFrame intake、
StateFrame/RobotIOStatus 发布和硬件生命周期；daemon 主循环不执行串口交易。
`RealRobotIO::read_latest()` 仅快照缓存、按 FL/FR/RL/RR × hip/thigh/calf
组装 canonical StateFrame，按 monotonic now 计算每关节及 IMU age，返回有效性、
online/error/fault/safety 与序号；`submit()` 仅校验 schema、identity、session、
sequence、数值和时间并缓存命令。两者均不得等待串口、IMU 或四腿反馈；
accepted command 不等于 hardware executed。状态缓存只保留最新值，不向
MotionRuntime 暴露高频历史队列。

**执行线程。**v1 内部固定四个一腿一总线 motor worker、一个 IMU worker、
一个 execution/safety supervisor。每个 motor worker 独占 fd/SDK/三电机，
发 supervisor 批准的 snapshot，验证反馈包/ID，记录各电机 monotonic 样本时间、
通信及故障；legacy 约 2 ms 仅是 bring-up 测量起点，一条总线阻塞不能拖停其余
三条。IMU worker 独占 AB5465 parser 与 VQF，发布 canonical body-frame IMU：
`diag(1,-1,-1)`、gyro deg/s→rad/s、VQF 6D body→world wxyz；不构造 observation。
supervisor 在独立线程按同一 host monotonic clock 检查命令、target、反馈、IMU、
session、fault 和 E-stop，决定 effective mode 与每条总线的命令快照。优先级：
E-stop/硬件保护 → backend 锁存故障 → 无效或 stale 硬件状态 → frame/target watchdog
→ 请求的安全模式 → 正常 CommandFrame。worker 还须有 supervisor 停滞时的最后
失效保护或经验证的固件 watchdog；不得无限重放旧 snapshot。

**同步 policy 的前提。**v1 保持 `Policy::forward()`、history、previous_action 和
`MotionRuntime` 5 ms control / 每四周期一次 20 ms policy 的同步路径；不加 async
Policy API、推理队列或新 RL controller。只有最终执行与 watchdog 独立于
`motiond`/Torch/IPC 返回，才接受这个决定。推理迟到、motiond 卡死或失联时，
执行侧在 target 仍有效且所有状态 fresh、可写、无故障时可 BOUNDED HOLD；
新的显式 Passive/Disabled 命令立即覆盖旧 RL target。target 到期或状态失效则
退 SAFE HOLD/已验证的硬件 fallback，绝不继续 impedance hold。若实测同步
推理抖动仍影响步态，异步调度另作 v2 unit。

**Unit 0 必先补足时间语义。**现有 `CommandFrame` 只有 frame timestamp 与
`expires_at_ns`，IPC wire 同样缺 semantic target 年龄；每 5 ms 重新包装同一 RL
target 会掩盖其实际年龄。新增 `target_generated_at_ns`、`target_expires_at_ns`
（字段名可调整，语义不变）：成功推理且完成 action conversion 才更新 RL target
生成时间，余下三个 decimation cycle 原值不变；frame expiry 仍只管 transport/
control frame，不能延长 target hard expiry。由 control period、RL decimation、
允许 miss 次数推导 hard lifetime；Black 当前 5 ms × 4 × (1+1)=约 40 ms，
不是全机器人常量。frame 过期但 target 未到期时，仅在硬件状态全部合格的前提
下允许上一已验证 `q_command`、`dq=0`、原 Kp/Kd、`tau_ff=0` 的有限 hold。
state sensor sample、StateFrame 生成、target 生成、CommandFrame 生成、
execution/application 各有独立 timestamp；现有 MotionRuntime 把 `state.now_ns`
用于命令生成，不能代表推理结束后的真实时间。实机 safety 时间统一 host
monotonic 域；仿真/replay 保持各自明确的时钟域，不能直接改用 host wall time。
Unit 0 须同步修改 core validation、IPC wire/schema/version、MotionRuntime 时间
来源和 session/reset 清理，协调两端版本切换，保持 MuJoCo/replay 现有行为。

**配置、执行与诊断。**`backends/real` 私有版本化 hardware config（建议
`configs/hardware/black.yaml`）记录 robot identity、四 bus device/baud/timeout、
12 joint 的 bus/ID/sign/gear/fixed calf correction、IMU device/axis/unit/VQF 及
安全参数入口。每台机器的 straight/creep calibration 另以显式路径加载，按
joint name 键控、12 关节完整、有限、带版本与 robot identity；runtime multi-turn
offset、mapping/conversion 和标定均只归 real backend。任一配置/标定/身份/首批
样本/IMU 初始化失败，禁止 ControlEnabled 和 RL，不能暗用默认标定。
v1 仅接受 Disabled、Damping、JointImpedance；Velocity/Torque 显式 Rejected。
Disabled 无主动 position/velocity/PD/FF 输出，不预设等于断电；SAFE HOLD 为
Damping（`Kp=0,dq_target=0,Kd=经验证安全值,tau_ff=0`，无 position target）。
SDK mode、阻尼值和失联后的物理动作须台架确认，未确认前 active enable fail closed。
critical over-current/temperature、错误 motor identity、持续通信失败、无效标定、
E-stop 锁存 backend fault 并离开 ControlEnabled；v1 可要求重启/重新初始化清除，
不得伪造 ResetFault。`last_accepted_command_sequence` 记录缓存接受；全帧
`effective_command_sequence` 仅在所需四总线均成功发送/接受同一执行 snapshot
时推进，部分成功必须暴露为未完成及 per-bus 诊断。无 actuator ACK 时只能称
软件侧已发送，不能称物理执行已确认；fallback 也不能冒充旧 RL 命令生效。
外部 E-stop 覆盖全部软件模式，具体硬件通路另行验证。

**依赖边界与分步实现。**只集成所需 motor SDK headers/libs、AB5465 parser 和
VQF，固定版本及 binary/hash；依项目依赖机制或 `.deps/` 获取，不复制旧
`real_robot` workspace、也不提交私有预编译二进制。可在 `backends/real/src/internal/`
使用私有 MotorTransport/ImuTransport fake seam，不建公开多厂商 plugin。
建议模块为 `backends/real` 的 RealRobotIO、motor conversion、calibration、
IMU processing、worker/supervisor，另设 hardware config loader、
`apps/runtime_daemons/real_backendd.cpp` 和 `configs/hardware/black.yaml`；
不增加 Black 专属 MotionRuntime、RL controller 或 observation。当前 CMake 的
motiond 受 Torch 选项约束且源码直接依赖 Torch，`scripts/build.sh --target motion`
还打开 MuJoCo；后续最小拆分应使 real_backendd/read-only/basic motion path 无
MuJoCo、无 Torch 可构建，RL enable 才要求 Torch，保持同一进程拓扑。

| Unit | 范围与完成门槛 |
|---|---|
| 0 — core target timing | 只补 target 生成/硬过期、frame/state/command 时间语义、validation、IPC wire、RL decimation/session/reset；离线验证 timestamp 仅成功推理更新、三次复用不更新、重新包装不延寿、IPC round-trip/schema mismatch。无实机。 |
| 1 — config + pure conversion | hardware schema、joint→bus/ID、sign/gear/calf correction、独立 calibration、multi-turn offset、IMU static config；离线验证错误 identity、重复 bus/ID、缺项/NaN、转换 round-trip。不打开 serial。 |
| 2 — read-only RealRobotIO | fake transport 的 worker/cache、非阻塞 read_latest/submit、sample age/invalid packet/单 bus 与 IMU loss、fault/status、部分执行诊断；主动输出关闭。 |
| 3 — real_backendd read-only IPC | shared memory/session/heartbeat/StateFrame/RobotIOStatus、shutdown/reconnect；端到端读取真/假状态，仍禁止主动电机输出。 |
| 4 — execution supervisor | fake transport 测 frame/target expiry、bounded hold、motiond loss、session change、fault latch、partial bus、sequence、Disabled/Damping/JointImpedance、shutdown；Velocity/Torque 拒绝。 |
| 5 — actual transport | 接 legacy SDK、AB5465、VQF 与 serial，先只读核查 12 电机及 IMU、时间戳与 fault，不启用主动输出。 |
| 6 — hardware safety bring-up | read-only → Disabled 语义 → 单电机 Damping → 单腿 → 四腿 → GetUp → Stand → GetDown；每步台架记录与放行。 |
| 7 — RL enable | 仅前序全通过后进入 Black PPO，先零速度命令再低速前进，记录 freshness、延迟、watchdog、fault 与安全退出。 |

后续每单元保持 `./scripts/build.sh`、`./scripts/test/ctest.sh`、
`./scripts/test/ros2_headless.sh` 通过；Black PPO observation/action、history、
previous_action、policy switch、MuJoCo/replay 不得回退。**首个实机端到端里程碑**
是主动输出 Disabled 下的 12 motor feedback + IMU → canonical StateFrame → IPC
→ motiond/gateway，且 freshness、fault、calibration validity 可见，不是 RL 行走。
joint/IMU stale 数值、sample skew、transaction timeout、safe Kd、SDK Disabled
编码、固件/外部 E-stop 行为、effort/current ceiling、物理接线和实时抖动均需
实机 bring-up；本节未实现或验证硬件。具体后续实施已移交到
`quadruped_control` 项目，本仓库继续仅维护训练侧与显式 policy deployment contract。

------

## 20. Frozen Black HIM Observation / History / Estimator Target Contract

阶段状态：

```text
Black HIM observation / history / estimator target contract: COMPLETE
HIM estimator / HIMPPO / warm start / exporter:               NOT STARTED（后续 unit 完成；task 注册见 §23 / §24）
```

本单元只建立 HIM algorithm integration 之前的 task-side 数据契约，不包含任何算法更新，
也**不注册** `black-flat-him` / `black-rough-him` task（HIM runner 未实现）。
`black-flat` / `black-rough` 的现有 PPO 任务不受影响。

### 20.1 Actor history

```text
group                 actor（复用普通 Black PPO 的全部 actor term）
MjLab history_length  6
flatten_history_dim   False
actor group obs       [B, 6, 45]（frame-major，MjLab 内部 oldest → newest）
当前帧                actor_history[:, -1, :]
canonical flatten     actor_history.flip(1).flatten(1) → [B, 270]（newest → oldest）
```

actor term 顺序、scale、noise、clip 全部保持冻结的 Black PPO contract：

```text
command / base_ang_vel / projected_gravity / joint_pos / joint_vel / actions
```

history 由 MjLab v1.6.0 `ObservationGroupCfg.history_length` 原生实现
（pipeline 仍为 compute → noise → clip → scale → history），不是 runner-side FIFO，
也不是旧 legged_gym 的 `obs_buf` 更新。
task-side 有意保持 `flatten_history_dim = False`；term-major flatten 不能当作
official HIM 的 270-D history。canonical frame-major flatten 由 model adapter
（`him.canonical_him_history()`）在后续 algorithm unit 负责。

### 20.2 History reset semantics

采用 MjLab v1.6.0 原生语义：reset 后第一帧 post-reset observation 填满整个 history
（`[obs0] × 6`）。partial reset 只影响被 reset 的 env，其余 env 的 history 不前进、
不被污染。不沿用旧 HIMLoco 可能保留上一 episode history 的隐式行为。

### 20.3 Estimator velocity target

```text
group    estimator_velocity
shape    [B, 3]
term     envs_mdp.base_lin_vel（root_link_lin_vel_b，body frame）
scale    2.0（official HIM obs_scales.lin_vel）
noise    none（privileged target）
frame    body frame（不是 raw world-frame velocity）
```

`base_lin_vel` 与 Black 的 IMU velocimeter（`robot/imu_lin_vel`，imu site 位于 trunk
原点）等价；使用 root body-frame lin vel 更直接对应 official HIM。

### 20.4 Estimator target encoder input

official HIM 的 critic packing trick（`next_critic_obs[:, 3:48]`）不迁移，显式定义为：

```text
successor actor frame 去掉 command（42）
+ successor scaled true base linear velocity（3）
= 45
```

由 `him.him_target_encoder_input(frame, velocity, spec)` 构造；切片维度来自
`BlackHimObservationSpec`（`command_dim` / `velocity_dim`），不是 magic slice。
当前 actor frame 的唯一来源是 `actor_history[:, -1, :]`；不新建第二套 noisy
current observation group（否则 noise 会独立重采样，与 history 当前帧不一致）。

### 20.5 Terminal transition

HIM estimator 使用 successor observation；MjLab 的 `step()` 会对 done env 自动 reset，
返回的是新 episode 第一帧，不能作为 terminal successor target。

采用 MjLab v1.6.0 原生 `RecorderTerm.record_pre_reset`（在 `_reset_idx` 之前触发），
从 terminal state 重新计算并保存：

```text
env.extras["him_terminal_env_ids"]             [n]
env.extras["him_terminal_actor_frame"]         [n, 45]
env.extras["him_terminal_estimator_velocity"]  [n, 3]
```

- 计算前调用 `sim.forward()` + `sim.sense()`：`step()` 中 derived quantities 落后一个
  physics substep，terminal target 必须基于 terminal state 本身；
- actor frame 复用与 ObservationManager 相同的 pipeline（compute → noise → clip →
  scale）与已 resolve 的 term cfg，因此 corruption 开关一致；
- noise 为独立重采样，与正常 transition 同分布，而不是同一个 draw；这与 official HIM
  的 `compute_termination_observations` 一致；
- 没有 done env 的 step 结束时清除这些 key，避免 stale target 被 algorithm 使用；
- 代价：有 reset 的 step 会额外执行一次 forward / sense 与 actor frame 计算
  （仅供 terminal target 使用）。

### 20.6 Critic

critic 继续使用当前 259 维（原 72 维 + 187 维 height_scan），flat / rough 相同。
不恢复旧 HIMLoco 的 238-D privileged observation。

### 20.7 维度 specification

`him.black_him_observation_spec(env)` 从 runtime resolved dims 读取：

```text
single_frame_dim         = 45
history_length           = 6
history_dim              = 270
command_dim              = 3
velocity_dim             = 3
action_dim               = 12
target_encoder_input_dim = (single_frame_dim - command_dim) + velocity_dim = 45
```

后续 algorithm 必须从这里取维度，不得硬编码 270 / 238 / 45 / 12。

### 20.8 Config 与 task registration

```text
black_flat_him_env_cfg  = black_flat_env_cfg  + HIM observation / terminal contract
black_rough_him_env_cfg = black_rough_env_cfg + HIM observation / terminal contract
```

reward / command / reset / DR / termination / action / terrain / robot config 全部复用，
没有第二套 `black_config`。`black-flat-him` / `black-rough-him` 尚未注册：HIM
actor-critic / runner 未实现，不能复用普通 PPO runner（其 actor 输入为单帧 45-D）。

### 20.9 本单元不做

```text
HIM estimator 网络 / Sinkhorn / prototype loss
HIMPPO update / custom HIM runner
PPO→HIM warm start
HIM TorchScript exporter
deployment contract 修改
BlackW / symmetry
```

------

## 21. Frozen Black HIM Algorithm Contract

阶段状态：

```text
Black HIM algorithm（HIMPolicy / HIMEstimator / HIMPPO / HIMRolloutStorage）: COMPLETE
MjlabOnPolicyRunner integration / task registration / checkpoint round-trip:     NOT STARTED（后续 unit 完成；见 §22 / §23 / §24）
```

本单元在 Unit 1 task-side contract 之上实现 HIM 模型与训练算法，并在 RSL-RL v5.4.2 +
MjLab v1.6.0 上完成真实 rollout → storage → update。仍未注册
`black-flat-him` / `black-rough-him`，也不包含 runner / warm start / exporter。

### 21.1 Source authority

```text
official:  InternRobotics/HIMLoco @ ef289acaa62795009363b7b819c9186690630441
           （2024-05-14）
           rsl_rl/rsl_rl/modules/him_{estimator,actor_critic}.py
           rsl_rl/rsl_rl/algorithms/him_ppo.py
           rsl_rl/rsl_rl/storage/him_rollout_storage.py
RSL-RL:    rsl-rl-lib 5.4.2（uv.lock；本机安装源码）
MjLab:     1.6.0
legacy:    coverMoon/super-dog（本地 ~/PROJECT/Dog/Train/HIMLoco）仅作工程参考
```

与 official 的真实差异（已重读源码确认，不沿用 Unit 1 结论）：

```text
target encoder    official tar_hidden_dims = [128, 64]
                  super-dog 与旧 alldog prototype = [128, 128]（有意不采用）
target input      official 在 238-D privileged obs 里取 [3:48] / [45:48]
                  旧 alldog prototype 取 [0:45]（含 command）+ [45:48]（两处偏离）
                  本单元显式化为 next_estimator_input / next_estimator_velocity
optimizer         official = Adam(actor_critic.parameters())，estimator 也在同一个 PPO optimizer，
                  只靠 no_grad 避免被更新；本单元显式分账（见 21.4）
```

### 21.2 网络结构（released code，不是论文）

```text
source encoder: history_dim → 128 → 64 → velocity_dim + latent_dim（ELU）
target encoder: target_dim  → 128 → 64 → latent_dim（ELU）
prototypes:     32 × latent_dim，temperature 3.0
actor:          actor_input_dim → 512 → 256 → 128 → action_dim（ELU）+ Gaussian std
critic:         RSL-RL 原生 MLPModel（259 → 512 → 256 → 128 → 1，normalization 开启）
```

Black 当前实例（全部由 `HIMSpec` 派生）：

```text
history_dim 270 / target_dim 45 / velocity_dim 3 / latent_dim 16
actor_input_dim 64 / action_dim 12
```

### 21.3 Estimator objective（official）

```text
velocity loss    = MSE(estimated velocity, successor scaled true base lin velocity)
prototype        = 每次 update 前对 proto.weight 做一次 L2 normalize（no_grad, in-place）
score_s/score_t  = normalized source/target latent @ normalized proto.weight.T
q_s/q_t          = sinkhorn(score)（no_grad，eps 0.05，iters 3）
swap loss        = -0.5 * (q_s·log_p_t + q_t·log_p_s).mean()，log_p = log_softmax(score/3.0)
total            = velocity loss + swap loss（无额外权重）
optimizer        = Adam，lr 1e-3，max_grad_norm 10.0
```

### 21.4 actor / source encoder 梯度边界

```text
HIMPolicy.get_latent   ：source encoder 前向在 torch.no_grad() 下执行
PPO optimizer 参数      = actor policy params（MLP + distribution std）+ critic params
estimator optimizer     = source encoder + target encoder + prototypes
两个 optimizer 参数交集 = 空；同一个 parameter 不会同时属于两者
```

official 把 estimator 参数也放进 PPO optimizer（靠 `no_grad` 避免被更新）；这里改为显式
分账，使「PPO optimizer 不拥有 estimator」成为可验证不变量（§17.13）。

### 21.5 Actor input / target encoder input

```text
actor input          = current frame + estimated velocity + normalized latent
                       （Black 当前 45 + 3 + 16 = 64）
target encoder input = successor frame 去掉 command + successor scaled velocity
                       （Black 当前 42 + 3 = 45）
terminal successor   = Unit 1 的 env.extras terminal pre-reset 数据（done row override）
```

official 的 critic packing trick（`next_critic_obs[:, 3:48]`）不迁移：target encoder
输入改为显式的 `next_estimator_input`，由 `HIMSpec.command_dim` 切片；
normal transition 使用 step 后的正常 successor observation，done row 使用 terminal
pre-reset 数据，不使用 reset 后新 episode observation。

```text
done row 语义（MjLab / RSL-RL）：RslRlVecEnvWrapper.dones = terminated | time_outs
两者都是 episode boundary，全部使用 terminal pre-reset override
auto_reset=False：MjLab step() 返回的就是 terminal observation，且不触发 recorder，
因此 process_env_step 直接用 obs 即为正确 terminal successor
```

### 21.6 Storage

```text
HIMRolloutStorage(RolloutStorage)
    add_transition        + next_estimator_input  [T, B, target_dim]
                          + next_estimator_velocity [T, B, velocity_dim]
    mini_batch_generator  HIMBatch，额外携带同一 batch_idx 的 successor target
```

current canonical history 不额外保存（update 时从 `observations[actor]` 现算）。
不复制 legacy `HIMRolloutStorage`，也不重复保存 actor / critic tensor。

### 21.7 Production files

```text
src/alldog_mjlab/algorithms/him/spec.py       HIMSpec / HIMInterface / canonical history / target input
src/alldog_mjlab/algorithms/him/estimator.py  HIMEstimator / sinkhorn
src/alldog_mjlab/algorithms/him/policy.py     HIMPolicy（RSL-RL actor 模型接口）
src/alldog_mjlab/algorithms/him/storage.py    HIMRolloutStorage / HIMTransition / HIMBatch
src/alldog_mjlab/algorithms/him/ppo.py        HIMPPO
```

删除 legacy prototype：`actor_critic.py`、`runner.py`（自定义 runner 明确不迁移；
`HIMActorCritic(270, 238, 45, 12)` / `HIMRunner` / ONNX dummy 270 等 Black-specific
hardcode 全部移除）。

### 21.8 本单元不做

```text
MjlabOnPolicyRunner / HIM runner integration
black-flat-him / black-rough-him task registration
PPO→HIM actor warm start
flat HIM → rough HIM resume
checkpoint / resume round-trip 完整验证
HIM TorchScript / ONNX exporter
```

------

## 22. Frozen Black HIM Runner / Checkpoint Contract

阶段状态：

```text
Black HIM runner integration（MjlabOnPolicyRunner）+ checkpoint / resume: COMPLETE
task registration / PPO→HIM warm start / flat→rough HIM resume / exporter: NOT STARTED
（task 注册与 warm start 后续完成：§23 / §24；exporter 仍未开始）
```

本单元把 §21 的算法接入现有 `MjlabOnPolicyRunner`，不新增 custom HIM runner，也不注册
task；验证通过本地 config 构造 + runner 直接实例化（与 MjLab train CLI 同路径，只是不经过
registry）。

### 22.1 Runner 调用链

```text
ManagerBasedRlEnv(black_flat_him_env_cfg)
   ↓
RslRlVecEnvWrapper
   ↓
MjlabOnPolicyRunner(env, agent_cfg_dict, log_dir, device)
   ↓
OnPolicyRunner.__init__ → HIMPPO.construct_algorithm(obs, env, cfg, device)
   ↓
HIMPolicy（含 HIMEstimator）+ MLPModel(critic) + HIMRolloutStorage
   ↓
OnPolicyRunner.learn() → alg.act / process_env_step / compute_returns / update
```

- 无 `HIMRunner` / `HIMOnPolicyRunner`；`MjlabOnPolicyRunner` 内不含任何 HIM 特判。
- `HIMPPO.construct_algorithm` 与 `PPO.construct_algorithm` 同签名，并补齐
  `share_cnn_encoders` pop 与 `rnd_cfg` / `symmetry_cfg = None`（RSL-RL 5.4.2 的
  `OnPolicyRunner.learn` / `Logger` 会直接读 `cfg["algorithm"]["rnd_cfg"]`）。

### 22.2 Config layering

```text
BLACK_CONFIG（唯一人工入口 black_config.py）
   ├── policy / algorithm / runner   ← black_ppo_runner_cfg 与 black_him_runner_cfg 共用
   └── him（HimParams）               ← HIM-only：latent_dim / encoder dims / prototypes /
                                         temperature / estimator lr / max_grad_norm
```

- `rl_cfg.py` 提供 `black_ppo_algorithm_kwargs` / `black_critic_model_cfg` /
  `black_actor_distribution_cfg`，两条路径复用同一映射，不复制 PPO 数值。
- `him_rl_cfg.py` 只声明 HIM-only：`HIMPPO` / `HIMPolicy` class、estimator 结构、
  `HimRunnerParams`（history/velocity group、command_dim、latent_dim、terminal keys）。
- HIM stage run 名：`flat_him` / `rough_him`（`load_run` 对应 `.*_flat_him$` /
  `.*_rough_him$`），与普通 PPO run 分离。

### 22.3 Observation group 路由

```text
obs_groups actor  = ["actor"]      （[B,6,45] history）
obs_groups critic = ["critic"]     （[B,259]）
estimator_velocity（[B,3]）不进入 actor / critic obs set，只被 HIMPPO.process_env_step 读取
actor normalizer = disabled；critic normalizer = enabled（与 Black PPO 一致）
```

### 22.4 Module ownership

```text
source encoder 唯一 Parameter object：HIMPolicy.estimator.encoder
HIMPolicy.state_dict() 含 estimator.{encoder,target,proto} + mlp + distribution.std_param
HIMEstimator.state_dict() 是 HIMPolicy.state_dict() 的带前缀子集（无重复 authoritative state）
checkpoint 不单独保存 estimator_state_dict，只额外保存 estimator optimizer state
inference（HIMPolicy.forward）只用 source encoder + actor MLP；target encoder / prototypes
不参与推理，也不被 policy inference 读取
```

### 22.5 Checkpoint contract

`HIMPPO.save()` = `PPO.save()`（actor_state_dict / critic_state_dict /
optimizer_state_dict）+ `estimator_optimizer_state_dict`；runner save 再补 `iter`、
`infos.env_state.common_step_counter` 与 `infos.env_state.command_curriculum`
（§4.4；由 curriculum_checkpoint 共公共层写入，PPO / HIM 同语义）。

保存 / 恢复：

```text
actor policy params（含 mlp / distribution std）          ✔
source encoder / target encoder / prototypes                ✔（在 actor_state_dict 内）
critic params                                                ✔
critic normalizer buffers                                    ✔
PPO optimizer state（含 adaptive lr 的 param_groups）        ✔
estimator optimizer state（Adam step/exp_avg/exp_avg_sq）    ✔
current learning rate                                        ✔（PPO optimizer param_groups[0]["lr"]）
training iteration                                           ✔
MjLab common_step_counter（training metadata）                ✔
command curriculum state（range / EMA / streak / buffer）     ✔（§4.4，HIM checkpoint 同样包含）
```

### 22.6 Resume 语义边界

```text
training-state continuation
    模型 / optimizer / normalizer / adaptive lr / iteration / common_step_counter

simulator trajectory continuation
    未保存完整 qpos/qvel、warp RNG、command RNG 等，不保证
    resume 后逐 step 与未中断训练 bitwise 相同
```

resume 后的环境从新的 reset state 继续；不实现 simulator serialization。

### 22.7 本单元不做

```text
black-flat-him / black-rough-him task registration
PPO→HIM actor warm start
flat HIM → rough HIM resume workflow
HIM TorchScript / ONNX exporter（HIMPolicy 尚无 as_jit / as_onnx）
multi-GPU HIM（reduce_parameters / broadcast_parameters 未按 HIM 参数分账适配；
当前项目默认单进程训练）
```

------

## 23. Frozen black-flat-him Task + PPO→HIM Warm Start Contract

阶段状态：

```text
black-flat-him registration:                        COMPLETE
PPO → HIM warm start（black-flat → black-flat-him）: COMPLETE
black-rough-him / flat→rough HIM resume:            COMPLETE（§24）
exporter:                                           NOT STARTED
```

### 23.1 Registration

```text
task_id     black-flat-him
env cfg     black_flat_him_env_cfg
play cfg    black_flat_him_env_cfg(play=True)
rl cfg      black_him_runner_cfg("flat")
runner_cls  BlackHimOnPolicyRunner
```

`black-flat` / `black-rough` 保持 `VelocityOnPolicyRunner` + `black_ppo_runner_cfg`。
`black-flat-him = black-flat + HIM observation/history/terminal contract + HIMPPO`，
未修改 reward / command / terrain / reset / termination / DR / action semantics /
control dt / joint order；Black common config 仍只有一套。

### 23.2 Training UX（真实 CLI）

```bash
# 随机初始化
uv run train black-flat-him --log-root <root> ...

# PPO → HIM warm start（初始化，不是 resume）
uv run train black-flat-him --log-root <root> \
    --agent.warm-start True \
    --agent.load-run '<black-flat run regex>' \
    --agent.load-checkpoint 'model_.*.pt'

# HIM → HIM full resume
uv run train black-flat-him --log-root <root> \
    --agent.resume True --agent.load-run '<him run regex>'
```

- `--agent.warm-start` 与 `--agent.resume` 互斥，同时使用直接报错。
- source checkpoint 定位复用 `load_run` / `load_checkpoint`，路径解析与 MjLab
  `run_train` 的 resume 路径一致（`<log_root>/<experiment_name>/<run>/<ckpt>`）。
- HIM run 写入自己的 `*_flat_him` run；source PPO run 只读。
- warm-start provenance（source task / checkpoint / 迁移维度）写入 checkpoint 的
  `infos.warm_start`。

### 23.3 Warm-start state contract

```text
state                          transfer
actor（first layer 特殊映射 + 后续层）  yes
action distribution（std）       yes
critic                         yes
critic normalizer              yes
PPO optimizer                  no
estimator（source/target/proto） no（保持新初始化）
estimator optimizer            no
iteration                      no（0）
common_step_counter            no（新 env state）
simulator state / RNG          no
command curriculum             range only（source PPO 有 curriculum state 时拷贝 vx range；
                               EMA / streak / buffer fresh；旧 PPO checkpoint 无 state 时
                               warning + config 初始 [-1,1]，warm start 不失败）（§4.4）
```

### 23.4 Actor migration

```text
PPO actor  : 45 → 512 → 256 → 128 → 12
HIM actor  : 64 → 512 → 256 → 128 → 12

first Linear : W_him[:, :45] = W_ppo；W_him[:, 45:] = 0；b_him = b_ppo
后续 Linear  : mlp.2 / mlp.4 / mlp.6 逐值复制
std          : distribution.std_param 逐值复制
```

因为新增 19 列为 0，warm-start 初始化瞬间
`HIM actor mean(history) == PPO actor mean(current_frame)`，与 source encoder 输出无关。

### 23.5 Fail-loud validation

`warm_start_from_ppo_actor` 严格校验并在不兼容时报错，不使用 `strict=False` partial load：

```text
source 缺 actor_state_dict / critic_state_dict
source 含 estimator.*（说明是 HIM checkpoint，应用 --resume）
source actor input dim != HIM current frame dim
first layer hidden dim 不一致
后续 actor / distribution state shape 不一致
critic state keys / shape 不一致
```

### 23.6 本单元不做

```text
black-rough-him registration
flat HIM → rough HIM resume
HIM TorchScript / ONNX exporter
rl_sar / quadruped_control / BlackW / symmetry / multi-GPU
长训练 / 收敛 / reward / HIM 超参调优
```

------

## 24. Frozen black-rough-him Task + Flat HIM → Rough HIM Full Resume Contract

阶段状态：

```text
black-rough-him registration:                          COMPLETE
flat HIM → rough HIM full resume:                      COMPLETE（CPU + CUDA）
rough HIM PPO→HIM warm start:                          显式不支持（fail-loud）
exporter / BlackW / 长训练收敛验证:                      NOT STARTED
```

### 24.1 Registration

```text
task_id     black-rough-him
env cfg     black_rough_him_env_cfg（= black_rough_env_cfg + HIM contract，Unit 1 已有）
play cfg    black_rough_him_env_cfg(play=True)
rl cfg      black_him_runner_cfg("rough")
runner_cls  BlackHimOnPolicyRunner（与 black-flat-him 同一个 class，无 rough-specific runner）
```

四个 Black task：`black-flat` / `black-rough`（VelocityOnPolicyRunner + black_ppo_runner_cfg）、
`black-flat-him` / `black-rough-him`（BlackHimOnPolicyRunner + black_him_runner_cfg）。
flat / rough HIM 的 runner cfg 除 stage 命名外逐字段相等（测试断言）。

### 24.2 训练路线与 CLI

```text
正式路线：black-flat PPO →（warm start）black-flat-him →（full resume）black-rough-him

# PPO → HIM warm start（仅 black-flat-him）
uv run train black-flat-him --agent.warm-start True --agent.load-run '.*_flat$' ...

# flat HIM → rough HIM full resume
uv run train black-rough-him \
    --agent.resume True \
    --agent.load-run '.*_flat_him$' [short-smoke overrides]

# rough HIM 上 warm start 显式拒绝（ HimRslRlOnPolicyRunnerCfg.warm_start_supported=False，
# BlackHimOnPolicyRunner.__init__ fail-loud）：
#   "PPO→HIM warm start currently supported only for black-flat-him. "
#   "Train/warm-start flat HIM first, then full-resume into black-rough-him."
# 不根据 checkpoint shape 自动放行。
```

`--agent.warm-start` 与 `--agent.resume` 仍互斥；checkpoint 解析复用
`get_checkpoint_path()`（load_run / load_checkpoint 正则，最新匹配）。

### 24.3 Full-resume state contract（与普通 PPO 跨 stage 完全同语义）

transfer（经继承自 `MjlabOnPolicyRunner` 的标准 `load()` 路径，无 HIM 特判）：

```text
actor MLP + distribution std                    yes
source encoder / target encoder / prototypes    yes（actor_state_dict 内）
critic + critic normalizer                      yes
PPO optimizer state（含 adaptive lr）            yes（step / exp_avg / exp_avg_sq 逐 tensor 验证）
estimator optimizer state                       yes（同上）
learning rate                                   yes
training iteration                              yes
common_step_counter                             yes（继承；与 rough env runtime state 是两回事）
```

fresh（不迁移）：

```text
flat qpos/qvel / episode / command sampler runtime / simulator state /
RNG trajectory / terrain state —— 全部不迁移；rough env 与 terrain curriculum
按 rough cfg 新建（terrain_levels 由 rough env 初始化 contract 产生）。
```

iteration 语义与普通 PPO 一致：`max_iterations` 是 additional iterations
（source iter = N 时继续 N → N + M），不是绝对上限（实测 source iter=1 → 2 iters → 3）。

### 24.4 已验证事实

```text
checkpoint parity: actor / critic / estimator / normalizer / 双 optimizer
                   （step / exp_avg / exp_avg_sq）/ lr / iteration / counter 逐项一致
model parity:      固定 synthetic obs 下 deterministic action / critic /
                   source encoder velocity+latent 的 max_abs_error = 0.0
rough env:         terrain generator + terrain_levels curriculum + OOB truncation +
                   nconmax 128 均存在；terrain levels 属于新 rough env
terrain curriculum: term 实际执行（episode 推进后 level 状态合法；极短 smoke 不伪造
                   level 数值变化的 PASS）
command:           flat HIM == rough HIM（同一 initial 范围 + §4.3 command curriculum；
                   flat HIM→rough HIM resume 时 range restore、statistics fresh）
CLI:               flat HIM checkpoint → black-rough-him full resume（CPU + CUDA）PASS
```

### 24.5 本单元不做

```text
HIM 长训练 / 收敛验证 / reward / 超参调优
rough PPO → rough HIM warm start（路线外，显式拒绝）
HIM TorchScript / ONNX exporter
rl_sar / quadruped_control / BlackW / symmetry / multi-GPU
```

------

## 25. Next Migration Order

Black flat/rough 与训练侧 sim2real contract 已完成；部署代码由独立项目接手。
下表保留完成状态与本仓库当前下一候选：

```text
1. stuck termination                     （完成）
2. reward migration                      （完成：Black flat v1 baseline = 10 项，见 §11）
3. domain randomization                  （完成：6 项，见 §12）
4. command baseline                      （完成：初始范围 + native sampler + §4.3 课程，见 §4）
5. Black flat final PPO verification     （完成：500-iteration run + 独立进程 reload + 8-command play/eval，见 §17.1）
6. Black rough PPO
   （已完成前置：rough terrain generator + terrain curriculum，见 §13；
     terrain scan + critic privileged height，见 §13.4；
     terrain-relative base-height reward，见 §11.4；
     out_of_terrain_bounds safety truncation，见 §6.4；
     sanity / baseline 训练（约 500 iteration）完成，见 §17.2；
     下一前置：无 —— 长训练 / 定量评估仍未做）
7. Black PPO deployment contract / sim2sim compatibility
   （COMPLETE：`rl_sar` 与 `quadruped_control` 的 config / trace / rollout、
     跨 runtime observation / actor / pre-safety `q_policy` 均 PASS；
     经真实硬件验证的位置限属于 deployment safety layer，最终 `q_command` 有意不同，
     见 §17.10 / §19.12。ONNX metadata 归属见 §18.1 / §19.4。）
   motor mapping / calibration、actuator / PD、IMU / orientation、
   timing / freshness / watchdog / failure 静态契约 COMPLETE（§19.14–§19.17）；
   用户已确认 IMU 安装与 legacy 配置一致。实体电机接线、标定、
   effort/current ceiling、固件安全动作及数值门槛仍待 bring-up；real backend 未实现。
   RealRobotIO v1 架构与 Unit 0–7 实施顺序已静态冻结（§19.18）；
   quadruped_control production 实施已移交到其独立项目，非本项目当前任务。
8. Black training configuration consolidation v2
   （完成：black_config.py 是单一人工入口；flat/rough 默认配置逐字段等价，
     CPU local migration verification 通过；CUDA 本轮设备不可用。）
9. Black training/tuning workflow 或用户决定下一单元。
10. Black HIM observation / history / estimator target contract
    （完成：§20。actor history [B, 6, 45] + estimator_velocity [B, 3] +
     terminal successor recorder；black-flat-him / black-rough-him 未注册。）
11. HIM model + RSL-RL algorithm integration
    （完成：§21。HIMPolicy / HIMEstimator / HIMPPO / HIMRolloutStorage；
     CPU + CUDA 真实 rollout → storage → update PASS，optimizer 分账与 terminal
     override 已验证；black-flat-him / black-rough-him 仍未注册。）
12. MjlabOnPolicyRunner integration + checkpoint / resume state
    （完成：§22。HIMPPO 接入现有 MjlabOnPolicyRunner，无 custom HIM runner；
     estimator optimizer state / adaptive lr / iteration / common_step_counter 均 round-trip；
     CPU + CUDA runner learn / save / load / resume PASS。）
13. black-flat-him registration + PPO→HIM actor warm start
    （完成：§23。black-flat-him = black-flat + HIMPPO；BlackHimOnPolicyRunner 在初始化边界
     做 actor first-layer 45→64 零填充映射 + distribution + critic + normalizer 迁移；
     CPU + CUDA CLI / runner / parity / negative cases PASS。）
14. black-rough-him registration + flat HIM → rough HIM full resume
    （完成：§24。注册 + standard load() full resume + warm-start 显式拒绝；
     CPU + CUDA checkpoint parity / model parity / runner smoke PASS。）
15. Black rough stairs terrain（up / down）
    （完成：§13.5。native box stairs + HEAD lineage proportions（7 类）；
     terrain curriculum / height_scan / OOB / reward contract 不变；
     CPU + CUDA PASS。）
16. Black HIM training validation — flat convergence + flat→rough training behavior
    （先真实确认 Black HIM 能在 flat 上稳定学习（随机初始化直训为首选，
     官方 HIMLoco 标准路线），再 full resume 到 rough 完成合理续训；
     "算法能跑" ≠ "算法训练有效"。不直接开始 exporter。）
17. Black performance-based forward-speed command curriculum + checkpoint state
    （完成：§4.3 / §4.4。super-dog Black 后期 update_command_curriculum 的移植：
     buffer 256 / EMA 0.2 / streak 2 / low 8 + high 4 / ±0.1 clip ±2.0；
     10 类 Curriculum/command/* telemetry；checkpoint infos.env_state.command_curriculum
     + none/range/full restore（same-stage full / cross-stage range / PPO→HIM range /
     旧 checkpoint none+warning）；经 §4.4 公共层同时服务 PPO/HIM；
     vy/wz/sampler 不变；play 无 curriculum。CPU + CUDA PASS。
     下一单元：Black command curriculum training validation（§1 next candidate）。）
18. Wolf robot asset / joint order / actuator contract（stage 1）
    （完成：§27。单模式完整 STL MjSpec.from_file（无 collision-only 双模式；
     该旧描述被 §27 覆盖）；16 IdealPd actuator（腿 60/2/60、轮 0/1/17）；
     显式 WOLF_POLICY_JOINT_NAMES 与轮符号 contract；损坏 imu_Link.STL 删除；
     轮径 200→160mm 后 FK 实测几何接触 root z = 0.4289、受控站立稳态 0.3773；
     CPU + CUDA smoke PASS。）
19. Wolf flat PPO / HIM / command curriculum integration
    （完成：§28。wolf-flat + wolf-flat-him 注册；actor 53-D / critic 56-D /
     action 16-D 交错 action contract；IMU 原生观测；性能驱动课程复用
     （max ±4.0 m/s，robot provenance）；PPO→HIM warm start 53→72；
     TorchScript 导出 [1,53]→[1,16] / [1,318]→[1,16]；CPU + CUDA PASS；
     长训收敛、DR、rough、sim2real 未开始。）
```

已插入完成的非 behavior 任务：

```text
configuration readability / tuning refactor v1   （完成，behavior-neutral）
Black training configuration consolidation v2    （完成，behavior-neutral）
```

进入每一步前重新检查实际代码和本文件。

不要同时推进多个 behavior unit。

------

------

## 27. Wolf Robot Asset / Joint / Actuator Contract（stage 1）

阶段状态：

```text
Wolf 机器人资产 + 默认姿态 + 关节顺序 + 执行器控制契约: COMPLETE（完整 STL 模型；CPU + CUDA）
wolf-flat task / RL task 接入:                              NOT STARTED（下一单元）
```

### 27.1 资产与 MJCF

`src/alldog_mjlab/robots/wolf/xmls/wolf.xml` 为模型权威来源，`get_spec()` 单模式
`MjSpec.from_file` 直接加载（本地 STL 完整、被 `.gitignore` 排除 Git 跟踪——已知且
有意的资产管理方式；资源缺失时 fail-loud，不做 collision-only 回退/双模式）。

```text
修改（相对原始 XML）:
    删除原 <actuator>（16 <motor>）—— 执行器全部由 MjLab IdealPdActuatorCfg 创建
    四个轮子碰撞圆柱 geom 命名：FL/FR/RL/RR_wheel_collision（只命名不改几何）
    删除损坏的 imu mesh 引用（原 imu_Link.STL 为 84 字节损坏文件，已删除；
        imu 外观在 base_link.STL 中；imu_Link body 保留作为后续 IMU 挂点）
保留   16 hinge joint + root freejoint、axis/range/transform、显式 inertial、
       碰撞/摩擦/接触参数、视觉 mesh、轮安装方式与半径
模型    nq/nv/nu = 23/22/16；ngeom 50（17 visual mesh + 33 collision）；
       总质量 33.6686 kg（= XML 显式 inertial 累加，逐位一致）
wheel   碰撞圆柱 radius 0.08 m / half-width 0.0225 m（2026-10 轮径 200→160mm，
        依据 URDF/mujoco/wolf 更新版 XML；轮 Link4 STL 已同步）；半轮距 = ±0.213 m
躯干    base_link 碰撞盒 half-size 0.215x0.10x0.06 m（2026-10 依
        URDF/mujoco/wolf 更新把 Y 半宽 0.135 → 0.10；inertial / 质量 / 视觉
        mesh 不变；模板 ncon/nefc 仍 24/96，容量 64/256 不变）
```

### 27.2 显式 joint / policy contract（wolf_constants.py，唯一 contract 来源）

```text
腿顺序      FL → FR → RL → RR；每腿 hip → thigh → calf → foot（foot=驱动轮）
16 joints   WOLF_POLICY_JOINT_NAMES（显式 tuple；编译 natural order 当前恰好一致，
            测试记录该事实，contract 不依赖它）
wheel       WOLF_WHEEL_JOINT_NAMES / WOLF_WHEEL_COLLISION_GEOM_NAMES
符号        WOLF_WHEEL_FORWARD_SIGN = FL:+1, FR:-1, RL:+1, RR:-1
            策略正轮速 = 机身 +x 前进；验证 = 默认姿态 FK 轴向（FL/RL +y、FR/RR -y，
            a×ẑ=+x̂）+ 地面闭环四轮滚动 +x（dx=+0.121 m）PASS
```

### 27.3 默认姿态与站立（完整模型 MuJoCo 实测）

```text
joint 默认姿态（rad）:
    hip 0；thigh FL/RR +0.82 / FR/RL -0.82；calf FL/RR +1.43 / FR/RL -1.43；
    foot 0。关节速度 0；root 单位四元数；root pos (0,0,0.4432)。
几何接触 root z = 0.4432 = 轮心偏移 0.3632 + 轮半径 0.08（calf=1.43 姿态 FK 实测）。
INIT_STATE pos z = 0.4432：轮最低点 ≈ 0，可直接站立。
受控站立（腿部位返 PD 至 default + 轮部速度 PD 0，2.5 s）:
    以下数值为历史记录：在首版 Kp=60/Kd=2.0、calf=1.52、root z=0.4289 旧姿态下
    实测（root z 0.4289 → 稳态 0.3773，静力沉降 ≈ 5.2 cm，thigh 静力矩 ≈ 11 N·m
    → 稳态误差 ≈ 0.23 rad；roll/pitch ≈ 0；4 轮接地；max contact force ≈ 85 N；
    max leg τ 13.9 N·m ≪ 60；无自碰撞 / 弹飞 / 持续塌陷 / 数值发散）。
    当前 Kp=80/Kd=3.0（用户 2026-10 调参轨迹：60/2.0 → 80/3.0 → 50/1.2 →
    80/3.0）+ calf=1.43 姿态下未重跑受控站立沉降记录（待下轮验证记录补充）。
```

### 27.4 执行器 contract

```text
全部 MjLab v1.6.0 原生 IdealPdActuatorCfg，逐关节 16 个（sort_actuators=True）：
    hip/thigh/calf  position PD  Kp=80, Kd=3.0,  effort_limit=60 N·m
                    （腿部 PD 2026-10 用户调参轨迹：60/2.0 → 80/3.0 → 50/1.2 →
                     80/3.0，当前实际值 Kp=80/Kd=3.0；实机电机规格仍待确认；
                     dq_target=0）
    foot（轮）       velocity PD  Kp=0,  Kd=1.0,  effort_limit=17 N·m
                    （不使用 torque action / XML velocity servo / 自定义 actuator）
控制语义 = mjlaw τ = clamp(Kp(pos_t−q)+Kd(vel_t−dq)+effort, ±limit)：
    轮部 Kp=0 ⇒ 静止位置误差不产生轮矩；正负 velocity target 方向、
    ±17 / ±60 限幅均经 mjlaw 数值探针 PASS。
    编译后 actuator/ctrl 顺序与 policy action order 是两个概念；下一阶段
    JointPositionActionCfg（leg scale 0.25，2026-10 调参）与 JointVelocityActionCfg
    （wheel scale 10 rad/s）负责显式 mapping，本轮未实现。
```

### 27.5 Wolf IMU observation contract（本单元冻结）

`imu_Link` 正式作为 IMU 安装坐标系（body 保留在 base_link 下，pos (0,0,0.05975)、
单位四元数、无 joint / actuator / inertial；本负载 IMU 外观在 base_link.STL 中，
损坏的 imu_Link.STL 已删除）。

```text
site      imu_site（pos 0,0,0 / quat 单位，与 imu_Link 原点重合；尺寸 0.005）
sensor    imu_ang_vel（gyro @ imu_site，IMU 局部系角速度，rad/s）
          imu_upvector（framezaxis objtype=body objname=world reftype=site；
          世界 ẑ 在 IMU 系的表达）
访问      scene["robot/imu_ang_vel"] / scene["robot/imu_upvector"]
          （MjLab v1.6 Scene 自动把 XML 原生 sensor 包为 BuiltinSensor，
          不重复注册 BuiltinSensorCfg）
```

冻结的 wolf-flat Actor observation 语义（term 实现留到 task 单元）：

```text
base_ang_vel       = mdp.builtin_sensor(robot/imu_ang_vel)          [3] rad/s
projected_gravity  = mdp.projected_gravity_from_sensor(robot/imu_upvector)
                   = -imu_upvector = R_world_imu.T @ (0,0,-1)      [3] 单位向量
```

```text
实测（完整模型，MuJoCo/MJWarp）:
    静止水平           gyro=0、projected_gravity=(0,0,-1)         PASS
    5 组 roll/pitch/yaw 倾斜    framezaxis == R.T ẑ（独立坐标变换对照，
                      误差 <1e-9）                              PASS
    非零角速度        MuJoCo freejoint qvel[3:6] 为 body-local 语义（实测确定），
                      gyro == qvel[3:6]（site 与 base 平行）      PASS
    root-based 对照   site 轴与 base 平行 ⇒ gyro ==
                      root_link_ang_vel_b、-upvector ==
                      projected_gravity_b（max err = 0.0）        PASS
    Scene 端到端      CPU + CUDA 均通过（含 projected_gravity_from_sensor
                      的 -upvector 符号）                        PASS
不变量    nq/nv/nu、总质量 33.6686、base 惯性未被修改；Actor 仍为 53-D，
          无新增 observation 维度。
待确认    XML 当前假定 IMU 三轴与 base 平行；实机安装朝向核对前不能视为
          硬件验证完成。今后若调整安装朝向，Actor 仍消费 IMU 坐标系测量值。
```

### 27.6 验证记录（tests/check_wolf_robot.py、tests/check_wolf_imu.py，不提交）

```text
CPU  PASS（静态 / 编译 / joint+actuator mapping / PD law 探针 /
     wheel forward direction FK + 闭环 / 受控站立 / INIT 0.45 站立）
CUDA PASS（MJWarp Simulation 构建 + 默认姿态 20 步 finite）
git diff --check: clean
IMU（check_wolf_imu.py）: 静态 / 静止读数 / 5 组倾斜 / 非零角速度 /
    root-based 对照（err=0.0）/ Scene 端到端 CPU + CUDA 全部 PASS
```

### 27.7 待确认硬件参数与下一单元

```text
待确认   腿部 80/3/60 与轮部 17 N·m 的实机电机规格；实机 IMU 安装位置
下一单元：Wolf flat 长训验证（真实收敛评估，"能跑" ≠ "训练有效"）；
         DR（旧 BlackW 20+ 项，逐项开关）与 rough / sim2real 之后的阶段未开始。
```

## 28. Wolf Flat PPO / HIM / Command Curriculum Task（本轮集成，COMPLETE）

一次性集成单元：wolf-flat（普通 PPO）与 wolf-flat-him（HIM）注册为完整训练任务，
复用 Black 已验证的基础设施（HIM 算法层零修改、PPO/HIM 算法零修改）。长训收敛、
DR、rough、sim2real 未开始，不得声称完成。

### 28.1 注册与实现布局

```text
src/alldog_mjlab/tasks/velocity/wolf/
    __init__.py           任务注册（wolf-flat / wolf-flat-him）
    wolf_config.py        单一人工训练参数入口（镜像 black_config 风格）
    env_cfgs.py           term 顺序 contract / selector 绑定 / sensor 装配
    observations.py       signed wheel velocity（forward sign）
    rewards.py            leg / wheel action rate 分组（16-D contract）
    rl_cfg.py             wolf_ppo_runner_cfg / wolf_him_runner_cfg（数值来自 WOLF_CONFIG）
    him_runner.py         WolfHimOnPolicyRunner（仅 override task 标识）

复用（未复制的 Black 公共件）：
    black/rewards.py      tracking / 罚项公式（机器人无关）
    black/curriculums.py  ForwardSpeedCommandCurriculum（参数注入化）
    black/curriculum_checkpoint.py  checkpoint 状态与 restore 模式
    black/him.py          HIM observation/history/terminal contract（group 名 = 算法接口）
    black/rl_cfg.py 助手   PPO/HIM cfg 构造（config 注入参数化）
    black/him_runner.py   HIM runner（task 标识类属性化）

runner_cls：
    wolf-flat      VelocityCommandCurriculumRunner（原 BlackVelocityOnPolicyRunner
                   改名，robot-agnostic；旧名保留为兼容别名）
    wolf-flat-him  WolfHimOnPolicyRunner（BlackHimOnPolicyRunner 子类 +
                   WARM_START_SOURCE_TASK = "wolf-flat"、ROBOT = "wolf"）
```

### 28.2 Frozen contracts

```text
action 16-D（8 个原生 action term 交错，FL→FR→RL→RR 每腿 3+1）:
    [FL_hip FL_thigh FL_calf FL_wheel | FR ... | RL ... | RR ...]
    leg  = JointPositionActionCfg  q_target = q_default + 0.25 * raw（2026-10 0.20→0.25）
    wheel= JointVelocityActionCfg  dq_target = sign * 10.0 * raw
          sign = FL +1 / FR -1 / RL +1 / RR -1（robots/wolf 冻结值）
    无默认 [-1,1] clip（Gaussian init std 1.0）

actor 53-D（单帧）:
    command 3 − base_ang_vel 3（robot/imu_ang_vel）− projected_gravity 3
    （robot/imu_upvector = -upvector）− leg_joint_pos_rel 12（policy 顺序，
    不 biased）− leg_joint_vel 12 − signed_wheel_vel 4（乘 forward sign）−
    last_raw_action 16
    scale: command (2,2,0.25) / ang_vel 0.25 / gravity 1.0 / joint_pos 1.0 /
    joint_vel 0.05 / wheel_vel 0.05 / last_action 1.0
    noise（raw）：ang_vel ±0.3 / gravity ±0.05 / joint_pos ±0.08 /
    joint_vel ±2.0（wheel 同档）；command / last_action 无噪声

critic 56-D = actor 53-D 同布局无噪声 + 末尾 3-D 真实 body-frame base_lin_vel
    （scale 1.0；obs_normalization True）

HIM：
    actor history [B, 6, 53]（MjLab 原生 oldest→newest）
    source encoder input 318 / target encoder input 53（frame 去 command + scaled
    base_lin_vel ×2.0）/ estimated velocity 3 / latent 16 / actor input 72 /
    action 16 / critic 56
    部署 canonical history = newest→oldest flatten（复用 canonical_history()）

command（2026-10 收窄 Y / Yaw 初值，改善跟踪起步）：
    vx 初始 [-1,1] / 课程 max ±4.0（仅 vx 扩展，步长 0.1；无 Y/Yaw 课程）
    vy [-0.2,0.2] / yaw [-0.8,0.8] / resample 10 s
    standing 0.1 / forward-only 0.2 / world 0 / heading 关闭

reward（四任务同一套 14 项，v2 起分轴 tracking；dict 顺序 = logging 顺序）:
    track_linear_velocity_x +1.0（X 轴单独指数项；课程评分来源）/
    track_linear_velocity_y +1.0（Y 轴单独指数项，v2 拆分，原 XY 合并项废弃）/
    track_angular +1.0（用户 2026-10 调参，0.5 → 1.0）/
    lin_vel_z -1.0 / body_ang_vel -0.05 / base_height -2.0（target 0.40 m）/
    leg_action_rate -0.02 / wheel_action_rate -0.005（2026-10 调参）/
    sigma 0.25（分轴后
    单轴误差平方除以 sigma，body-frame 来源不变）
    leg / wheel action rate 按 16-D contract 显式分组（leg = 12 位置通道、
    wheel = 3/7/11/15），不整条 16 维求和
    姿态（§28.5，四任务统一）：orientation L1 -1.5（v2 重命名，原 key
    "upright" / scale "upright" / func base_orientation_l1；公式不变）+
    hip_default -0.50（2026-10 由 -0.30 调参）/ stand_still -0.40 / run_still -0.20 /
    dof_pos_limits -0.20 / leg_torques -0.0001
    dof_pos_limits / leg_torques 为 native 薄包装（rewards.dof_pos_limits /
    rewards.leg_torques_l2 转发 mjlab.envs.mdp，不复制数学；全部 14 项 func
    入口在 rewards.py，由 env_cfgs 注册）
    base_height func 差异：flat = world-z / rough = terrain-relative
    （terrain_scan footprint 均值）；target / weight / key / 顺序一致
curriculum state v2（2026-10，§4.3 Wolf 侧同步）：state 新增
    ``tracking_reward_term``（= "track_linear_velocity_x"）；评分公式不变
    （仍除以 |weight|×dt×steps，消除 weight/dt 影响）；恢复兼容：v2→v2 full
    校验评分来源一致，旧 v1 state 只允许 range（full fail-loud），
    auto 下 v1 自动降级为 range 并打 warning

termination：time_out（20 s，time_out=True）+ illegal_contact
    （base_link 对 terrain，力阈值 1.0 N，history 4 substeps）；
    无倾角终止、无 stuck、无 OOB。play 下 illegal_contact 移除。

reset：root pose 不随机（default 站立 z=0.4432）；root 速度六轴小幅扰动；
    hip/thigh/calf 小幅 offset（±0.3 内，thigh/calf 非零）；轮子零位零速。
    无 DR 事件（nominal dynamics）。

simulation：timestep 0.005 × decimation 4 = policy 50 Hz；
    容量历史：128/512 曾被穿地模板（qpos0 root z=0，ncon=124/nefc=496）钉住；
    2026-10 模板抬高后（§31）flat 容量改为 64/256（运行时需求候选值；
    运行时实测 nefc ≈ 8–36，overflow 未出现，需长训监控）
```

### 28.3 Command curriculum（复用 + provenance）

ForwardSpeedCommandCurriculum（§4.3 状态机不变）参数化改造：CurriculumTermCfg.params
新增 `curriculum_params`（Wolf 传 WOLF_CONFIG，Black 仍默认读 BLACK_CONFIG）与
`robot` provenance；checkpoint state 追加 `robot` 字段（旧 v1 checkpoint 无该字段
时仅 warning）。恢复拒绝跨机器人（black ↔ wolf 双向 fail-loud）；PPO→HIM warm
start 固定 range；同 stage resume auto→full。Black 数值 / checkpoint 格式不变
（Black builder 显式传 robot="black"，行为等价）。

### 28.4 Warm start / resume / export

```text
wolf-flat-him:
    随机初始化直接训练              PASS（CPU + CUDA 真实参数更新）
    --agent.warm-start True         PASS（wolf-flat PPO → HIM：53→72，
          拷贝 53 列 / 补零 19 列；critic / distribution 迁移；estimator 独立初化）
    --agent.resume True             PASS（full restore + curriculum state 日志）
    Black → Wolf warm start         fail-loud（双保险：45 != 53 维度 + robot provenance）
    Wolf → Black（black-flat 载入 wolf checkpoint）  fail-loud（robot mismatch）

export（RSL-RL 原生 as_jit + reload 数值等价，max_abs_diff = 0.0）:
    wolf-flat      input [1, 53]   output [1, 16]
    wolf-flat-him  input [1, 318]  output [1, 16]
    Black 回归：[1,45]→[1,12] / [1,270]→[1,12] 不变（export 自身已有 Black 检查）
    exporter 已 task-aware 化：frame / history / action 维度从 env
    observation_manager.group_obs_dim + action_manager 推导，并与 policy 网络结构
    交叉校验（不硬编码 45/12；Black contract 值仍由 env 推导得 45/12 验证）。
    默认输出 <run>/exported/policy.pt；--task-id / --load-run / --checkpoint /
    --output-dir 全部保留。
```

### 28.5 Wolf 姿态奖励 v1/v2（用户迭代，2026-10；v1 曾仅 flat，v2 起四任务统一）

针对姿态扭曲 / 关节长期偏离 default 的增强（BlackW blackW_config/env.py
经验迁移；历史参考 super-dog 本地 HIMLoco，非官方 InternRobotics/HIMLoco）。
v2 范围修正（有意修改）：取消「flat 专属」限制，Rough 也切换为同一套 14 项（tracking 分轴 + orientation 重命名后与 flat 完全一致）；
Rough 原 8 项基线（upright native L2 -0.5、无姿态项）已废弃，**Rough 奖励
配置变更后的训练效果未验证**（长时间训练 / 收敛均未评估）。

```text
任务范围：wolf-flat / wolf-flat-him / wolf-rough / wolf-rough-him 同一套 14 项
    （key / weight / 顺序完全一致）；唯一 flat/rough 差异是 base_height func
    （flat = world-z / rough = terrain-relative footprint 均值，terrain_scan）。
upright（key 不变）：L1 公式
        |projected_gravity_x| + |projected_gravity_y|，weight -1.5，四任务统一
    （rewards.base_orientation_l1；BlackW `_reward_orientation` 的无地形自适应
    版）；rough 原 native flat_orientation_l2 / -0.5 分支已取消。
hip_default（新增，-0.30）：四腿 hip 相对 default 的 L1 偏差 × 指令衰减
        alpha = clamp(1 - 0.35*min(|cmd_y|/0.5, 1) - 0.35*min(|cmd_yaw|/1.0, 1),
                      0.5, 1)
    （rewards.hip_default_l1；BlackW `_reward_hip_default` 同构；参数在
    WOLF_CONFIG.reward.posture：hip_y_ref=0.5 / hip_yaw_ref=1.0 /
    hip_y_scale=0.35 / hip_yaw_scale=0.35 / hip_min_scale=0.5）。
stand_still（新增，-0.40）：静止门控的 12 腿关节（hip/thigh/calf，显式排除
    轮）回中 L1；门控 = norm(cmd_xy)<0.1 且 |cmd_yaw|<0.1 同时满足
    （rewards.stand_still_leg_l1；BlackW `_reward_stand_still` 同构）。
    v2 起 rough 同样启用（下同，姿态 5 项均四任务共用）。
dof_pos_limits（新增，-0.20）：native mdp.joint_pos_limits，SceneEntityCfg
    显式 12 腿关节；soft limit = 现有机器人 soft_joint_pos_limit_factor=0.9
    （不改 MJCF 限位；轮关节不参与位置限位奖励）。
leg_torques（新增，-0.0001）：native mdp.joint_torques_l2，SceneEntityCfg
    显式 12 腿 actuator（actuator 名 = 目标关节名，create_motor_actuator
    name=joint_name；实测解析结果 FL_hip…RR_calf 12 个，无轮）。
关节 / actuator 选择全部显式名称（WOLF_LEG_JOINT_NAMES /
    WOLF_LEGS_FL_FIRST_JOINT_NAMES），不依赖 MJCF natural order。
不变：track_linear_velocity_x +1.0 / track_linear_velocity_y +1.0 /
    track_angular_velocity +1.0（用户调参）/ lin_vel_z -1.0 /
    body_ang_vel -0.05 / base_height -2.0 / leg_action_rate -0.02 /
    wheel_action_rate -0.005；不新增足端周期 / 强制抬轮 / 轮速差 / 接触髋惩罚 /
    固定步态约束（run_still 为 BlackW 已验证的直线行走回中门控项，v2 加入）。
高度目标：base_height_target 用户曾试调 0.45（与 flat 模板站立高度一致）后
    恢复 0.40 m（2026-10；Wolf 完整模型站立稳态实测 0.3961 m；历史首版即 0.40）。
run_still（v2 增补，-0.20）：12 腿关节（hip/thigh/calf，显式排除轮）回中 L1 ×
    门控（|cmd_x| > 0.1 严格大于 & |cmd_y| < 0.1 & |cmd_yaw| < 0.15 严格小于；
    BlackW `_reward_run_still` 同构，legacy 权重 -1.0，本轮取 -0.20）；
    与 stand_still 共享同一 12 腿 SceneEntityCfg 选择器（rewards.run_still_leg_l1；
    参数在 WOLF_CONFIG.reward.posture：run_still_x/y/yaw_threshold = 0.1/0.1/0.15）。
命名统一（v2）：reward key "orientation"（原 "upright"）/ scale
    RewardScales.orientation（原 "upright"）/ func rewards.orientation_l1
    （原 base_orientation_l1）；权重 -1.5 与公式不变；仅影响 TensorBoard tag，
    不影响 policy I/O。
奖励入口梳理（v2）：全部 14 项 reward func 均定义在 rewards.py；
    dof_pos_limits / leg_torques_l2 为 native 薄包装（转发
    mjlab.envs.mdp.joint_pos_limits / joint_torques_l2，不复制数学）；
    wolf_config.py 管参数 / rewards.py 管函数 / env_cfgs.py 管装配。
课程评分来源（v2）：curriculum params 的 reward_term_name 与
    WOLF_TRACKING_VELOCITY_REWARD_TERM 改为 "track_linear_velocity_x"；
    评分公式不变（episode_sum / (steps × dt × |weight|)，消除 weight/dt 影响）。
训练效果（长训收敛 / 是否真正抑制姿态扭曲）未验证 —— 明确 NOT RUN：
    完整 PPO/HIM 训练与数百 iteration 收敛评估。
```

### 28.6 验证记录（tests/check_wolf_task.py，不提交）

```text
CPU  PASS（cfg static × task × play / action→target 数值映射（FPS+wheel sign）/
     IMU sensor 数据 / signed wheel velocity 对照 / rollout 有限性 /
     provenance fail-loud / CLI：3+2 iter 真实更新、resume full、warm start 53→72、
     export 0.0e+00）
CUDA PASS（runtime 段 GPU 重跑 + 1024 env × 3 iter 真实训练 mean_reward
     上升；wolf-flat / wolf-flat-him 均通过）
Black 回归：check_black_flat（CPU+CUDA）/ rough（CPU+CUDA）/ rough_him /
     command_curriculum / him / him_algo / him_runner / him_warm_start 全部 PASS。
未验证：长训收敛 / 高速（±4）行为 / DR / rough / sim2real —— 均未开始。

姿态奖励 v1/v2 补充（tests/check_wolf_posture_rewards.py，不提交）：
CPU PASS（四任务注册/权重（v2 统一 14 项：分轴 tracking_x/y + orientation
     L1 -1.5 + 姿态项含 run_still；含关键分轴数值检查）；四任务 base_height target = 0.40；runtime 数学：
     default 零偏差 = 0、±0.1 偏差对称、alpha 用例 1.0/0.65/0.65/0.50/0.895
     全中（范围 [0.5,1.0]）、stand_still 双门控与纯 yaw 关门、
     run_still 门控（±0.11 开 / x=0.1 关 / y=0.1 关 / yaw=0.15 关 / yaw=0.14
     开 / 静止与横移与纯 yaw 关、正负对称、零偏差=0）、
     dof_pos_limits 超限为正/限内为 0、leg_torques == 12 腿 actuator_force 平方和、
     upright == sum|gravity_xy|；选中关节/actuator：hip 4 / 腿关节 12 /
     腿 actuator 12 全部无轮（actuator 名 = 关节名显式解析）；
     compute(dt) == Σ raw×weight×dt；PPO 53/56/16 与 HIM [B,6,53]+[B,3] 不变；
     reward 输出 [num_envs] finite）
受控站立沉降实测（Kp=50/Kd=1.2，单环境 500 步）：root z 0.4432 → 稳态下限
     0.3775，沉降 ≈ 6.6 cm（历史参考：Kp=60/Kd=2.0 首版为 0.3773/5.2 cm）——
     仅测量报告，base_height_target 未改。
v2 统一补充：四任务 14 项同序同权重；rough runtime 重验（base_height
     terrain-relative 接口 terrain_scan / target 0.40、选关节 12/actuator 12
     无轮、14 项 compute [num_envs] finite、PPO shape 53/56 不变）；
     rough 原 8 项基线巳废弃，**rough 新奖励配置的训练效果未验证**。
回归：check_wolf_task / check_wolf_rough / check_wolf_dr / check_wolf_robot /
     check_wolf_template_capacity 全部 PASS（CPU）。
NOT RUN：完整 PPO/HIM 训练（长训收敛与姿态效果未验证）、CUDA 矩阵。
```

## 29. Wolf Domain Randomization（本轮集成，COMPLETE）

一次性集成单元：Wolf flat PPO / HIM 共用的 Domain Randomization，全部在
`wolf_config.DomainRandomizationParams`（默认全关）逐项开关；play 强制全关；
minimal 训练 profile = `wolf_config.MINIMAL_DR`（轮地摩擦 + 腿部 PD gains）。
PPO 与 HIM 同一 DR 配置。Actor/critic observation layout（53/56）、16-D action
layout、raw action 语义均未修改。

### 29.1 实现位置

```text
src/alldog_mjlab/tasks/velocity/wolf/randomization.py
    自定义 DR 实现：轮摩擦乘子复合、body inertia scale、腿 PD×motor strength
    组合、calf backlash play 模式 action term、轮 target scale/bias action term、
    轮 obs bias class obs term、wheel radius+初始高度补偿 event、
    multiplicative initial joint pos reset
src/alldog_mjlab/tasks/velocity/wolf/env_cfgs.py
    DR 事件表（dict 顺序 = 同 mode 内应用顺序；wheel friction 依赖 ground
    friction 顺序敏感）、actuator delay 注入（_wolf_robot_cfg）、critic
    wheel_vel term 显式去 bias
src/alldog_mjlab/tasks/velocity/wolf/wolf_config.py
    DomainRandomizationParams / CalfBacklashParams / WheelTargetBiasParams /
    WheelObsBiasParams / MINIMAL_DR
```

原生优先使用：payload/com 用 `dr.body_mass(add)` /「dr.body_com_offset(add)」；
link/wheel mass 用 `dr.body_mass(scale)`；push 用 native `push_by_setting_velocity`；
disturbance 用 native `apply_external_force_torque`；delay 用原生 actuator
`delay_min_lag/delay_max_lag`（"3/4 policy step" → ×decimation 物理 step）；
wheel radius 用 `dr.geom_size(scale)`（写后自动刷新 rbound/aabb）。
轮地摩擦为自定义实现（原生 `dr.geom_friction` 无法处理 pair max() 下限，见 §29.3）：
`randomize_ground_friction`（per-env 标量写全部机器人 geom + 地面 geom 压 0）+
`randomize_wheel_friction_multiplier`（读当前值 × per-env 共享乘子）。

### 29.2 遷移对照表（旧 BlackW → Wolf）

| Legacy feature | 旧版当前实际行为 | Wolf 实现 | MjLab 表达 | 验证结果 |
|---|---|---|---|---|
| base payload mass [-1,2]kg | 每 reset 重采样，add 到 base mass | IMPLEMENTED | dr.body_mass(add) | bounds PASS |
| base COM offset ±0.05 | add 到 base COM | IMPLEMENTED | dr.body_com_offset(add) | bounds PASS |
| link mass [0.9,1.1] | 每 body 独立采样，“自 body 起”均值 | IMPLEMENTED（base body 除外） | dr.body_mass(scale) | PASS |
| friction [0.25,1.25] | 单系数套用 actor 全部 shape | IMPLEMENTED | 自定义 randomize_ground_friction（per-env 标量写全部机器人 geom；地面 geom 压 0） | bounds + contact 级 PASS |
| wheel friction ×[0.4,1.0] | base×scale，后乘（per-env 共享，四轮同值） | IMPLEMENTED | 自定义复合（读当前值再乘 per-env 共享乘子；地面 geom 压 0） | contact 级 PASS（contact == 轮 geom，min 0.142 < 1.0） |
| restitution [0,0.1] | Isaac Gym shape restitution | UNSUPPORTED（未迁移） | v1.6 无可靠恢复系数机制（solref 不等价） | — |
| motor strength [0.9,1.1] | 终端力矩乘因子，"clip 后语义 = 栱增益缩放" | IMPLEMENTED | 自定义：合成进 IdealPd kp/kd | bounds PASS |
| hip motor strength [0.8,1.05] | per-hip 因子，全局后乘 | IMPLEMENTED | same（仅 hip actuator 第二因子） | bounds PASS |
| hip passive damping ×[0.8,1.5] | URDF dof damping=0，乘子无效 | LEGACY INERT（不迁移） | Wolf XML damping=0，base 无意义 | 静态分析 |
| calf backlash play 模式 | effective target + Kp 弱化（每 substep 状态机） | IMPLEMENTED | action term 状态机，写入 q+kp_scale×(eff−q) | 状态机逐项复算 PASS |
| Kp/Kd ×[0.9,1.1] | per-env 标量因子（全腿共享） | IMPLEMENTED（同语义 per-env 标量） | 自定义复合（相对 default） | bounds PASS |
| initial joint pos ×[0.5,1.5] | 旧 base 美人默认 reset（与开关无关，恒为乘法） | IMPLEMENTED（开启时替换 leg offset reset；基线 offset reset 保留） | 自定义 reset_joints_by_default_scale | bounds PASS |
| inertia ×[0.9,1.1]（含 base） | per-body per-axis 独立缩放 principal axes | IMPLEMENTED（间题：Wolf 关闭时与 DR-on 均为 body_inertia 直接缩放；包含 base+links） | 自定义 randomize_body_inertia_scale | bounds PASS |
| disturbance ±30N / 8 步 | body local 系恒力，8 步重采样 | NOT EQUIVALENT（有意） | native apply_external_force_torque（world 系；均匀分布统计等价）；reset 自动清零 | 写入/清零 PASS |
| push 15s / ±1m/s | root xy 速度**赋值**（非增量） | NOT EQUIVALENT（与 Black MjLab 版先例一致） | native push_by_setting_velocity（增量） | 冒烟 rollout |
| leg delay ≤3 policy step | 每 reset 固定 lag∈{0..2}（旧 base 实际 randint(0,3)；config value 未用） | NOT EQUIVALENT（有意） | native actuator delay（0..12 physics step，≤每 policy step 重采样；episode 初 buffer 回填无延迟待填满） | stepdown 测量 PASS + 配置 PASS |
| wheel delay ≤4 policy step | 每 reset 固定 lag∈{0..4} | NOT EQUIVALENT（同上） | 同上（0..16 physics step） | 同上 |
| wheel motor strength [0.8,1.2] | wheel τ 乘因子（Kd 通道） | IMPLEMENTED | 轮 actuator Kd 缩放（Kp 恒 0） | bounds PASS（无 stiffness 注入） |
| wheel vel ref scale [0.9,1.1] | per-wheel，乘 target | IMPLEMENTED | action term target scaling | dq_target=2×10+3 精确 PASS |
| wheel vel ref bias ±0.3 | per-wheel，加到 ref | IMPLEMENTED | action term target bias | 同上 PASS |
| wheel vel obs bias ±0.5 | per-wheel，加观测 | IMPLEMENTED | actor-only class obs term（critic 显式去 bias） | critic==真实值 PASS;actor==真实+bias±noise PASS |
| wheel mass [0.9,1.1] | wheel body mass×scale | IMPLEMENTED | dr.body_mass(scale)（wheel body） | bounds PASS |
| wheel inertia [0.8,1.2] | wheel body inertia×scale | IMPLEMENTED | 自定义 randomize_body_inertia_scale(wheel) | bounds PASS |
| wheel radius ×[0.9,1.1] | per-env 固定采样（startup） | NOT EQUIVALENT（有意：per-episode 重采样） | dr.geom_size(scale)+root pose Z 读改补偿（保留 XY/quat/velocity） | r=0.12 静置抬高 PASS；rbound 一致 PASS；root XY/quat/velocity 保留 PASS |
| wheel base half width ×[0.95,1.05] | learned 轮速模式未使用该参数 | LEGACY INERT（不迁移） | Wolf learned 轮速无前馈半轮距项 | 静态分析 |

### 29.3 关键实现约束（已冻结）

- MuJoCo 接触摩擦按 geom pair 取 **max** 结合：地面 geom 默认 1.0 会把
  wheel-ground contact 摩擦下限抬高到 1.0（采样范围低段与乘子结果不可达）。
  摩擦事件因此把地面 geom（plane terrain 的 `terrain` geom）切向摩擦压 0，
  contact 摩擦 = 机器人 geom 值：
  - `dr_ground_friction`：per-env 单标量绝对值写全部机器人 collision geom
    （旧版单系数语义），同时压地面；
  - `dr_wheel_friction`：轮 geom = 前序事件写入的当前值 × per-env 共享乘子
    （旧 BlackW 语义：每环境单标量，四轮同值，非逐轮独立采样），同时压地面。
  两事件均要求 plane terrain（Wolf flat）；摩擦 DR 全关时地面/机器人 geom
  均为 nominal 默认值，行为不变。中性 DR（range=(1,1)）contact 摩擦 ==
  nominal 1.0（已验证）。
- 轮半径高度补偿只读改当前 root pose 的 Z（`z += r_new − r_nominal`），
  XY / quaternion 原样保留、velocity（qvel）不触碰（直接读改 sim qpos；
  derived kinematics 在 reset 事件阶段未经 forward，不可用 root_link_pose_w）。
- 所有质量/COM/惯量/gain 条目相对 compile-time default 建立（“scale/add” relative
  default），关关开开连续 reset 无漂移（验证：5 次 reset 分布同区间）。
- 腿/轮 delay 融入 IdealPdActuator 原生 delay（腿≤12、轮≤16 physics step；
  `update_period=decimation`，`per_env_phase=False`）；关闭时无 buffer、无残余。
- calf backlash 关闭时用原生 JointPositionActionCfg（无状态机实例）；轮 target
  scale/bias 关闭时同理；`MINIMAL_DR` = ground_friction+Kp+Kd。
- 站立契约（用户确认 2026-10，已提交）：腿部 PD Kp=80/Kd=3、默认 calf 角
  ±1.43 rad、`WOLF_DEFAULT_ROOT_Z = 0.4432`（= calf=1.43 下 FK 轮心偏移
  0.3632 + 轮半径 0.08；旧值 0.4289 为 calf=1.52 旧姿态几何接触高度，弃用）。

## 30. Wolf Task Independence + Wolf Rough PPO/HIM（本轮集成，COMPLETE）

一次性单元：Wolf 任务实现完全脱离 `tasks.velocity.black.*` import（独立原实现，
不建共享包装层），并注册 wolf-rough / wolf-rough-him。flat 的 PPO/HIM、课程、DR、
checkpoint、warm start、export 契约全部保留（已有 wolf-flat 旧 checkpoint 直接可用）。
不改通用 HIM 算法（`algorithms/him/*` 零修改）、不改 Black 生产代码。

### 30.1 Task independence

```text
新增 Wolf 本地文件（独立同构实现，公式 / 状态机语义与 Black 已验证版本一致）：
    curriculums.py           WolfForwardSpeedCommandCurriculum + state 序列化
                             （version/stage/robot/vx/EMA/streak/buffer 字段与旧
                              Wolf checkpoint 完全一致；CURRICULUM_STATE_VERSION=1）
    curriculum_checkpoint.py mixin + WolfVelocityCommandCurriculumRunner（ROBOT="wolf"）
    him.py                   HIM task-side contract（group 名 / extras key / 6 帧布局
                              不变；[B,6,53] oldest→newest；canonical newest→oldest）
    terrain.py               WolfRoughSlopeTerrainCfg + wolf_rough_terrain_generator_cfg
    rewards.py               追加 5 个 tracking/罚项公式 + base_height_l2_terrain
                             （35-ray footprint；Black 未用项不复制）
重写：rl_cfg.py（WolfRslRlOnPolicyRunnerCfg / WolfHim*Cfg；字段值不变）、
       him_runner.py（WolfHimOnPolicyRunner，warm start 逻辑同构）、__init__.py
Wolf config：CommandCurriculumParams 改为 Wolf 本地 dataclass（不再 import
     black.black_config）；新增 TerrainParams；runner 增加 rough / rough_him
     StageRunnerParams（run_name/load_run 与 flat 同风格）。
```

Provenance 修复：旧注册把 wolf-flat PPO 挂在 Black 目录 runner（类属性
`ROBOT="black"`），与 checkpoint state `robot="wolf"` 不一致；现在
`WolfVelocityCommandCurriculumRunner.ROBOT = "wolf"`；provenance 校验（跨 robot
fail-loud）保持开启，不关闭。

### 30.2 Wolf rough terrain / 环境（首版候选数值）

```text
terrain generator（curriculum=True，7 类 x 10 行，值与 Black rough 已实现版一致；
仅数值复制，不运行时引用 BLACK_CONFIG）：
    patch 8x8m / border 20m / horizontal 0.1 / vertical 0.005 / platform 3.0m
    proportions: flat 0.10 / up 0.05 / down 0.05 / rough_slope 0.10 /
                 obstacles 0.20 / stairs_up 0.25 / stairs_down 0.25
    difficulty [0,0.9]；slope [0,0.7]；rough noise base 0.015 / gain 0.1 /
    step 0.005 / downsample 0.2；obstacles [0.06,0.26]m x [1,2]m x 20；
    stairs base 0.05 / gain 0.18 / width 0.30；max_init_terrain_level 5
rough_slope = WolfRoughSlopeTerrainCfg（slope+noise 双 native 数学相加，同 Black
     已验证实现语义）。
MJWarp capacity：rough 模板 spawn 高度 z=10 后模板 ncon=0 / nefc=0（原
     178/712 下限消除，见 §32）；rough_nconmax=128 / rough_njmax=256
     （候选 B，runtime 峰值余量 ~4x，待 4096 实训回验，overflow 需保持 NO）。
```

rough 环境特化（与 flat 共用全部基础控制/observation/action/reset/DR 契约）：

- terrain_scan：Wolf 本地 RayCastSensorCfg（base_link frame / yaw 对齐 /
  GridPatternCfg 1.6x1.0 @0.1 = 17x11 = 187 rays / max 5.0m / geom group 0）；
  仅 rough 注册，flat 没有；
- base_height reward：flat = world-z（base_height_l2_flat），rough = terrain_scan
  中央 7x5=35 rays clearance 均值（base_height_l2_terrain）；key/weight/顺序不变，
  高度目标仍读 WOLF_CONFIG.reward.base_height_target；
- termination：+ native out_of_terrain_bounds（time_out=True）；play 两项均移除；
- curriculum 任：rough train 同时开 native terrain_levels_vel + Wolf command
  curriculum（stage="rough"，阈值/EMA/buffer 与 flat 完全同一套参数）；flat 只有
  command curriculum；play 全清空；
- rough play 保留 generator（回放地形分布与训练一致）。
```

### 30.3 Frozen contracts（flat 不变项 + rough 增量）

```text
observation / action（flat = rough = play，四任务一致；无新增维度）：
    PPO：actor 53-D 单帧 / critic 56-D / action 16-D
    HIM：history [B,6,53]（oldest→newest）/ source encoder input 318 /
         velocity target 3（×2.0）/ latent 16 / actor input 53+3+16=72 /
         action 16 / critic 56
    rough 不给 actor/critic 新增 height_map；terrain_scan 只服务 base_height reward。
checkpoint / resume：
    checkpoint 数据格式不变（infos.env_state.command_curriculum：version / stage /
    robot / vx range / EMA / streak / buffer）；restore mode：
    same-stage resume → full；flat→rough（PPO 或 HIM）→ range（EMA/streak/buffer
    fresh）；PPO→HIM warm start → range；旧 checkpoint 无 state → none + warning；
    cross-robot → fail-loud。wolf-rough-him 不支持 PPO→HIM warm start
    （warm_start_supported=False，请求报错）。
模型/optimizer/iteration 的原生行为不变；不继承上一任务的 terrain level / env
    runtime state（terrain_levels 从 max_init_terrain_level 或 checkpoint env_state
    不含该项开始；command curriculum 只按上述 mode 恢复）。
```

### 30.4 DR 在 rough 下的实际支持范围

```text
摩擦 DR（ground/wheel）：rough 下地面 geom 选择改为 terrain 实体全部 patch geom
    （``geom_names=(".*",)``；flat 保持单个 ``terrain`` geom）；压 0 语义一致，
    contact 摩擦 = 机器人 geom 值。选择器无匹配时 fail-loud（
    _press_terrain_friction 空选择：ValueError）。验证：rough contact（读
    sim.data.contact）== 轮 geom 值（19 contacts，714 geoms 压 0）。
其余 DR 项：全部只写机器人实体（mass/COMfiction/gains/delay/bias/backlash/
    target scale/push/disturbance/wheel radius），terrain 无关，语义不变。
轮半径 DR 的 root z 补偿在 rough 下同样成立（正文见 §29.3；spawn 高度由
    reset_base + env_origins 提供，事件仅追加 delta）。
安全限制：若未来 terrain 相关 DR（如地形 restitution）无法与 generator 安全组合，
    必须在配置构建阶段 fail-loud 并在本节登记，不允许静默禁用。
rough 首轮 DR 默认全部关闭（nominal rough 基线优先）。
```

### 30.5 验证记录（tests/check_wolf_rough.py + tests/check_wolf_task.py，不提交）

```text
STATIC/LIGHT CPU PASS：
    - 四 task（train+play）配置构建；observation/action shape：actor 53 / critic 56 /
      action 16（PPO）；rough HIM [B,6,53] + estimator 3；
    - flat/rough reward 装配（仅 base_height func 差异；weight/key/顺序一致）、
      terrain generator 7 类 10 行参数、curriculum（rough train = terrain_levels +
      command(stage=rough)；flat = command；play 空）、termination（rough 有
      OOB，play 无）、sensor（flat 无 terrain_scan，rough 有且 frame=base_link）；
    - rl_cfg：run_name / experiment_name / warm_start_supported 门控；
    - provenance：cross-robot fail-loud + robot="wolf" range 恢复（函数级）；
    - rough 摩擦 DR：714 patch geoms 压 0；13 步 rollout 后 19 个 wheel-terrain
      contact 摩擦 == 对应轮 geom 值；
    - rough PPO/HIM 6 步 rollout finite；
    - flat 回归（调用既有 check_wolf_task 静态+provenance 段与 check_wolf_dr
      static/nominal/friction 段）全部 PASS。
NOT RUN：PPO/HIM 实际训练（flat→rough resume / warm start / HIM full resume 的
    端到端训练验证）、rough 长训收敛、rough DR-on 训练、CUDA 全矩阵、
    rough capacity 长程 overflow 监控、export（rough 网络与 flat 同 shape，
    预期直接复用）；由后续独立 review 决定验证方式。
```

### 29.4 验证记录（tests/check_wolf_dr.py，不提交）

```text
历史（§29 初次集成）：
CPU  PASS（静态 cfg / 全关 contract 逐位一致 + 中性 DR==nominal + 无残留 buffer /
     参数 bounds / 摩擦复合 / 无漂移×5 reset / initial joint pos 分布 / target 编制 /
     obs bias 隔离 / backlash 状态机 / delay stepdown ∈[0,16] / r=0.12 静置抬高 /
     xfrc reset 清零 / PPO+HIM DR-on 真实训练 2×2iter）
CUDA PASS（同套全项 GPU 重跑，含 DR-on 训练 smoke）
回归：check_wolf_task（CPU+CUDA）/ check_black_flat（CPU+CUDA）/
     check_wolf_robot（新站立高度）/ check_wolf_imu 全部 PASS

摩擦 / 轮半径定向修复（本轮）：
CPU  PASS（静态 cfg 顺序 / 全关 + 中性 DR 等价含 contact 级摩擦 == 1.0 /
     friction composition：contact（读 sim.data.contact）== 轮 geom =
     base×U(0.4,1.0) per-env 共享、min 0.142 < 1.0（max() 下限已消除）/
     wheel-only：轮 = 1.0×U(0.4,1.0) + 地面压 0 / 轮半径 root pose 保留：
     XY/quat/velocity 不变、仅 Z += (r−0.08) / r=0.12 静置抬高）
CUDA SKIPPED（本轮仅轻量定向检查，未重跑 GPU 矩阵；摩擦/半径写入路径无
     device 分支，预期一致，待下轮完整验证补齐）
未验证：PPO/HIM DR-on 训练冒烟（本轮跳过）、长训收敛质量、rough/sim2real、
     delay sim2real 等价性 —— 未开始/未重跑。
```

### 30.6 Wolf rough wheel_force_lift 奖励（本轮集成，COMPLETE；2026-10 改为水平力门控）

```text
定义：reward = Σ_{i∈{FL,FR,RL,RR}}(1[‖f_xy^i‖ ≥ F_min] · max(v_z^i, 0))，输出 [B]。
    F_min = RewardParams.wheel_force_lift_min_horizontal_force = 1.0 N：水平接触力
    二值门控。旧的 ‖f_xy‖·v_z 力乘速度语义已废弃——平地静态噪声与弱接触不再被
    线性放大，且同一向上速度在 10 N 与 200 N 水平力下得分相同。
    f = 轮-terrain 接触力世界系 net force（ContactSensorCfg reduce="netforce"，
    fields=("force",)，num_slots=1；mujoco_warp 对每个 contact 做
    contact_frame.T @ force 后求和，恒为世界系；符号约定 primary→secondary，
    轮压地方向为负 z，本公式只用水平模长，与符号无关）。
    v_z = 轮刚体（*_Link4）世界系线速度竖直分量，clamp min=0（仅向上计入）。
不加入：力乘速度线性项 / 指令门控 / timer 状态机 / 前方障碍检测 / 目标抬升高度 /
    top-k 轮选择 / 地形自适应 / 额外接触惩罚。
权重：RewardScales.wheel_force_lift = 0.5（专属区）。
阈值：RewardParams.wheel_force_lift_min_horizontal_force = 1.0 N，经 env_cfgs
    reward params 显式传入 wheel_force_lift(min_horizontal_force=...)。
注册范围：仅 wolf-rough / wolf-rough-him（含各自 play）；flat 两任务保持
    14 项不注册。flat 也不注册配套 sensor。
sensor：wheel_force_contact（仅 rough），primary =
    WOLF_WHEEL_COLLISION_GEOM_NAMES（FL/FR/RL/RR 轮碰撞 geom 显式顺序），
    secondary = terrain body；顺序通过 sensor.primary_names 逐 call 断言。
轮部速度：WOLF_WHEEL_BODY_NAMES（FL/FR/RL/RR 的 *_Link4，
    preserve_order=True）→ data.body_link_lin_vel_w。
验证（tests/check_wolf_force_lift.py，CPU，不提交）：
    - 静态 cfg：flat 14 项无该项 / rough train+play+HIM 15 项、weight=0.5、
      min_horizontal_force=1.0 N 传递、sensor 配置 PASS；
    - 公式（stub env）：无接触 0 / 水平力<阈值 0 / 水平力≥阈值×向上速度求和 /
      向下 clamp 0 / 10N 与 200N 同向上速度等价 / shape [B] PASS；
    - 真实 env：primary 顺序 == 轮碰撞 geom 顺序、force [B,4,3]；每步 raw 与
      公式同刻逐值等价（在 reward_manager.compute 调用内手动重算，derived
      quantities 在 reward 时刻 stale 一个物理子步，step 返回后已刷新）；
      平地静态 100 步：Σfz=-330.4 N ≈ -m·g（33.67 kg）；settled raw max=0.251
      （加权 0.126/step，远小于 tracking 满量 1.0/step）；复位下沉瞬态 raw
      peak=1.34（加权 0.669/step）；
    - rough 生地形 40 步静止 + 120 步前进：raw>0 样本 229/240，raw max=0.042，
      weighted max=0.021（×dt 后 4e-4/step），无异常尖峰；
    - CUDA SKIPPED（本轮轻量验证，无 device 分支）。
未验证：障碍立面（非竖直法向）接触下的世界系衰减语义仅有 v1.6.0 源码
    （frameT @ force）+ mjlab 文档背书，未做独立物理复核；1.0 N 门控阈值 /
    0.5 权重对训练收敛的影响未做系统实验（由后续 wolf-rough 训练观察）。
```

## 31. Wolf Flat 模板 Spawn 高度与仿真容量优化（本轮集成，COMPLETE）

诊断（tests/diag_wolf_capacity.py，不提交）发现：Wolf flat 编译模板 qpos0
（freejoint root z=0、关节 0）与 plane 相交，实测模板 ncon=124 / nefc=496（4 行
pyramidal × 124 contacts），MJWarp put_data 的硬下限把 flat 容量钉在
128/512（~12x 运行时需求余量：活动样本峰值 ncon=16 / nefc=40）。

本轮修复：不改 MJCF / 不写 MjModel.qpos0，在 task 装配层用 MjLab v1.6.0 原生
`SceneCfg.spec_fn`（Scene.__init__ 内 attach 之后、compile 之前调用）把 Wolf
root body（`robot/base_link`）的 spec body pos z 抬到 0.45，由 MuJoCo 正常编译
流程产生 qpos0。

### 31.1 实现位置

```text
env_cfgs.py：
    _build_wolf_env_cfg 对全部 flat / flat-him（train + play）路径调用：
    _configure_template_spawn_height（仅 not rough 分支）：把模块顶层回调
    ``_wolf_flat_template_spec_fn`` 挂到 ``cfg.scene.spec_fn``（不嵌套定义：
    train CLI 用 asdict + yaml.dump 保存 env.yaml，``!!python/name:`` 标签只能
    表示有稳定可导入名称的对象，局部函数 qualname 含 ``<locals>`` 不可靠）：
        spec_fn 校验根 body 存在（robot/base_link）、全 spec 恰好 1 个 freejoint、
        body 不是 world body，写入 body.pos = [0,0,template_root_z]；
        cfg.scene.spec_fn 已被占用时 fail-loud（baseline 从不设置该字段）。
    常量 WOLF_TEMPLATE_ROOT_BODY = "robot/base_link"；freejoint 类型判定用
    mjtJoint（freejoint 名为 MJCF 的 robot/floating_base_joint，不属于
    root body 名前缀，不能按名字前缀过滤）。
wolf_config.py SimulationParams：
    template_root_z = 0.45（注释注明只影响模板，不改训练 reset 高度）；
    flat nconmax 128→64 / njmax 512→256（由运行时需求决定的候选值；
    rough 保持 256/1024 不变）。
```

### 31.2 实测结果与 frozen 值

```text
模板（编译 qpos0）：
    修改前：root z=0 / 关节 0 → ncon=124 / nefc=496
    修改后：root z=0.45 / 关节 0 → ncon=24 / nefc=96
            （calf Link3 cylinder ×4 + wheel ×4 处于 dist≈0 边缘；
              0.45 相对站立 FK 清 6.8mm，knee/calf 常规位形不触地）
    qpos0[0:7] = [0,0,0.45,1,0,0,0]；全部 16 个 hinge ref 仍为 0（qpos0[7:] == 0）。
    keyframe init_state（z=0.4432 + 0.82 关节）不变（key 与 qpos0 是独立机制）。
容量：flat 64/256（<模板 24/96 下限 + 运行时需求；overflow 必须 NO）；
    rough 后续由 §32 处理（同机制 spec_fn + 候选 B 128/256）。
    （历史初版改为：当时 rough 模板 spawn 下限 178/712，弯 256/1024）
不变项（实测）：
    EntityCfg INIT_STATE / default_root_state z 仍 0.4432（reset 后 root z 实测
    全部 0.4432，无 0.45 偏移）；default joint pos（EntityCfg init_state 显式
    joint_pos，不读 keyframe）不变；action / observation / reward / DR / HIM 契约
    全部不变。
```

### 31.3 行为等价验证（tests/check_wolf_template_capacity.py，不提交）

```text
对照方式：同 seed 双 env（旧装配：无 spec_fn + 128/512 模拟优化前；
新装配：spec_fn + 64/256），corruption 关闭（manager 初始化前设置，
构建后切换无效），交替 reset 前显式 reseed 保证 reset 采样序列一致。
CPU PASS：reset（root/joint/qvel/obs actor+critic）逐位级一致；20 步 rollout
    每步 root/joint/qvel/obs/reward/termination/ncon/nefc 一致（tol 1e-5）；
    runtime nefc_max=20 / ncon_max=13，overflow 全 0；
    HIM [B,6,53]+[B,3]+action 16 finite；rough 装配不变断言。
CUDA:0 PASS（256 envs / 6 步 smoke）：连续量容差按步放宽
    （tol_i = 1e-3 × 4^i；轮地切向接触为混沌系统：边缘触点 dist≈0 在 kernel
    微小数值差下离散翻转并指数放大，实测 step3 joint_vel 差 ~0.002 相对
    0.05%，属混沌分歧而非契约差异）；|Δnefc| ≤ 8/世界（efc_address 是未初始化
    workspace 字段，per-world ncon 估计不可靠，只作 informational）；
    overflow 全 0；nefc_max=36 / ncon_max=15。
回归：check_wolf_task（更新两处过期断言后）/ check_wolf_dr / check_wolf_rough /
    check_wolf_robot / check_wolf_imu 全部 PASS（CPU）。
未验证：长训容量 overflow 监控（64/256 为候选，若真实训练触发 overflow 需上调）、
    rough 模板 spawn 状态优化、CUDA 全矩阵重跑。
注意：put_data 以 qpos0 种子化 warp 派生量（xquat/xmat/ximat）——优化前的
    首步派生值与优化后不同，但 reset 后行为已验证一致；该影响仅限首步。
```

## 32. Wolf Rough 模板 Spawn 高度与 MJWarp 容量优化（本轮集成，COMPLETE）

诊断（tests/diag_wolf_rough_template.py，不提交）与 §31 同因：rough 无 spec_fn
时模板 qpos0 root z=0，深插 generator 地形 patch 之下，模板实测 ncon=178 /
nefc=712（contact rows 712 / limit rows 0），pin 住 rough 容量 256/1024。

### 32.1 模板 spawn 高度（复用 §31 机制）

```text
env_cfgs.py：
    _write_template_root_height(spec, root_z)：flat/rough 共用的校验 + 写入
    （根 body 存在 / 全 spec 恰 1 个 freejoint / 写 body.pos [0,0,z]）；
    _wolf_rough_template_spec_fn（模块顶层，YAML !!python/name: 验证 PASS）写入
    WOLF_CONFIG.simulation.rough_template_root_z；
    _configure_template_spawn_height(cfg, rough) 按 stage 选回调；
    _build_wolf_env_cfg 全部路径（flat/rough × train/play × PPO/HIM）都挂接。
    spec_fn 被占用时依旧 fail-loud。
wolf_config.py SimulationParams：
    rough_template_root_z = 10.0（初始候选，实测高于 terrain 全部 geom z 上界
    2.718 m，视为安全；模板接触 clean 无法顶开后不再抬高，如需变动需另行任务）。
结果（实测）：模板 qpos0 root z = 10.0，ncon 178→0 / nefc 712→0。
不变项：INIT_STATE / default_root_state z=0.4432、joint ref、actuator、地形几何、
    spawn origin、reset 语义、MJCF、训练容量 schema 同 §31。
```

### 32.2 容量：候选 B（最终值）

```text
rough_nconmax 256→128 / rough_njmax 1024→256（同 Black rough）。
依据（模板清零后实测 runtime per-world 峰值，中等强度动作短 rollout）：
    - CPU 4 env、L0–5：activity nefc_max 52 / ncon 17；reset 期 nefc≤20
    - CPU 8 env、max_init_level=9（难度 0.9 地形）：activity nefc_max 64 / ncon 20
    - CUDA 256 env、难度 9：activity nefc_max 72 / ncon 35
      （72/256 = 28%、35/128 = 27%，两候选均余量充足，overflow 全 NO）
    两候选 A（128/512）/ B（128/256）均验证 clean，采 B（与 Black 同值）。
训练期更激进动作 / DR 方差未纳入本诊断；4096 实训验收由用户完成。
```

### 32.3 行为一致性对照（tests/check_wolf_rough_equiv.py + 决断实验，不提交）

```text
固定 seed=123 / 4 env / 24 步 zero-action：
    - 修改前后：default_root_state、reset 后 root pos/quat/vel、joint pos/vel、
      actor obs 首 8 维 全部一致（round diff = 0）；
    - 首步之后出现微小时序差异，24 步后最大发散至 root pos ~5e-4 m / 轮接触力 66 N
      （混沌放大，不代表 contract 差异）。
    决断实验：当前代码 + 模板 z 强改回 0（frozen dataclass
    object.__ setattr__）→ 与旧代码逐位一致（0 DIFF）——差异 100% 来源于
    旧模板深插地形在 reset 时刻残留的求解器浮点微扰（qacc_warmstart 已排除，
    具体内部状态未归类），量级：reset 传感器力 ≤5e-4 N、首步 reward 差 ≤1e-5；
    新模板 z=10 下连续两 run 逐位一致可复现。
    语义结论：reset/write-observable 契约不变，残留性差异属 MjLab 求解器实现
    细节，不影响训练语义；新模板为更干净基线。
其它回归：check_wolf_rough.py FULL PASS（含 spec_fn 四任务绑定断言 + 容量值）、
    check_wolf_force_lift.py PASS（settled 阈值同步为当前 PD 调参后的站立水平）、
    check_wolf_template_capacity.py 揭示既有失败（旧版同样失败，与本轮无关）。```

### 32.4 HField overflow 复查（情况 A + 余留）

```text
- 全部本轮检查（首次构建 / reset / 40 步活动 rollout / 256 env CUDA）中
  Data.overflow 位掩码均为 0（无 HFIELD / NEFC / 其它 flag），控制台亦无
  "height field collision overflow" 警告；
- 旧模板下限 ncon=178 / nefc=712 已消除，init 期 deep-penetration 路径不复存在；
- 余留：训练期（2000+ env、机器人摔倒 / 整机沿 hfield 墙滑动）是否可能再触发
  HFIELD overflow，需 4096 实训确认；本环境内未复现，不下结论。
已知既有失败（与本轮无关，未修）：
    - check_wolf_template_capacity.py 断言 reset root z==0.4432，实测
      0.447/0.436 交替（PD 调参后站立瞬态变化，旧代码同 FAIL）；
    - check_wolf_task.py / check_wolf_posture_rewards.py 在 max_abs_vx 与
      hip alpha 期望处 FAIL（参数调参 prior commit 导致，同前记录）。
```

## 33. Wolf Rough HField Overflow 定向诊断（本轮集成，COMPLETE；诊断结论）

现象：真实训练（128/256 容量、约 3500 env）反复打印
`height field collision overflow, number of collisions >= 50`。

触发机制（mujoco-warp 3.11.0 `ccd_hfield_kernel_builder`）：对每个
(hfield patch × robot collision geom) 对按 geom 支撑 AABB 在 hfield 帧内展开
80×80 子格（dx≈0.101 m）；每格 2 个候选三角棱柱（2026-10 修正：候选须经
高度剪枝 + CCD 检查产生真实接触后 count 才递增，count = 有效接触候选数，
上限 MJ_MAXCONPAIR=50；`count ≥ 50` 时置 `OverflowType.HFIELD`
（per-world、sticky、reset 清零）+ console 每 substep 重印。每对最终仍只写
按深度选出的 ≤4 个接触点（候选空间被裁到处理序前 50）。

结论（专用诊断脚本 tests/diag_wf_hfield_overflow.py，不提交）：

```text
触发配对：terrain HFIELD × robot/base_link 碰撞盒（0.215×0.10×0.06），
    且仅该 geom 可触发（隔离实验因果验证 + 解析排除）：
    base 盒斜置支撑 AABB 最大 6x6 cells（36 cells / 72 候选 > 50）；
    轮圆柱 ≤16 cells、Leg1-3 盒/柱 ≤9-16 cells，其余数学上不可能 ≥51。
触发状态：仅倒地/翻滚低姿态——base 盒贴/陷 hfield 表面
    （实测事件：root_z 0.05~0.48 m、roll −102°~135°、base 接触力 ~1191 N、
    base_contact_found=True 弹跳沉降过程）；正常行走（3×24 env×120 步乱动作
    + 8×40 步）0 次溢出。
分辨率对照（同 crash 序列、同 seed）：horizontal_scale 0.10 → 5 次 bit 事件；
    0.125 → 0 次（支撑覆盖 cell 数低于阈值）。仅诊断对照，未写生产。
溢出类型区分：HFIELD（本轮目标）；NEFC / EPA_HORIZON 仅人工深插极端状态出现
    （正常行走 nefc 峰值 72/256 余量充足）；NARROWPHASE / CCD 等未出现。
console 警告数量 ≠ 受影响 env 数：每 (world, pair, substep) 重复打印；
    受影响规模应以 per-world bit 统计（建议训练循环加轻量 bit 计数器）。
物理影响：溢出仅裁剪候选空间到前 50；每对仍写出深度排序最优 ≤4 接触点，
    深层候选丢失可能使该 pair 接触点选择有细微差异，不改变求解数学。
最小修复候选（未实施，优先序）：1) 拆分 base_link 碰撞盒为两段小 box
    （结构性消除；接触分布/摩擦求解体积改变，需重训+行为对照，属机器人
    生产碰撞几何修改）；2) 等 mujoco-warp 上游支持 (HFIELD, BOX/CYLINDER)
    multiccd；3) horizontal_scale 0.125（实测可消除，但改变全部 hfield
    地形接触与训练分布，不推荐）。
未验证：2000+ env / curriculum 难度 >5 / 激进策略下的按 world 受影响比例；
    warp 上游后续版本变化；base 盒拆分后的训练行为。

## 34. Wolf Rough HField 警告控制 + overflow 监测（本轮集成，COMPLETE）

触发链回顾（§33）：训练倒地/翻滚 → base_link 碰撞盒 × hfield 有效接触候选
≥50 → HFIELD overflow bit（sticky）+ console 每 substep 重印刷屏。

```text
实现（wolf_config.py / env_cfgs.py，仅 rough 消费；flat / play 无诊断开销）：
    SimulationParams.rough_warn_overflow = True（默认开；风险确认后可关）。
    WolfRoughSimulationCfg(SimulationCfg)（task-local 模块顶层，YAML
    (!!python/name:) 序列化验证 PASS）：覆写 apply_wp_opt——先
    super().apply_wp_opt() 再 wp_opt.warn_overflow = False；仅在
    rough_warn_overflow=False **且 rough train** 时装入 cfg.sim（play 不关闭
    ——它不注册 overflow metrics，保留原生警告便于调试；构造保留
    SimulationCfg 全部字段：nconmax/njmax/mujoco/broadphase/nan_guard
    逐字段一致验证 PASS）。
    调用时机在 put_model 后 / CUDA Graph 捕获前（Simulation.apply_wp_opt
    的原生时机），不在 graph 捕获后改选项。
    注意：warn_overflow 是 mujoco-warp 3.11.0 布尔**总开关**，关闭的是全部
    overflow 类型打印（非仅 HFIELD）；Data.overflow bit 不受影响。
    metrics（仅 rough train 注册；play/flat 不加）：
        overflow_hfield / overflow_other，func 经 MjLab TorchArray 原生
        ``overflow[:]``（__getitem__）读底层 Tensor（直接
        torch.as_tensor(代理对象) 有运行时异常风险，已修复）取
        Data.overflow 位（HFIELD=1<<5），reduce="max"，无 CPU 同步/无逐
        env 打印；手工置位验证：1/4 env HFIELD → 0.25，other bit → 0.5，
        bit 读取后不消失（sticky 语义保持）。

HFIELD ↔ termination 关联诊断（tests/check_wf_overflow_diag.py，不提交；
    采样点 = wrap reward_manager.compute，即 termination 计算后 / reset 前；
    bit sticky → 维护 per-env seen，只计本 episode 首次置位步）：
    256 env × 60 步 crash 场景一次有效样本：首次 HFIELD 事件 15 次，
    其中 13/15 与 illegal_contact 发生在同一控制步（86.7%）；
    14/15 最迟在下一控制步终止（93.3%）；尚有 1 个未及时终止样本
    （深插滑动步状态，穿透充分）。
    ⚠ 复跑稳定性警告：同码同 seed 重跑出现过 0 事件样本（Warp 层隐藏
    随机性，机制未归类）——上述比例仅来自单次有效样本，**标记为未验证**，
    不据此宣称覆盖全部真实训练形态。
开关决策：证据方向支持"溢出集中于终止步"，但样本不可稳定复现且有未
    及时终止的深插个例 → rough_warn_overflow 默认保持 True；是否关闭由
    用户在 3500+ env 实训核对 "overflow_hfield ≤ illegal stops" 比例后决定。
风险（保留状态）：诊断复现不稳定（同码同 seed 有 0 事件样本重跑）；真实
    3500 env 训练分布未验证。本轮仅为监测通道与警告开关的功能修复，
    **不宣称 HField 碰撞溢出问题已被解决**——触发机制（§33 base 盒）与
    消除候选（拆分碰撞盒 / warp 上游 / 分辨率）仍是开放项。

验收状态：
    STATIC PASS：cfg 字段保真（8 字段一致）、YAML 序列化、metrics 注册范围
        （rough train only / play 与 flat 不加）、check_wolf_rough 全 PASS
        （spec_fn 绑定 / shape / rollout smoke）、flat sim 类不变。
    LIGHT PASS：metrics 手工置位（0.25/0.5 + bit 可读）；ON/OFF 双跑
        termination 结果一致（本次为 0 事件样本，等价弱证据）。
    NOT RUN / 未验证：OFF 后 console 警告消失（依赖可复现倒地样本，
        本轮未能稳定复现）、真实 3500 env 训练的 metrics/log 分布、
        ON/OFF 在真实碰撞下的完整等价矩阵。
```

## 26. Update Rule

每完成一个 behavior unit：

1. 先确认 production代码和本地验证通过。
2. 更新本文件相关 contract。
3. 将已完成项从 `In Progress` 移入 frozen sections。
4. 将下一项写入 `In Progress`。
5. 删除已经失效的风险或 TODO。
6. 不记录冗长开发过程，只保留当前有效状态。

本文件始终描述：

```text
现在是什么
```

而不是：

```text
曾经做过什么
```
