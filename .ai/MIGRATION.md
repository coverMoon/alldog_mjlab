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
Black rough PPO behavior migration
```

Black flat 的行为语义已全部冻结（reward / DR / command / final PPO verification，见 §11 / §12 / §4 / §17.1）。
Black rough v1 的 planned behavior unit 也已完成：terrain generator + terrain curriculum
+ terrain scan / critic privileged height + terrain-relative base-height reward +
out_of_terrain_bounds termination（§13 / §13.4 / §11.4 / §6.4）；
rough 其余 9 项 reward 仍是 flat 语义，但已无待迁移的 planned behavior unit。

当前尚未进入：

```text
Black rough PPO sanity training（尚未启动，也未验证训练效果）
Black sim2real
HIM observation/history
HIM algorithm integration
BlackW migration
```

当前 task：

```text
black-rough（flat contract + rough terrain + terrain scan + terrain-relative base height
             + OOB safety truncation）
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
```

Black 当前主要 task config：

```text
src/alldog_mjlab/tasks/velocity/black/params.py
src/alldog_mjlab/tasks/velocity/black/env_cfgs.py
src/alldog_mjlab/tasks/velocity/black/rewards.py
src/alldog_mjlab/tasks/velocity/black/terminations.py
src/alldog_mjlab/tasks/velocity/black/terrain.py
src/alldog_mjlab/tasks/velocity/black/rl_cfg.py
```

### Configuration layout

```text
params.py
    人工调参入口：若干 frozen typed parameter group，只放训练者预期会查看 /
    调整的 numeric / range 参数。访问形式 category first：
        params.command.*
        params.observation_noise.*
        params.reset.*
        params.termination.*
        params.reward.*
        params.domain_randomization.*
    最多两层，不引入总容器 / Hydra / OmegaConf / 第二套 Config framework。

env_cfgs.py
    MjLab task assembly + policy / task interface contract
    （term 顺序、selector、observation scale、sensor 身份、与 native 的差异）

rewards.py
    Black 专用 reward math

terminations.py
    Black 专用 stateful termination math
```

`params.py` **不是**第二套 runtime config：MjLab `ManagerBasedRlEnvCfg` 仍是唯一
runtime config，环境完全由 `env_cfgs.py` 组装，`params.py` 只是它的数值来源。
interface contract 不得放进 `params.py`，否则会被误读成普通训练超参数。

本地 migration verification：

```text
tests/check_black_flat.py
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

Black flat v1 的 command 是**固定范围 + MjLab native sampler**，训练全程不变。

```text
generator:
    MjLab v1.6 UniformVelocityCommand
    （body-frame 速度指令，按 resampling 重采样）

range:
    vx   [-1, 1] m/s
    vy   [-1, 1] m/s
    wz   [-π, π] rad/s

resampling:
    10 s（固定，不随机）

heading command:
    disabled（rel_heading_envs = 0，ranges.heading = None）

native sampler 比例:
    standing     10%   （指令强制为 [0, 0, 0]）
    forward-only 20%   （vx ≥ 0.3 且 vy = wz = 0；standing 优先）
    world-frame   0%
    reset 初速度  0%

curriculum:
    disabled（terrain curriculum 与 command curriculum 均已移除）
```

因此 `params.command.*` 是**最终训练 contract**，不是「curriculum 前的初始范围」；
`cfg.curriculum == {}`。

### 4.1 What was intentionally not migrated

```text
official HIMLoco:
    performance-based lin_vel_x curriculum（tracking > 0.8 扩 ±0.2 / max ±2）
    + 与之耦合的 high/low speed env command sampler

super-dog（68f1c1c 及后期）:
    buffer / EMA / pass streak / required_passes / max_curriculum
    low/high command bucket
    terrain_probe / stand_probe / stop_probe
```

这些都是 historical optional training strategy，可在 super-dog 查看；本项目 v1 不复刻。

### 4.2 Framework difference

```text
This is not an exact HIMLoco command migration.
```

official HIMLoco 的 curriculum 与其 command sampler 耦合，不能只取更新公式；
super-dog 的 curriculum 又依赖 buffer/EMA/streak 状态机与三类 probe env。

Black flat v1 有意简化为：

```text
固定范围 + MjLab native standing / forward-only sampling
```

保留 10% standing 覆盖以训练零指令行为，保留 20% forward-only 覆盖以增加前进样本；
standing / forward 的判定与采样完全复用 framework。

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

当前 training termination：

```text
black-flat   train / play : time_out + illegal_contact + stuck
black-rough  train        : time_out + illegal_contact + stuck + out_of_terrain_bounds
black-rough  play         : time_out + illegal_contact + stuck
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

时 termination：

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

当前 Black rough 数值（由 runtime terrain config 推导，非硬编码）：

```text
effective grid   10 x 5（curriculum 模式一个 terrain 一列）
patch            8 x 8 m
border_width     20 m
margin           0.3 m（native default，未显式传入）
half_x / half_y  60.0 / 40.0 m
limit_x / limit_y  59.7 / 39.7 m
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

train / play 的区别仅剩：DR、actor observation corruption、episode 长度；command
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
已由 reset contract（`params.reset.joint_position`）覆盖。

motor strength / action delay 分别在 sim2real contract 阶段处理。

------

## 13. Frozen Black Rough Terrain Contract

Black rough v1 的 terrain 由 MjLab v1.6.0 native terrain generator 生成
（`TerrainEntity` + `TerrainGeneratorCfg` curriculum 模式），只新增一个 task-local
sub-terrain primitive（rough slope），不引入第二套 terrain framework。

数值在 `params.terrain.*`，terrain 数学在 `tasks/velocity/black/terrain.py`，装配在
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
cfg.curriculum.keys() == {"terrain_levels"}
func = mjlab.tasks.velocity.mdp.terrain_levels_vel（native）
params.command_name = "twist"
command_vel 已移除（command contract 与 flat 相同）
```

推进 / 回退公式由 native 实现，与 legacy `_update_terrain_curriculum()` 一致：
walked distance > size[0] / 2 升一级；walked distance < ||command_xy|| x
max_episode_length_s x 0.5 降一级；达到 num_rows 时随机新 level；首次 reset
（common_step_counter == 0）不改 level，因此 `max_init_terrain_level` 生效
（`randint(0, max_init_terrain_level + 1)`，inclusive，与 legacy 相同）。

### 13.4 Terrain scan 与 critic privileged height

rough 的 terrain scan 复用 MjLab v1.6 native `terrain_scan` sensor（`RayCastSensorCfg`），
只把 frame 绑到 `robot/trunk`（与 native rough task 相同）：

```text
name                 terrain_scan（恰好一个；flat 会移除它）
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
black-flat   actor 45 / critic 72   （无 terrain_scan）
black-rough  actor 45 / critic 259 = flat critic 72 + height_scan 187
```

critic term 顺序被显式冻结（不依赖 native `critic_terms = {**actor_terms, ...}` 的 dict 顺序）：

```text
base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions,
command, foot_height, foot_air_time, foot_contact, foot_contact_forces,
height_scan            ← 仅 rough，追加在最后
```

height_scan 数值语义为 native `mjlab.envs.mdp.height_scan()`：

```text
raw   = sensor frame z - terrain hit z   （offset = 0；ray miss 时取 max_distance）
scale = 1 / max_distance = 0.2
noise = 无
clip  = 无
```

Intentional difference（**不是 legacy 238-D / HIM privileged observation migration**）：

```text
legacy（super-dog black_env.py）
    heights = clip(root_z - 0.5 - measured_heights, -1, 1) * 5.0
    采样 x ∈ [-0.8, 0.8] step 0.1、y ∈ [-0.5, 0.5] step 0.1（187 点，与 MjLab 同网格）
    height noise raw scale = 0.1
    privileged = 45 + base_lin_vel 3 + external disturbance 3 + heights 187 = 238

Black rough PPO v1
    MjLab native height_scan 语义（offset 0、scale 0.2、无 noise / clip）
    不含 external disturbance 分量；45 + 3 + 187 的 238-D layout 属于后续 HIM
    observation contract，不在本 baseline 内
```

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
```

command 已冻结为固定范围 + native sampler，且不再有任何 curriculum（§4）。
train / play 的 command contract 完全相同。

本轮追加 rough training 的 native OOB safety truncation（time_out=True）；
**Black rough PPO 尚未训练**（无 sanity / full run，也没有任何训练效果结论）。

下一阶段：

```text
Black rough PPO sanity training
```

（sanity 训练通过后才考虑长训练；sim2real / deployment contract 阶段需一并处理
§18.1 的 ONNX metadata 导出问题。）

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

Black flat v1 有意不迁任何 command curriculum（见 §4）：范围固定，standalone 由
§4 的 native sampler 提供 standing / forward-only 覆盖。

```text
状态: not migrated by design（不是 deferred 的 TODO）
```

官方 HIMLoco 的 performance-based curriculum 与 super-dog 的 buffer/EMA/probe 机制
如需启用，应作为新的 behavior unit 提出，并先说明要解决的训练问题。

------

### Critic Observation

flat critic 沿用 MjLab privileged observation（72 维）。
rough critic 在 flat 72 维之后追加 187 维 terrain height scan（共 259 维，见 §13.4）。
旧 Black/HIM privileged critic layout（238-D）尚未迁移。

不要在 PPO actor 任务中修改 critic layout。

------

## 16. Explicitly Not Started

以下均未开始，不得提前宣称支持：

```text
black-rough PPO 训练
    （任务已注册，planned behavior unit 已完成，但尚未做过 sanity training，见 §13）

Black sim2real/deployment contract

HIM single-step/history observation contract

HIM estimator

HIM actor-critic/storage/update integration

black-flat-him

black-rough-him

blackw-flat

blackw-rough
```

------

## 17. Local Verification Baseline

本地：

```text
tests/check_black_flat.py        flat 的 migration verification tool
tests/check_black_rough.py       rough 的 terrain / terrain scan verification tool
tests/render_black_rough.py      rough terrain 的可视化渲染（人工检查用）
```

作为 migration verification tool。

`tests/check_black_rough.py` 覆盖（见 §13 / §13.4）：flat 仍为 plane / 空 curriculum /
无 terrain_scan；rough generator 的 size / num_rows / difficulty_range / border /
max_init / 5 类 sub-terrain 与 proportion；curriculum 只含 terrain_levels；逐行难度
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
少量 env 的 zero / random rollout smoke。

当前应持续覆盖：

```text
Black asset compile

nq / nv / nu

default pose

per-joint PD

effort limit

action dimension/order/mapping

command contract（固定范围 / native sampler 比例 / 无 curriculum 且 step counter 推进后仍不变）

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

------

## 18. Known Risks

当前需要持续注意：

1. MuJoCo / mjwarp contact sensor在深度 penetration 情况下观察过 `found` 存在但 force为0的现象。正常落地/趴地时 force可正常达到明显大于1 N。目前 illegal-contact threshold继续保持1 N，后续根据训练日志判断是否需要处理。由于 trunk 与地面的接触力只在某个高度区间可靠，本地验证的强制触地 probe 会从浅到深扫几个 root 高度取首个触发，而不是固定单一高度。
2. 当前 actor contract已经固定，但 critic仍属于 MjLab baseline，后续 HIM阶段不能将其误认为旧 Black privileged observation。
3. 当前 `algorithms/him/` 中可能仍存在 Black-specific hardcoded dimensions。HIM阶段必须清理，不在当前 Black PPO阶段提前修改。
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

------

## 19. Next Migration Order

长期方向是「flat 先做完，再 rough，最后 sim2real / HIM」，因此：

```text
1. stuck termination                     （完成）
2. reward migration                      （完成：Black flat v1 baseline = 10 项，见 §11）
3. domain randomization                  （完成：6 项，见 §12）
4. command baseline                      （完成：固定范围 + native sampler，无 curriculum，见 §4）
5. Black flat final PPO verification     （完成：500-iteration run + 独立进程 reload + 8-command play/eval，见 §17.1）
6. Black rough PPO
   （已完成前置：rough terrain generator + terrain curriculum，见 §13；
     terrain scan + critic privileged height，见 §13.4；
     terrain-relative base-height reward，见 §11.4；
     out_of_terrain_bounds safety truncation，见 §6.4；
     下一前置：无 —— 可开始 sanity training，但尚未开始）
7. Black sim2real/deployment contract     （须一并处理 §18.1 的 ONNX metadata 导出）
8. HIM observation / estimator / algorithm integration
```

已插入完成的非 behavior 任务：

```text
configuration readability / tuning refactor v1   （完成，behavior-neutral）
```

进入每一步前重新检查实际代码和本文件。

不要同时推进多个 behavior unit。

------

## 20. Update Rule

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