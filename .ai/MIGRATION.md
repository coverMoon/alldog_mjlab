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
Black deployment contract / sim2sim compatibility
```

Black flat PPO baseline、Black rough PPO baseline（约 500 iteration，见 §17.2）与 rough 的
MJWarp runtime workaround（`nconmax = 128`，见 §10）均已完成；当前处于部署契约 / sim2sim 阶段
（§19），其中 deployment contract 已冻结（§19.1）、actor-only TorchScript 导出与数值验证已
完成（§19.3 / §17.3）、legacy `rl_sar` 的 45-D 单帧部署 config 已完成（§19.6 / §17.4）、
sim2sim observation / action trace 已数值对齐（§19.7 / §17.5）、rl_sar MuJoCo locomotion
rollout 已完成（§19.8 / §17.6）、`quadruped_control` 的 45-D config、
observation / action / torque trace 与 locomotion rollout 均已完成
（§19.9 / §19.10 / §19.11）。MJWarp GPU convex CCD 的 upstream 根因尚未修复。

当前尚未进入：

```text
Black real robot backend / sim2real（§19 的 sim2sim contract 验证通过前不开始）
HIM observation/history
HIM algorithm integration
BlackW migration
```

当前 task：

```text
Black PPO 训练侧（MjLab）与部署侧（rl_sar / quadruped_control）observation / action
轨迹对比（§19.5 步骤 8）
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
- 不要把该值传播到 flat，也不要顺手改动 `njmax` 等其它 sim capacity。

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
Black rough PPO sanity / baseline training: COMPLETE（§17.2）
Black rough MJWarp contact capacity workaround: COMPLETE（§10 / §17.2）
Black actor-only TorchScript export:  COMPLETE（§19.3 / §17.3）
Black legacy rl_sar 45-D deployment config: COMPLETE（§19.6 / §17.4）
Black sim2sim observation / action trace: COMPLETE（§19.7 / §17.5）
Black rl_sar MuJoCo locomotion rollout: COMPLETE（§19.8 / §17.6）
Black quadruped_control 45-D deployment config: COMPLETE（§19.9 / §17.7）
Black quadruped_control observation/action/torque trace: COMPLETE（§19.10 / §17.8）
Black quadruped_control MuJoCo locomotion rollout: COMPLETE（§19.11 / §17.9）
Black deployment / sim2sim contract:  IN PROGRESS（§19）
```

command 已冻结为固定范围 + native sampler，且不再有任何 curriculum（§4）。
train / play 的 command contract 完全相同。

当前单元是部署契约 / sim2sim 兼容（§19）：deployment contract 已冻结（§19.1）、
actor-only TorchScript 导出与数值对齐已完成（§19.3 / §17.3）、legacy `rl_sar` 的
45-D 单帧部署 config 已建立并验证可加载（§19.6 / §17.4）、sim2sim 数据链已数值对齐
（§19.7 / §17.5）、rl_sar MuJoCo locomotion rollout 已验证（§19.8 / §17.6）、
`quadruped_control` 的 45-D config 与 observation / action / target / torque trace 均已验证
（§19.9 / §19.10 / §17.7 / §17.8），`quadruped_control` 的 locomotion rollout 也已完成
（§19.11 / §17.9）。下一步是训练侧与部署侧的跨 runtime observation / action 轨迹对比
（§19.5 步骤 8）。
rough 的 sanity / baseline 训练属于 pipeline / baseline verification，**不是** long-run
convergence 结论。

下一阶段：

```text
§19.5 的部署迁移顺序（剩余步骤 7 → 8：quadruped_control locomotion rollout → 跨 runtime
轨迹对比 → 最后才考虑 real robot backend / sim2real）
```

（sim2sim contract 验证通过前不要开始 real robot backend / sim2real；本阶段不要开始 HIM
integration；ONNX metadata 归属见 §18.1 / §19.4。）

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
Black real robot backend / sim2real
    （§19 的 sim2sim contract 验证通过前不开始；ONNX metadata 归属见 §18.1 / §19.4）

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
export command    uv run python -m alldog_mjlab.utils.export_policy \
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

未做：MuJoCo backend rollout、observation/action trace 数值对比、运动表现评价（下一单元）。

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
torque saturation（rl_sar 计算值，policy 限幅 33.5）
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
RlController::create。策略实际 step 需要交互式终端输入，留给下一单元。

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

结论：**PASS**（policy I/O 与 torque 链全部对齐；torque 上限差异不影响本轮 policy 轨迹，
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
关节位置限                        基本未触发：|q_target - (default+0.25a)| > 0.5 rad 的样本 0 个；
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
- 部署侧现状（见 §19.4）：两个目标运行时都用 TorchScript + 显式配置，因此该 metadata 问题
  **不是**当前部署的主要阻塞项；部署阶段同样不要为了它重命名 action term。

------

## 19. Frozen Black Deployment / Sim2Sim Contract

阶段状态：

```text
Black flat PPO baseline:               COMPLETE
Black rough PPO baseline:              COMPLETE
Black rough MJWarp runtime workaround: COMPLETE
```

下一阶段：

```text
Black deployment contract / sim2sim compatibility
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

### 19.1 当前 Black PPO policy（= deployment）contract

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

action：12 维 position residual。

```text
target_joint_pos = default_joint_pos + 0.25 * policy_action
```

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
导出              RSL-RL 5.4.2 原生 runner.export_policy_to_jit()（actor.as_jit()）
环境              task registry 的 play cfg（num_envs = 1），维度取自 env 而非 checkpoint
```

导出模块的 deployment contract（冻结）：

```text
input   float32 [1, 45]   actor 单帧 observation
output  float32 [1, 12]   policy action
```

验收已满足：同一 observation 下与 checkpoint actor 的 deterministic forward 在
atol 1e-6 / rtol 1e-5 内一致（实测三组 probe 的 max abs diff 均为 0.0，见 §17.3）。
导出侧不做任何额外 normalization / scaling（actor normalization 为 disabled，§5.3）。

尚未验证：sim2sim 侧的 observation 预处理与 action 后处理是否与训练侧一致（属 §19.5 步骤 4 之后）。

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
8. Compare observation/action traces between training-side and deployment-side runtimes.
                                                                      （下一单元）
9. Only after sim2sim contract is verified, proceed to real robot backend / sim2real.
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
torque limit                部署 33.5 N·m vs 训练 effort_limit 20 N·m
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
torque        训练 effort_limit 20 / rl_sar policy 限幅 33.5 / 部署 MuJoCo actuatorfrcrange ±20
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

torque limit（本单元按指示只报告，未改架构）：

```text
RlConfig 没有 torque_limits 字段；策略侧扭矩上限只能由 RobotModel.joints[].limits.max_effort 表达，
当前为 hip/thigh 23.7 N·m、calf 59.25 N·m（近期提交 6bb7f1d 同步自 blackW 电机规格），
再由 backends/mujoco 的 joint_control 与 MuJoCo actuator ctrlrange 取交集。
⇒ 部署侧 policy torque limit 33.5 N·m 在当前 schema 下**无法按 policy 表达**，
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
action            q_target = clamp(default + 0.25 * action, 关节位置限)，
                  无额外 action clip（clip=100 不触发）；max_position_jump=1.0 未触发
torque            tau_raw = kp*(q_target-q) + kd*(dq_target-dq) + ff
                  → clamp 到 [max(ctrl_min,-max_effort), min(ctrl_max,max_effort)]
                  → MuJoCo 无额外 clamp（actuator_force == qfrc_actuator == data.ctrl）
                  实测 RL 阶段 |tau_raw| ≤ 16.6 N·m，从未触到 23.7/59.25/33.5/20
关节范围          training / RobotModel / MuJoCo 三者一致（hip ±0.5、thigh [-1.2,1.6] 或
                  [-1.6,1.2]、calf [-2.5,-0.85] 或 [0.85,2.5]）
```

当前 torque 语义与另外两个 runtime 的差异（仅记录，未决定）：

```text
training            20 N·m 全场统一（mjlab effort_limit）
rl_sar              33.5 N·m policy 限幅，再被 MuJoCo 模型交到 ±20
quadruped_control   policy 级无上限；实际 23.7（hip/thigh）/ 59.25（calf），backend 与 MuJoCo 一致
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
torque 语义（本 runtime）   policy 级无上限；实际上限 23.7（hip/thigh）/ 59.25（calf）；
                            本轮 RL 阶段 max 20.93 N·m，0.00% 触发截断
关节位置限                  与训练 MJCF 完全一致，因此 deployment clamp 实质不生效
                            （0 个样本偏离 > 0.5 rad），不构成 training / deployment 差异
```

结论：**PASS**（engineering-level）。L1 属策略行为、L2 属跟踪精度，均不构成本阶段阻塞。

------

## 20. Next Migration Order

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
     sanity / baseline 训练（约 500 iteration）完成，见 §17.2；
     下一前置：无 —— 长训练 / 定量评估仍未做）
7. Black deployment contract / sim2sim → Black sim2real
   （进行中：deployment contract 已冻结（§19.1）、actor-only TorchScript 导出与数值验证已完成
     （§19.3 / §17.3）、legacy `rl_sar` 的 45-D 单帧 deployment config 已完成并验证可加载
     （§19.6 / §17.4）、sim2sim observation / action trace 已数值对齐（§19.7 / §17.5）、
     rl_sar MuJoCo locomotion rollout 已完成（§19.8 / §17.6）、`quadruped_control` 的
     45-D 单帧 deployment config、observation / action / torque trace 与 locomotion rollout
     均已验证
     （§19.9 / §19.10 / §19.11 / §17.7 / §17.8 / §17.9）；下一单元为 §19.5 步骤 8 的
     训练侧 / 部署侧跨 runtime observation / action 轨迹对比；ONNX metadata 归属见
     §18.1 / §19.4；真实机器人 backend /
     sim2real 需在 sim2sim contract 验证通过后开始）
8. HIM observation / estimator / algorithm integration
```

已插入完成的非 behavior 任务：

```text
configuration readability / tuning refactor v1   （完成，behavior-neutral）
```

进入每一步前重新检查实际代码和本文件。

不要同时推进多个 behavior unit。

------

## 21. Update Rule

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