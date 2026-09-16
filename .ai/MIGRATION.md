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
Black flat PPO behavior migration
```

当前目标是先完成并冻结 Black flat PPO 的行为语义，然后再进入 rough terrain、sim2real 和 HIM。

当前尚未进入：

```text
Black rough
Black sim2real
HIM observation/history
HIM algorithm integration
BlackW migration
```

当前 task：

```text
black-flat
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

当前 Black velocity command：

```text
resampling time = 10.0 s

heading command = False

vx ∈ [-1.0, 1.0] m/s
vy ∈ [-1.0, 1.0] m/s
wz ∈ [-π, π] rad/s
```

当前命令来源：

```text
twist
```

注意：

当前 MjLab command curriculum 仍然存在，并可能在训练过程中修改 velocity ranges。

该行为尚未完成迁移，暂时 deferred。

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
time_out
illegal_contact
stuck
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
out_of_terrain_bounds
```

flat task 不使用 orientation-based fall termination。

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
    raw = (root_link_pos_w.z - 0.43)²               flat：world z 即离地高度
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
    base_height_l2_flat / joint_power_l1 / dof_acc_l2（stateful class term）
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

`play=True` 为 nominal physics：上面 6 个 DR event 整组移除，只保留 reset events
（`reset_base` / `reset_hip_joints` / `reset_thigh_joints` / `reset_calf_joints`），
并继续关闭 actor corruption、curriculum 与 push。

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

## 13. In Progress

Black flat domain randomization：

```text
COMPLETE
```

当前 DR 已冻结为 §12 的 6 项，train 生效、play 为 nominal physics。

下一阶段：

```text
Black command curriculum
```

进入 curriculum 前先比较 legacy 的 curriculum 规则（阈值 / EMA / required passes /
buffer）与 MjLab `command_vel` staged velocity curriculum，以及
command ranges 被 curriculum 改写后的语义。

------

## 14. Deferred Black Flat Work

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

旧 Black command curriculum与 MjLab当前 staged velocity curriculum尚未对齐。

状态：

```text
deferred
```

------

### Critic Observation

当前 critic仍沿用 MjLab privileged observation。

旧 Black/HIM privileged critic layout尚未迁移。

不要在 PPO actor任务中修改 critic layout。

------

## 15. Explicitly Not Started

以下均未开始，不得提前宣称支持：

```text
black-rough

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

## 16. Local Verification Baseline

本地：

```text
tests/check_black_flat.py
```

作为 migration verification tool。

当前应持续覆盖：

```text
Black asset compile

nq / nv / nu

default pose

per-joint PD

effort limit

action dimension/order/mapping

command contract

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

## 17. Known Risks

当前需要持续注意：

1. MuJoCo / mjwarp contact sensor在深度 penetration 情况下观察过 `found` 存在但 force为0的现象。正常落地/趴地时 force可正常达到明显大于1 N。目前 illegal-contact threshold继续保持1 N，后续根据训练日志判断是否需要处理。由于 trunk 与地面的接触力只在某个高度区间可靠，本地验证的强制触地 probe 会从浅到深扫几个 root 高度取首个触发，而不是固定单一高度。
2. MjLab command curriculum仍会改变训练后期 command ranges，因此当前“初始 command contract”不等于完整 curriculum contract。该行为已在本地验证中显式记录：`params.py` 的 `BLACK_COMMAND_*_RANGE` 是初始值，首次 reset 后 `command_vel` curriculum 会把 `ang_vel_z` 改写成 `[-0.5, 0.5]`。
3. 当前 actor contract已经固定，但 critic仍属于 MjLab baseline，后续 HIM阶段不能将其误认为旧 Black privileged observation。
4. 当前 `algorithms/him/` 中可能仍存在 Black-specific hardcoded dimensions。HIM阶段必须清理，不在当前 Black PPO阶段提前修改。
5. Legacy Black reward 存在两个版本：当前 `black_config.py` / `black_env.py`（HEAD，含 2026-07-15 的 `1f344d9` 覆盖式同步）与 2026-06-29~07-03 的日志 lineage。Black flat v1 的 reward authority 不再取二者之一：核心公式改用 InternRobotics/HIMLoco 官方 Go1 baseline，机器人数值取 Black intrinsic（见 §11.0）；`68f1c1c` 只作为 optional shaping / 历史调参 / sim2real 诊断参考。`1f344d9` 不作为迁移依据（它同时改动 reward / command / DR / terrain / termination / PPO 且无对应决策记录），这是迁移 source decision，不是对该提交作者意图的事实断言。`super-dog` 的 shaping 项若日后需要启用，须先确认取哪一版。

------

## 18. Next Migration Order

长期方向是「flat 先做完，再 rough，最后 sim2real / HIM」，因此：

```text
1. stuck termination                     （完成）
2. reward migration                      （完成：Black flat v1 baseline = 10 项，见 §11）
3. domain randomization                  （完成：6 项，见 §12）
4. command curriculum
5. Black flat final PPO verification
6. Black rough PPO
7. Black sim2real/deployment contract
8. HIM observation / estimator / algorithm integration
```

已插入完成的非 behavior 任务：

```text
configuration readability / tuning refactor v1   （完成，behavior-neutral）
```

进入每一步前重新检查实际代码和本文件。

不要同时推进多个 behavior unit。

------

## 19. Update Rule

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