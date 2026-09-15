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
    人工调参入口：只放训练者预期会查看 / 调整的 numeric / range 参数
    （command range、observation noise、reset range、termination 阈值、
    已冻结 reward 的 weight / sigma）

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

当前 MjLab baseline仍存在 encoder bias domain randomization。

旧 Black没有完全对应的同类机制。

状态：

```text
deferred to DR migration
```

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

### 11.1 Tracking

`black-flat` 的 tracking 使用 task-local 实现：

```text
src/alldog_mjlab/tasks/velocity/black/rewards.py
```

term name 继续沿用 MjLab native 名称，只替换 func / weight / params：

```text
track_linear_velocity
    func = track_linear_velocity_xy
    reward = exp(-Σ(vx/vy 误差²) / sigma)
    weight = 2.0
    sigma = 0.25

track_angular_velocity
    func = track_angular_velocity_z
    reward = exp(-(wz 误差²) / sigma)
    weight = 1.5
    sigma = 0.25
```

`sigma` 即 legacy `tracking_sigma`，denominator 不是 `sigma²`。

不使用 MjLab native `track_*`：native 会把 `v_z²` / `ω_xy²` 并入同一个
exponential，与 Black 的 tracking contract 不等价。

只使用 raw command（`command_manager.get_command`）与 body-frame root 速度
（`root_link_lin_vel_b` / `root_link_ang_vel_b`）。

`dt` 缩放由 `RewardManager`（`scale_rewards_by_dt=True`）统一完成，reward 函数
只返回 raw 值；legacy weight 可直接作为 MjLab weight。

training / play 共用同一 contract。

未迁移（仍为 MjLab baseline）：

```text
orientation / upright
base_height
pose / stand_still
action_rate_l2 / smoothness
foot / gait rewards
termination reward
```

### 11.2 Base motion stability

tracking 不包含 `v_z` 与 `ω_xy`，这两个职责由独立 penalty 承担：

```text
lin_vel_z
    func = vertical_linear_velocity_l2
    raw = body-frame v_z²
    weight = -2.0

body_ang_vel
    func = angular_velocity_xy_l2
    raw = body-frame ω_x² + ω_y²
    weight = -0.05
```

两个函数均返回非负 raw magnitude，负号由 weight 负责；无 command gating、
无 exponential、无 abs；不乘 `dt`（由 `RewardManager` 统一处理）。

`body_ang_vel` 沿用 MjLab native term name，但不使用 native math：
`mdp.body_angular_velocity_penalty` 读的是 **world-frame** body 角速度，
legacy 用的是 body-frame root 角速度（pitch 90° 时两者分别约为 5.0 / 104.0）。

同时四个职责严格解耦：

```text
track_linear_velocity   ← vx / vy 误差
track_angular_velocity  ← wz 误差
lin_vel_z               ← vz
body_ang_vel            ← ωx / ωy
```

#### flat vs rough intentional difference

`black-flat` 的 `lin_vel_z` 采用 legacy 的 **terrain-level-0** 分支：

```text
reward = v_z²
```

legacy 在 rough terrain 下还有一层地形系数：

```text
terrain_levels > 0 → ×0.1
```

该系数依赖 terrain level state，属于 `black-rough` 阶段，本 task 不引入。
这是有意的 task specialization，不是遗漏。

------

## 12. In Progress

当前下一行为单元：

```text
Black reward migration:
orientation reward
```

legacy `orientation`（weight -0.8）与 MjLab `upright`（weight 1.0）公式不同：
legacy 是 `|roll| + |pitch|·adaptive`，MjLab 是 `exp(-Σg_xy²/std²)`。
本单元只处理 orientation，不顺手迁 `base_height` / `stand_still`。

------

## 13. Deferred Black Flat Work

尚未迁移/冻结：

### Reward

旧 Black reward 已完成迁移：tracking 组与**部分** stability 组（`lin_vel_z`、
`body_ang_vel`，见 §11）。

尚未迁移：

```text
orientation
base_height
stand_still
→ action / joint penalties
→ foot / gait rewards
→ termination reward
```

不要一次迁全部 reward。

------

### Domain Randomization

仍需逐项比较：

```text
friction
payload / mass
COM
link mass
motor strength
Kp/Kd
initial state
inertia
disturbance
push
action delay
encoder bias
```

不要在 reward或其他任务中顺手改 DR。

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

## 14. Explicitly Not Started

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

## 15. Local Verification Baseline

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

tracking reward formula / invariance / dt scaling

base motion stability penalty（v_z² / ω_xy²）formula / invariance / 与 tracking 的解耦

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

## 16. Known Risks

当前需要持续注意：

1. MuJoCo / mjwarp contact sensor在深度 penetration 情况下观察过 `found` 存在但 force为0的现象。正常落地/趴地时 force可正常达到明显大于1 N。目前 illegal-contact threshold继续保持1 N，后续根据训练日志判断是否需要处理。
2. MjLab command curriculum仍会改变训练后期 command ranges，因此当前“初始 command contract”不等于完整 curriculum contract。该行为已在本地验证中显式记录：`params.py` 的 `BLACK_COMMAND_*_RANGE` 是初始值，首次 reset 后 `command_vel` curriculum 会把 `ang_vel_z` 改写成 `[-0.5, 0.5]`。
3. 当前 actor contract已经固定，但 critic仍属于 MjLab baseline，后续 HIM阶段不能将其误认为旧 Black privileged observation。
4. 当前 `algorithms/him/` 中可能仍存在 Black-specific hardcoded dimensions。HIM阶段必须清理，不在当前 Black PPO阶段提前修改。
5. Legacy Black reward 存在两个版本：当前 `black_config.py` / `black_env.py`（HEAD，含 2026-07-15 的 `1f344d9` 覆盖式同步）与 2026-06-29~07-03 的日志 lineage。**reward migration 以 2026-07-03 lineage（`68f1c1c`）为准**；`1f344d9` 不作为 reward migration authority——它同时改动了 reward / command / DR / terrain / termination / PPO，且没有对应的 reward 决策记录，故不作为迁移依据。这是迁移 source decision，不是对该提交作者意图的事实断言。已按 `68f1c1c` 迁移：`tracking_lin_vel` 2.0、`tracking_ang_vel` 1.5（两版本来就一致）、`lin_vel_z` -2.0、`ang_vel_xy` -0.05。其余项（`orientation`、`base_height`、`stand_still`、`action_rate`、`smoothness`、`dof_acc`、`joint_power`、`foot_impact_vel`、`collision`、`feet_stumble`、`feet_air_time`、`foot_clearance`、`dof_pos_limits`、`torque_limits`、`trot`、`hip_pos`、`all_joint_pos`、`foot_slip`、`progress`、`raibert`、`termination`）均需在各自 behavior unit 开始前按该 lineage 逐项落地。

------

## 17. Next Migration Order

当前严格顺序：

```text
1. stuck termination            （完成）
2. reward migration             （进行中：tracking + lin_vel_z/body_ang_vel 完成，orientation 待开始）
3. domain randomization
4. command curriculum
5. Black flat final PPO verification
6. Black sim2real/deployment contract
7. Black rough PPO
```

已插入完成的非 behavior 任务：

```text
configuration readability / tuning refactor v1   （完成，behavior-neutral）
```

进入每一步前重新检查实际代码和本文件。

不要同时推进多个 behavior unit。

------

## 18. Update Rule

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