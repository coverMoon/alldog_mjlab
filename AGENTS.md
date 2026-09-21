# AGENTS.md

本文件用于指导 AI coding agent 和后续维护者在 `alldog_mjlab` 中进行开发。

项目目标是将旧 Isaac Gym / legged_gym / HIMLoco 四足强化学习工程逐步迁移到 MjLab，并最终支持 Black、BlackW、HIMLoco、复杂地形和 sim2real。

本文件记录长期稳定的开发规则。  
当前迁移阶段、已经冻结的具体 contract、已知风险和下一任务，以：

```text
.ai/MIGRATION.md
```

为准。

开始任何迁移任务前，必须同时检查本文件、`.ai/MIGRATION.md` 和实际代码。若文档与代码冲突，以实际代码为当前状态，并明确报告文档过期。

---

## 1. Repository

目标仓库：

```text
logical: coverMoon/alldog_mjlab
package: alldog_mjlab
production: src/alldog_mjlab/
```

开始工作前确认实际仓库：

```bash
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
```

不要根据仓库名猜测本地路径。

主要 production layout：

```text
src/alldog_mjlab/
├── robots/
│   └── black/
├── tasks/
│   └── velocity/
│       └── black/
├── algorithms/
│   └── him/
└── utils/
```

职责划分：

```text
robots
    机器人资产、MJCF、joint、actuator、default pose、
    effort limit、机器人固有参数

tasks
    observation、action、command、reward、termination、
    reset/event、terrain、curriculum、task sensor

algorithms
    HIM estimator、storage、actor-critic、PPO update、
    runner 等算法逻辑

utils
    与具体 task 行为无关的项目级工具，例如 policy export
```

不要重新引入 legged_gym 风格的大型 Env 类。

---

## 2. Source Authority

迁移过程中必须明确区分不同来源。

### 2.1 Target architecture

```text
coverMoon/alldog_mjlab
```

当前代码是最终目标架构。

已有代码和已经冻结的 contract 优先级高于旧工程的隐式行为。

---

### 2.2 Legacy Black / BlackW behavior

逻辑仓库：

```text
coverMoon/super-dog
```

本地 source root：

```text
~/PROJECT/Dog/Train/HIMLoco
```

注意：该目录虽然名为 `HIMLoco`，在本项目中代表旧 `super-dog`，不能自动视为官方 `InternRobotics/HIMLoco`。

常用路径：

```text
Black:
~/PROJECT/Dog/Train/HIMLoco/legged_gym/legged_gym/envs/black/

BlackW:
~/PROJECT/Dog/Train/HIMLoco/legged_gym/legged_gym/envs/blackW/

legged_gym base:
~/PROJECT/Dog/Train/HIMLoco/legged_gym/legged_gym/envs/base/
```

旧工程用于迁移：

```text
behavior
parameter semantics
historical tuning experience
sim2real experience
```

不要迁移其目录结构和大型环境类。

Black 任务优先读取 Black 源码和日志。  
BlackW 任务优先读取 BlackW。

只有明确进行跨机器人比较时才混合来源。

---

### 2.3 MjLab authority

框架权威：

```text
mujocolab/mjlab
```

严格使用：

```text
v1.6.0
```

涉及 MjLab API 时必须检查以下之一：

```text
MjLab v1.6.0 tag 源码
当前项目锁定版本的本地安装源码
```

禁止使用：

```text
MjLab main API
Isaac Lab API
记忆中的旧 API
```

替代实际 v1.6.0 行为。

优先使用 MjLab 原生：

```text
Manager
ManagerTerm
Entity
Actuator
Sensor
Event
Observation
Action
Termination
Curriculum
```

只有确认 v1.6.0 原生机制无法表达需求时，才增加自定义实现。

---

### 2.4 HIM algorithm authority

标准 HIM 算法来源：

```text
InternRobotics/HIMLoco
HIMLoco 原论文
```

`super-dog` 中修改后的 HIM 只能作为工程经验来源，不能自动视为标准算法。

HIM 阶段必须重新核对官方实现。

算法层不得长期保留机器人相关硬编码，例如：

```text
270
238
45
12
```

维度应来自 observation specification、environment contract 或算法配置。

---

### 2.5 Official locomotion reference

```text
mujocolab/anymal_c_velocity
```

可以用于参考：

```text
MjLab locomotion task organization
Manager usage
observation/action/reward design
```

不能作为 Black 行为权威。

---

## 3. Deployment Sources

### 3.1 Legacy deployment

逻辑仓库：

```text
N-W-wolf/rl_sar-black-W
```

本地：

```text
/home/windnotebook/PROJECT/Dog/real_robot
```

用途：

```text
历史 Black / BlackW 实机部署行为
policy I/O compatibility
joint mapping
default pose
PD
control frequency
已验证实机经验
```

不要自动继承所有历史参数。

如果旧 deployment 与当前 `alldog_mjlab` contract 冲突，必须显式比较后决定。

---

### 3.2 Main future deployment runtime

逻辑仓库：

```text
N-W-wolf/quadruped_control
```

本地：

```text
/home/windnotebook/PROJECT/Dog/quadruped_control
```

它是后续主要 deployment runtime。

当前重点：

```text
MuJoCo backend
replay
simulation-side runtime
policy integration
```

真实机器人 backend 尚需后续完成。

该仓库中的以下设计可以作为 deployment-side 参考：

```text
RobotModel
policy config
observation/action contract
MotionRuntime
RobotIO/backend separation
```

但：

```text
alldog_mjlab 不得依赖 quadruped_control 内部类或目录结构
```

二者只能通过显式 policy deployment contract 对接。

---

## 4. Contract Boundary

以下内容都属于显式 contract：

```text
joint order
action order
observation layout
history layout
observation scales
observation normalization
action semantics
action scale
default pose
policy dt
control dt
command layout
joint mapping
deployment mapping
```

不得依赖：

```text
dict 偶然顺序
regex 返回顺序
MJCF natural order
旧框架隐式行为
```

任何修改这些 contract 的任务，都必须先说明影响。

---

## 5. Black Joint Order

Black policy / deployment joint order 固定为：

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

Black 当前 MuJoCo natural order 可能为：

```text
FL → FR → RR → RL
```

二者不能混用。

所有 deployment backend 必须显式进行 mapping。

---

## 6. Current Black PPO Contract

具体冻结值始终以 `.ai/MIGRATION.md` 为准。

当前需要特别注意：

```text
Black PPO actor input = single-frame 45-D
```

不是：

```text
270-D HIM history input
```

当前 actor observation layout：

```text
[0:3]    command
[3:6]    base angular velocity
[6:9]    projected gravity
[9:21]   joint position relative to policy default
[21:33]  joint velocity
[33:45]  previous action
```

当前 PPO：

```text
history = none
actor running normalization = disabled
action dimension = 12
```

当前 action 语义：

```text
target_joint_pos
    = policy_default_joint_pos
    + action_scale * policy_action
```

Black 当前 action scale：

```text
0.25 rad
```

当前 nominal policy PD：

```text
Kp = 40
Kd = 1.2
```

policy period：

```text
0.02 s
50 Hz
```

physics / low-level simulation step：

```text
0.005 s
```

当前阶段不要把旧 HIM 的：

```text
45 × 6 = 270
```

重新引入 PPO deployment path。

---

## 7. Torque / Effort Semantics

必须区分：

```text
training actuator effort limit
deployment policy torque limit
robot hardware / RobotModel hard safety limit
```

它们属于不同层。

当前已知状态：

```text
alldog_mjlab training effort_limit:
    20 N·m

current Black deployment policy torque limit:
    33.5 N·m
```

这个 training / deployment 差异当前有意保留，除非新的 migration unit 专门处理，否则不要顺手修改。

Deployment policy torque limit 应允许：

```text
per-policy configuration
```

RobotModel 或硬件层限制表达物理安全上限。

不要把某个 policy 的 torque limit 静默变成机器人固有参数。

---

## 8. Task IDs

Task ID 使用全小写短名：

```text
black-flat
black-rough
black-flat-him
black-rough-him
blackw-flat
blackw-rough
```

不要创建另一套命名规则。

不要新增独立：

```text
train.py
play.py
```

优先使用 MjLab CLI。

---

## 9. Migration Strategy

每次只迁移一个明确 behavior / contract unit。

例如：

```text
command
action order
observation layout
termination
reset
reward group
terrain
deployment export
deployment config
```

禁止一次性重写整个 Black environment。

不要因为代码“可以更漂亮”做无关重构。

原则：

```text
small
verifiable
reversible
```

---

## 10. Current Long-Term Order

当前长期迁移方向：

```text
Black flat PPO
→ Black rough PPO
→ Black deployment contract / sim2sim
→ Black sim2real
→ HIM observation contract
→ HIM algorithm integration
→ Black HIM
→ BlackW flat
→ BlackW rough
→ BlackW HIM / sim2real
→ manipulation / perception / high-dynamic extensions
```

任何时候都先检查 `.ai/MIGRATION.md` 判断当前阶段。

不要同时推进多个阶段。

---

## 11. Before Modifying Code

每个 migration unit 开始前必须先明确：

```text
1. 本次目标
2. 要读取哪些来源
3. legacy 行为
4. 当前 alldog_mjlab 行为
5. MjLab v1.6.0 能力 / 限制
6. 哪些行为保持
7. 哪些行为有意改变
8. production files 允许修改哪些
9. verification 方法
```

如果来源发生冲突，必须先报告：

```text
当前 alldog_mjlab
legacy super-dog / deployment
MjLab v1.6.0
差异原因
行为影响
推荐决策
```

不要静默选择其中一个来源。

---

## 12. Production Change Discipline

修改范围应尽量小。

不要：

```text
顺手重构无关模块
更改已经冻结的 joint order
根据 MJCF 顺序推断 policy order
复制完整 legacy Env
引入第二套 config framework
硬编码算法维度
为了 exporter 修改 policy contract
```

`params.py` 当前定位：

```text
人工调参 numeric / range 参数入口
```

它不是新的 runtime config system。

MjLab：

```text
ManagerBasedRlEnvCfg
```

仍然是唯一 runtime environment config。

Interface contract 应放在负责 wiring 的代码中，不要伪装成普通调参参数。

---

## 13. Policy Export

当前 deployment runtime 主要使用：

```text
TorchScript
```

MjLab / RSL-RL training checkpoint：

```text
model_xxx.pt
```

是训练 checkpoint，不能直接假设可被：

```cpp
torch::jit::load()
```

加载。

TorchScript exporter 应优先使用当前锁定的 RSL-RL 原生能力，例如：

```python
runner.export_policy_to_jit(...)
```

不要：

```text
手工重建 actor 网络
手工猜 checkpoint state_dict layout
重复实现 RSL-RL checkpoint migration
```

当前 Black PPO TorchScript contract：

```text
input:
    float32 [1,45]

output:
    float32 [1,12]
```

导出后必须重新 `torch.jit.load()` 并进行 numerical equivalence verification。

---

## 14. ONNX

当前 Black 使用四个 action term：

```text
joint_pos_fl
joint_pos_fr
joint_pos_rl
joint_pos_rr
```

MjLab v1.6 velocity ONNX metadata exporter 假设存在：

```text
joint_pos
```

因此 stock metadata path 与当前 Black action contract 存在已知冲突。

不要为了修复 ONNX metadata：

```text
重命名 Black action term
修改 action order
修改 frozen contract
修改 MjLab 源码
```

当前两个主要 deployment runtime 使用：

```text
TorchScript + explicit config
```

因此 ONNX metadata 不是当前主要 deployment blocker。

---

## 15. Deployment Compatibility

训练工程负责定义：

```text
observation layout
history layout
joint order
action order
scales
default pose
action semantics
control / policy dt
```

deployment framework 负责：

```text
从 simulator / hardware state 构造 policy observation
执行 joint mapping
调用 policy
把 action 映射为低层 command
增加 deployment safety layer
```

不要为了兼容某个 deployment repo 而静默修改训练侧 contract。

需要改 contract 时必须先报告：

```text
training-side current contract
legacy deployment behavior
new deployment runtime behavior
hardware historical behavior
影响
```

---

## 16. Runtime Safety vs Policy Semantics

Deployment runtime 可以存在：

```text
action clipping
joint-limit clamp
maximum target jump
command timeout
hardware protection
```

这些需要明确标记为：

```text
deployment safety layer
```

不要把它们自动视为 policy 本身的 action semantics。

做 sim2sim strict trace comparison 时，应检查这些 safety mechanism 是否被触发。

如果被触发，必须单独报告。

---

## 17. Observation Trace Verification

部署阶段不要只比较最终 policy output。

推荐逐块比较 observation：

```text
command
angular velocity
projected gravity
joint position error
joint velocity
previous action
```

然后比较：

```text
complete observation
raw actor action
scaled action
target joint position
```

这样可以定位：

```text
joint order
quaternion convention
body/world frame
scale
default pose
last-action update timing
```

等问题。

---

## 18. Local Migration Verification

本地 verification tool：

```text
tests/check_black_flat.py
```

当前约定：

```text
tests/ 只用于 migration verification
不提交 Git
```

除非用户明确改变策略：

```text
不要 git add tests/
不要为 tests/ 修改 .gitignore
```

已有 contract test 不得因为新增测试而削弱。

新 behavior unit 必须增加对应 verification。

---

## 19. Verification Expectations

根据修改内容，尽可能覆盖：

```text
static contract checks
CPU runtime
CUDA runtime
checkpoint reload
policy inference
finite rollout
numerical equivalence
sim2sim
```

标准本地命令之一：

```bash
uv run python tests/check_black_flat.py --device cpu
```

CUDA 可用时：

```bash
uv run python tests/check_black_flat.py --device cuda
```

最终报告必须明确写：

```text
CPU PASS / FAIL
CUDA PASS / FAIL / SKIPPED
```

不能只写“测试通过”。

---

## 20. MjLab / RSL-RL Version Discipline

当前项目锁定：

```text
mjlab == 1.6.0
```

RSL-RL 具体版本以：

```text
uv.lock
```

为准。

涉及：

```text
runner
checkpoint
exporter
normalization
action manager
observation manager
sensor
event
termination
```

等 API 时，应检查实际锁定版本源码。

不要因为网上最新文档不同就修改当前实现。

---

## 21. Git Discipline

开始任务前：

```bash
git status --short
```

结束任务后至少报告：

```bash
git status --short
git diff --stat
git diff
```

不要覆盖用户已有未提交修改。

不要自动：

```text
git commit
git push
git add tests/
```

除非任务明确要求。

不要修改无关文件来获得“干净 diff”。

---

## 22. Documentation

阶段状态统一维护：

```text
.ai/MIGRATION.md
```

它应该记录：

```text
当前是什么
当前 frozen contract
当前风险
当前下一任务
```

不要将其写成完整开发日志。

每完成一个 migration unit：

```text
1. production code 验证通过
2. local verification 通过
3. 更新对应 frozen contract
4. 更新 current stage / next task
5. 删除失效 TODO / 风险
```

README 和 migration 文档只能描述已经实际实现并验证的能力。

不要提前声称：

```text
rough complete
HIM complete
sim2real complete
```

---

## 23. Required Final Report

完成 coding task 后，至少报告：

```text
目标
读取的来源
production files
local verification files
行为变化
有意保持不变的行为
与 legacy 的差异
与 MjLab framework 的差异
验证命令
CPU 结果
CUDA 结果
尚未验证风险
git diff / git status
下一 migration unit
```

如果涉及 deployment，还应额外报告：

```text
policy input/output shape
joint mapping
observation layout
action semantics
TorchScript / model path
trace comparison result
```

---

## 24. Things Agents Must Not Do

除非当前 migration unit 明确要求，否则禁止：

```text
修改 frozen joint order
修改 action order
修改 observation layout
恢复 270-D HIM history 到 PPO
开始 HIM integration
开始 BlackW migration
开始 real robot backend
静默修改 deployment contract
根据 natural MJCF order 推断 policy order
修改 MjLab / RSL-RL dependency source
用 MjLab main API 替代 v1.6.0
增加独立 train.py / play.py
提交 tests/
进行无关架构重构
```

---

## 25. Decision Rule

当存在不确定性时，优先顺序是：

```text
实际目标仓库代码
↓
.ai/MIGRATION.md 中已经冻结的 project decision
↓
MjLab v1.6.0 实际能力
↓
明确指定的 authoritative external source
↓
legacy engineering experience
```

如果这些来源冲突：

```text
停止修改
整理差异
报告影响
提出推荐决策
```

不要自行隐藏冲突。

---

## 26. Core Principle

这个项目迁移的是：

```text
行为
参数语义
policy contract
已经验证的工程经验
```

目标架构仍然是：

```text
MjLab-native
explicit contract
small migration units
verifiable behavior
deployment-compatible
```

任何修改都应能回答：

```text
改了什么？
为什么改？
依据来自哪里？
哪些行为保持？
如何证明没有破坏已有 contract？
```
