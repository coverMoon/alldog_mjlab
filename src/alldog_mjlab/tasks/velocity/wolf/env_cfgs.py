"""Wolf velocity task 的 MjLab task assembly（flat / rough 共用；task local）。

term 顺序 contract、selector 绑定、sensor 装配都显式写在本文件；训练数值在
wolf_config.py，reward 数学在本任务的 rewards.py（与 black/rewards.py 公式独立
同构，不 import），16-D action 分组逻辑在本任务的 observations.py / rewards.py。

与 Black task 的关键结构差异：
- actor 角速度 / 投影重力来自 Wolf XML 原生 IMU sensor（robot/imu_*，§27.5）；
- action 为 8 个原生 action term 的交错拼接（每腿 3 pos + 1 wheel，共 16 维）；
- wheel 有独立 obs / reward 通道；
- critic 56 维（flat / rough 共用），**不含** height_scan（与 Black critic 追加
  187 维不同；Wolf rough 不给 actor / critic 新增高度图输入，保持 checkpoint
  shape 跨地形兼容）；terrain 扫描仅服务 rough 的 base_height reward；
- 终止：time_out + base_link 非法接触 + rough 的 out_of_terrain_bounds
  （无倾角 / 无 stuck）；play 下移除非法接触与越界终止；
- flat = plane terrain + 无 terrain curriculum；rough = generator terrain +
  native terrain_levels_vel + Wolf 性能驱动 command curriculum；
- flat / rough 共用全部机器人专科 / action / observation / reset / DR 契约。
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mdp_dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from alldog_mjlab.robots.wolf import get_wolf_robot_cfg
from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_JOINT_NAMES,
    WOLF_LEG_ORDER,
    WOLF_POLICY_JOINT_NAMES,
    WOLF_WHEEL_COLLISION_GEOM_NAMES,
    WOLF_WHEEL_FORWARD_SIGN,
    WOLF_WHEEL_JOINT_NAMES,
    WOLF_WHEEL_RADIUS,
)
from alldog_mjlab.tasks.velocity.wolf.curriculums import (
    CURRICULUM_TERM_NAME,
    WolfForwardSpeedCommandCurriculum,
)
from alldog_mjlab.tasks.velocity.wolf.him import (
    configure_him_observations,
    configure_him_terminal_targets,
)
from alldog_mjlab.tasks.velocity.wolf.rewards import (
    angular_velocity_xy_l2,
    base_height_l2_flat,
    base_height_l2_terrain,
    base_orientation_l1,
    hip_default_l1,
    leg_action_rate_l2,
    run_still_leg_l1,
    stand_still_leg_l1,
    track_angular_velocity_z,
    track_linear_velocity_xy,
    vertical_linear_velocity_l2,
    wheel_action_rate_l2,
)
from alldog_mjlab.tasks.velocity.wolf.terrain import wolf_rough_terrain_generator_cfg
from alldog_mjlab.tasks.velocity.wolf.randomization import (
    CalfBacklashPositionActionCfg,
    ScaledBiasedWheelVelocityActionCfg,
    BiasedSignedWheelVelocity,
    randomize_body_inertia_scale,
    randomize_ground_friction,
    randomize_leg_pd_gains_strength,
    randomize_wheel_friction_multiplier,
    randomize_wheel_motor_strength,
    randomize_wheel_radius_with_height,
    reset_joints_by_default_scale,
)
from alldog_mjlab.tasks.velocity.wolf.observations import signed_wheel_velocity
from alldog_mjlab.tasks.velocity.wolf.wolf_config import (
    WOLF_CONFIG,
    DomainRandomizationParams,
)

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

# ---------------------------------------------------------------------------
# Interface contracts（不属于训练调参，勿当作超参数阅读）
# ---------------------------------------------------------------------------

# Policy action term 顺序 contract（16 维交错拼接）：ActionManager 按 dict 插入
# 顺序切分 flat policy action。每腿 block 为 [hip thigh calf wheel]，共 4 维 × 4 腿。
WOLF_ACTION_TERM_ORDER = (
    "joint_pos_fl",
    "wheel_vel_fl",
    "joint_pos_fr",
    "wheel_vel_fr",
    "joint_pos_rl",
    "wheel_vel_rl",
    "joint_pos_rr",
    "wheel_vel_rr",
)
_WOLF_ACTION_TERM_DIMS = {"joint_pos": 3, "wheel_vel": 1}
WOLF_ACTION_DIM = (
    sum(_WOLF_ACTION_TERM_DIMS["joint_pos" if name.startswith("joint_pos") else "wheel_vel"] for name in WOLF_ACTION_TERM_ORDER)
)
assert WOLF_ACTION_DIM == 16
# 轮通道在 16-D raw action 中的下标（每 leg block 末位；供 16-D 分组推导交叉验证）。
WOLF_WHEEL_ACTION_CHANNELS = (3, 7, 11, 15)

# Wolf PPO actor 单帧 observation layout contract（共 53 维）：
#   command → base_ang_vel(IMU) → projected_gravity(IMU) → leg_joint_pos →
#   leg_joint_vel → signed_wheel_vel → last_raw_action
# 不含 base_lin_vel / 接触 / 足端高度 / history（单帧）。
WOLF_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "leg_joint_pos",
    "leg_joint_vel",
    "wheel_vel",
    "actions",
)
WOLF_ACTOR_OBS_DIM = 3 + 3 + 3 + 12 + 12 + 4 + 16
assert WOLF_ACTOR_OBS_DIM == 53

# Wolf critic observation = actor 53 维同序 + nominal scale 无噪声 + 末尾 3 维
# 真实 body-frame base linear velocity（共 56 维）。
WOLF_CRITIC_TERM_ORDER = WOLF_ACTOR_OBS_TERM_ORDER + ("base_lin_vel",)
WOLF_CRITIC_DIM = WOLF_ACTOR_OBS_DIM + 3
assert WOLF_CRITIC_DIM == 56

# actor term → scale / noise 的映射属于 observation wiring，集中在这里。
_WOLF_ACTOR_TERM_SCALE = {
    "command": WOLF_CONFIG.observation.command_scale,
    "base_ang_vel": WOLF_CONFIG.observation.base_ang_vel_scale,
    "projected_gravity": WOLF_CONFIG.observation.projected_gravity_scale,
    "leg_joint_pos": WOLF_CONFIG.observation.joint_pos_scale,
    "leg_joint_vel": WOLF_CONFIG.observation.joint_vel_scale,
    "wheel_vel": WOLF_CONFIG.observation.wheel_vel_scale,
    "actions": WOLF_CONFIG.observation.last_action_scale,
}
_WOLF_ACTOR_TERM_NOISE = {
    "base_ang_vel": WOLF_CONFIG.noise.base_ang_vel,
    "projected_gravity": WOLF_CONFIG.noise.projected_gravity,
    "leg_joint_pos": WOLF_CONFIG.noise.joint_pos,
    "leg_joint_vel": WOLF_CONFIG.noise.joint_vel,
    "wheel_vel": WOLF_CONFIG.noise.wheel_vel,
}

# 非法接触终止 sensor 身份：illegal_contact_bodies（config 可配）对 terrain。
WOLF_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"

# rough terrain scan sensor 身份 contract（只存在于 rough 任务）：MjLab v1.6
# 原生 RayCastSensorCfg，17 x 11 = 187 rays，ray yaw 对齐，仅碰撞 group 0（terrain）。
# raw clearance 语义由 native ``envs_mdp.height_scan`` 提供（frame z - hit z）。
# flat 不注册该 sensor。
WOLF_TERRAIN_SCAN_SENSOR = "terrain_scan"

# rough terrain curriculum 的注册名（native ``terrain_levels_vel``；仅 rough train）。
WOLF_TERRAIN_CURRICULUM_TERM = "terrain_levels"

# 编译模板 spawn 状态的 root body（SceneCfg.spec_fn 目标；attach 后带实体前缀）。
WOLF_TEMPLATE_ROOT_BODY = "robot/base_link"

# Command term 名称（task wiring：reward / termination / curriculum 都按名取它）。
WOLF_COMMAND_NAME = "twist"

# Tracking 线速度 reward term 名：command curriculum 的 performance 采样来源。
WOLF_TRACKING_VELOCITY_REWARD_TERM = "track_linear_velocity"

# 腿 / 轮的执行器选择器（显式 policy 顺序驱动，不依赖 MJCF natural order）。
WOLF_LEG_ACTION_JOINT_NAMES = {
    leg: tuple(WOLF_LEG_JOINT_NAMES[leg][:3]) for leg in WOLF_LEG_ORDER  # hip/thigh/calf
}
WOLF_WHEEL_JOINT_NAME = {leg: (WOLF_LEG_JOINT_NAMES[leg][3],) for leg in WOLF_LEG_ORDER}
WOLF_LEGS_FL_FIRST_JOINT_NAMES = tuple(
    name for leg in WOLF_LEG_ORDER for name in WOLF_LEG_ACTION_JOINT_NAMES[leg]
)
WOLF_WHEEL_JOINT_POLICY_ORDER = tuple(f"{leg}_foot" for leg in WOLF_LEG_ORDER)


def _wolf_flat_template_spec_fn(spec: "mujoco.MjSpec") -> None:
    """Wolf flat 编译模板 spawn 高度回调（模块顶层，保证 yaml 可表示）。

    背景（MJWarp 容量诊断，tests/diag_wolf_capacity.py）：Wolf MJCF 编译后
    ``qpos0`` 的 root z=0（自由关节 qpos0 z 来自 body pos，INIT_STATE 的
    z=0.4432 只进入 ``init_state`` keyframe，不进 qpos0），模板 spawn 状态下
    机体与全部腿碰撞 geom 大深度插入 plane，模板 ncon=124 / nefc=496，被
    ``put_data`` 作为硬下限（io.py:1931/1951），把 nconmax/njmax 钉在 128/512。

    本回调把模板 root body 位置改为 z=WOLF_CONFIG.simulation.template_root_z
    （0.45），使编译后 ``qpos0`` root z=0.45、模板 ncon/nefc 大幅下降，从而容量
    改由运行时需求决定（活动样本峰值 ncon≤16 / nefc≤40）。已验证效果：
    ncon 124→24（仅抬 z、关节零位）、0（关节 default 时）。

    不改变的量：
    - ``default_root_state``（实体 cfg `INIT_STATE`，z=0.4432）——训练 reset
      高度语义不变（实测 reset 后 root z 仍为 0.4432）;
    - hinge joint `ref`（保持 0.0）、关节几何、default joint pose、actuator、
      quat（模板 root quat 保持单位）；
    - 不写编译后的 ``MjModel.qpos0``（qpos0 由 MuJoCo 正常编译流程从 spec
      生成），不替换 ``MjData``。

    必须定义在模块顶层：train CLI 用 ``asdict`` + ``yaml.dump`` 保存 env.yaml，
    yaml 的 ``!!python/name:`` 标签只能表示有稳定可导入名称的对象；嵌套局部
    函数的 qualname 含 ``<locals>``，dump/load 均不可靠。
    """
    body = next((b for b in spec.bodies if b.name == WOLF_TEMPLATE_ROOT_BODY), None)
    if body is None:
        raise ValueError(
            f"Wolf 模板 spec_fn：找不到根 body {WOLF_TEMPLATE_ROOT_BODY!r}，"
            "spec 结构与预期不符，fail-loud"
        )
    # 自由关节名来自 MJCF（robot/floating_base_joint），不属于 root body 名前缀；
    # 用“全 spec 恰好 1 个 freejoint”+ 根 body 名校验共同判定。
    free_joints = [
        j
        for j in spec.joints
        if int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise ValueError(
            "Wolf 模板 spec_fn：根 body 下自由关节数量应为 1，实际 "
            f"{len(free_joints)}（joints={[j.name for j in free_joints]}），fail-loud"
        )
    # 抬升模板 root 高度（写入 spec，由 MuJoCo 正常编译流程产生 qpos0）。
    body.pos[:] = [0.0, 0.0, WOLF_CONFIG.simulation.template_root_z]


def _configure_template_spawn_height(cfg: ManagerBasedRlEnvCfg) -> None:
    """把模块顶层 spec 回调挂到 ``cfg.scene.spec_fn``（仅 flat 装配路径调用）。"""
    if cfg.scene.spec_fn is not None:
        raise ValueError(
            "cfg.scene.spec_fn 已存在（Wolf 不覆盖既有 spec_fn；"
            "baseline 未使用该字段，出现该值说明装配链路有未知来源）"
        )
    cfg.scene.spec_fn = _wolf_flat_template_spec_fn


def _configure_command(cfg: ManagerBasedRlEnvCfg) -> None:
    """Wolf command contract：初始 vx 范围 + native sampler + 课程起点。

    语义与 Black `_configure_command` 一致：heading / world-frame 关闭、
    standing / forward-only 比例冻结；`lin_vel_x` 是课程起点（train 下由
    curriculum 扩展到 ±4 m/s），`lin_vel_y` / `ang_vel_z` 训练全程固定。
    数值全部来自 WOLF_CONFIG。
    """
    twist_command = cfg.commands[WOLF_COMMAND_NAME]
    assert isinstance(twist_command, UniformVelocityCommandCfg)
    twist_command.resampling_time_range = WOLF_CONFIG.command.resampling_time
    twist_command.ranges.lin_vel_x = WOLF_CONFIG.command.lin_vel_x
    twist_command.ranges.lin_vel_y = WOLF_CONFIG.command.lin_vel_y
    twist_command.ranges.ang_vel_z = WOLF_CONFIG.command.ang_vel_z
    twist_command.heading_command = False
    twist_command.rel_heading_envs = 0.0
    twist_command.ranges.heading = None
    twist_command.rel_standing_envs = WOLF_CONFIG.command.standing_fraction
    twist_command.rel_forward_envs = WOLF_CONFIG.command.forward_fraction
    twist_command.rel_world_envs = WOLF_CONFIG.command.world_fraction
    twist_command.init_velocity_prob = WOLF_CONFIG.command.init_velocity_prob
    # 移除 baseline 自带的 staged（time-based）velocity curriculum：Wolf 采用
    # 性能驱动（tracking EMA）课程，不用 step counter 阶段。
    cfg.curriculum.pop("command_vel", None)


def _wolf_robot_cfg(dr: DomainRandomizationParams) -> "EntityCfg":
    """构造 Wolf robot cfg；DR 开启时给对应 actuator 注入原生 actuator delay。

    delay 单位换算：legacy 的 3/4 policy step → ×decimation physics step。
    lag 在 [0, max] 内每 update_period（= 1 policy step）重采样一次；与旧版
    per-episode 固定 lag 的差异在 MIGRATION.md §29 标注为有意改变。
    关闭时不设置 delay（无 buffer，无残余延迟）。
    """
    import copy

    robot_cfg = get_wolf_robot_cfg()
    if not (dr.leg_delay_enabled or dr.wheel_delay_enabled):
        return robot_cfg
    robot_cfg = copy.deepcopy(robot_cfg)
    decimation = WOLF_CONFIG.control.decimation
    for act in robot_cfg.articulation.actuators:
        joint_name = act.target_names_expr[0]
        if joint_name in WOLF_WHEEL_JOINT_NAMES:
            if dr.wheel_delay_enabled:
                max_lag = dr.wheel_max_delay_steps * decimation
                act.delay_min_lag = 0
                act.delay_max_lag = max_lag
                act.delay_update_period = decimation
                act.delay_per_env_phase = False
        else:
            if dr.leg_delay_enabled:
                max_lag = dr.leg_max_delay_steps * decimation
                act.delay_min_lag = 0
                act.delay_max_lag = max_lag
                act.delay_update_period = decimation
                act.delay_per_env_phase = False
    return robot_cfg


def _configure_scene_and_sensors(
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams, rough: bool
) -> None:
    """robot entity 与 sensor 装配；移除 baseline 的 terrain / foot 传感器。

    Wolf 用不到 foot site / height scan：轮足结构没有足端摆动相概念，flat plane 的
    base height reward 直接读 world z，critic 也不需要 terrain height。rough 用
    Wolf 自己的 terrain_scan（等价 flat/rough 共用 sensor 的处理，按后重建）。
    DR 的 actuator delay 在 robot cfg 层注入（见 _wolf_robot_cfg）。
    """
    cfg.scene.entities = {"robot": _wolf_robot_cfg(dr)}
    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name not in ("foot_height_scan", "terrain_scan")
    )
    if rough:
        _configure_rough_terrain_scan(cfg)

    illegal_ground_contact = ContactSensorCfg(
        name=WOLF_ILLEGAL_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="body",
            pattern=WOLF_CONFIG.termination.illegal_contact_bodies,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=WOLF_CONFIG.termination.illegal_contact_history,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (illegal_ground_contact,)


def _configure_actions(
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams
) -> None:
    """Wolf policy action contract：每腿 1 个位置 term + 1 个轮速度 term 交错拼接。

    flat policy action 顺序 = WOLF_ACTION_TERM_ORDER（每腿 [hip thigh calf wheel]
    block，FL → FR → RL → RR，共 16 维）。

    - 腿部：`JointPositionActionCfg`（use_default_offset=True）→
      ``q_target = q_default + 0.20 * raw``；DR 开启时用带 calf backlash 状态机
      的子类 cfg（term 名 / 维度 / 顺序不变；backlash 关闭时与原生逐位等价）；
    - 轮部：`JointVelocityActionCfg`，scale 为 per-wheel dict（含 forward sign）→
      ``dq_target = forward_sign * 10.0 * raw``；DR 开启时用带 target
      scaling / bias 的子类 cfg（关闭时与原生逐位等价）；velocity target 直接
      写入理想 PD 执行器（Kp=0 / Kd=1 / limit 17 N·m，robots/wolf 冻结值）；
    - 不做默认 [-1,1] action clip（GaussianDistribution 初 std 1.0，raw 无界）。

    每关节一个 IdealPd 执行器（sort_actuators=True），`find_joints_by_actuator_names`
    以显式名字列表逐 term 选择器定位，不依赖 MJCF natural order。
    """
    cfg.actions.pop("joint_pos")

    backlash_on = dr.calf_backlash.enabled
    wheel_target_on = (
        dr.wheel_target.vel_ref_scale_enabled or dr.wheel_target.vel_ref_bias_enabled
    )

    def _position_cfg(leg: str) -> JointPositionActionCfg:
        base = dict(
            entity_name="robot",
            actuator_names=WOLF_LEG_ACTION_JOINT_NAMES[leg],
            scale=WOLF_CONFIG.control.leg_action_scale,
            use_default_offset=True,
        )
        if not backlash_on:
            return JointPositionActionCfg(**base)
        return CalfBacklashPositionActionCfg(
            **base,
            backlash_enabled=True,
            width_range=dr.calf_backlash.width_range,
            min_kp_scale=dr.calf_backlash.min_kp_scale,
            engage_start=dr.calf_backlash.engage_start,
            leak=dr.calf_backlash.leak,
        )

    def _velocity_cfg(leg: str) -> JointVelocityActionCfg:
        base = dict(
            entity_name="robot",
            actuator_names=WOLF_WHEEL_JOINT_NAME[leg],
            scale={
                WOLF_LEG_JOINT_NAMES[leg][3]: (
                    WOLF_CONFIG.control.wheel_action_scale
                    * WOLF_WHEEL_FORWARD_SIGN[leg]
                )
            },
            use_default_offset=True,
        )
        if not wheel_target_on:
            return JointVelocityActionCfg(**base)
        return ScaledBiasedWheelVelocityActionCfg(
            **base,
            vel_ref_scale_enabled=dr.wheel_target.vel_ref_scale_enabled,
            vel_ref_scale_range=dr.wheel_target.vel_ref_scale_range,
            vel_ref_bias_enabled=dr.wheel_target.vel_ref_bias_enabled,
            vel_ref_bias_range=dr.wheel_target.vel_ref_bias_range,
        )

    action_terms = {}
    for leg in WOLF_LEG_ORDER:
        action_terms[f"joint_pos_{leg.lower()}"] = _position_cfg(leg)
        action_terms[f"wheel_vel_{leg.lower()}"] = _velocity_cfg(leg)
    assert tuple(action_terms) == WOLF_ACTION_TERM_ORDER
    cfg.actions = action_terms  # type: ignore[assignment]


def _configure_events(
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams, rough: bool
) -> None:
    """Reset contract + DR event contract（dict 顺序 = 同 mode 内应用顺序）。

    reset_base：root pose 不随机（default 站立，root 高度 = INIT_STATE z），六维
    速度小幅独立扰动；leg 关节按 hip / thigh / calf 三组小幅 offset；轮子显式
    归零位零速（不留隐式依赖）。

    DR（旧 BlackW 候选范围，全部默认关闭）：地面摩擦 → 轮摩擦乘子（依赖前序
    事件的当前值，顺序敏感）→ 质量 / COM / 惯量（均相对 nominal，无累计漂移）
    → 腿 / 轮执行器 → 轮半径（含初始高度补偿）→ 外部扰动。DR 开启时，
    initial_joint_pos 用 multiplicative reset 替换腿关节 offset reset。
    """
    base_events = dict(cfg.events)
    base_joint_reset = base_events["reset_robot_joints"]
    assert base_joint_reset.func is reset_joints_by_offset

    reset_base = replace(
        base_events["reset_base"],
        params={
            "pose_range": dict(WOLF_CONFIG.reset.root_pose),
            "velocity_range": dict(WOLF_CONFIG.reset.root_velocity),
        },
    )
    if dr.initial_joint_pos_enabled:
        # DR：multiplicative reset（旧版 _reset_dofs 语义），替换三组 offset reset。
        leg_resets = {
            "reset_initial_joint_pos": EventTermCfg(
                func=reset_joints_by_default_scale,
                mode="reset",
                params={
                    "scale_range": dr.initial_joint_pos_range,
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    ),
                },
            ),
        }
    else:
        leg_resets = {
            f"reset_{joint_group}_joints": replace(
                base_joint_reset,
                params={
                    "position_range": position_range,
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=tuple(f"{leg}_{joint_group}" for leg in WOLF_LEG_ORDER),
                    ),
                },
            )
            for joint_group, position_range in WOLF_CONFIG.reset.leg_position.items()
        }
    wheel_reset = replace(
        base_joint_reset,
        params={
            "position_range": WOLF_CONFIG.reset.wheel_position,
            "velocity_range": WOLF_CONFIG.reset.wheel_velocity,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=WOLF_WHEEL_JOINT_POLICY_ORDER
            ),
        },
    )
    cfg.events = {
        "reset_base": reset_base,
        **leg_resets,
        "reset_wheel_joints": wheel_reset,
        **_dr_event_terms(dr, rough),
    }
    # baseline 的 push / foot_friction / encoder_bias / base_com DR 事件全部不注册
    # （Wolf DR 事件统一在 _dr_event_terms 中显式构造）。


def _configure_rewards(cfg: ManagerBasedRlEnvCfg, rough: bool) -> None:
    """Wolf reward baseline：四任务同一套 13 项（dict 顺序即 logging 顺序）。

    tracking / 罚项 / 姿态公式在本任务 rewards.py（与 black/rewards.py 公式
    独立同构，不 import）；action rate 按显式 16-D contract 的 leg / wheel 分组。
    姿态奖励（upright L1 / hip_default / stand_still / run_still /
    dof_pos_limits / leg_torques）为 Wolf v1/v2 统一扩展（BlackW 经验迁移），
    flat / rough 共用同一公式 / 权重 / 选关节（有意修改 rough 原 8 项基线，
    见 MIGRATION §28.5）。
    唯一的 flat / rough 差异仍是 ``base_height`` func：flat = world-z（
    base_height_l2_flat）、rough = terrain-relative footprint 均值
    （base_height_l2_terrain）；target / weight / key / 顺序完全一致。
    """
    base_height_term = (
        RewardTermCfg(
            func=base_height_l2_terrain,
            weight=WOLF_CONFIG.reward.scales.base_height,
            params={
                "target_height": WOLF_CONFIG.reward.base_height_target,
                "sensor_name": WOLF_TERRAIN_SCAN_SENSOR,
            },
        )
        if rough
        else RewardTermCfg(
            func=base_height_l2_flat,
            weight=WOLF_CONFIG.reward.scales.base_height,
            params={
                "target_height": WOLF_CONFIG.reward.base_height_target,
            },
        )
    )
    posture = WOLF_CONFIG.reward.posture
    cfg.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=track_linear_velocity_xy,
            weight=WOLF_CONFIG.reward.scales.tracking_linear,
            params={
                "command_name": WOLF_COMMAND_NAME,
                "sigma": WOLF_CONFIG.reward.tracking_sigma,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=track_angular_velocity_z,
            weight=WOLF_CONFIG.reward.scales.tracking_angular,
            params={
                "command_name": WOLF_COMMAND_NAME,
                "sigma": WOLF_CONFIG.reward.tracking_sigma,
            },
        ),
        "lin_vel_z": RewardTermCfg(
            func=vertical_linear_velocity_l2,
            weight=WOLF_CONFIG.reward.scales.lin_vel_z,
        ),
        "body_ang_vel": RewardTermCfg(
            func=angular_velocity_xy_l2,
            weight=WOLF_CONFIG.reward.scales.ang_vel_xy,
        ),
        # L1 公式（BlackW 经验，见 rewards.base_orientation_l1；四任务统一）。
        "upright": RewardTermCfg(
            func=base_orientation_l1,
            weight=WOLF_CONFIG.reward.scales.upright,
        ),
        "base_height": base_height_term,
        "leg_action_rate": RewardTermCfg(
            func=leg_action_rate_l2,
            weight=WOLF_CONFIG.reward.scales.leg_action_rate,
        ),
        "wheel_action_rate": RewardTermCfg(
            func=wheel_action_rate_l2,
            weight=WOLF_CONFIG.reward.scales.wheel_action_rate,
        ),
    }
    # 腿部 12 关节（hip/thigh/calf）显式选择；actuator 名 = 目标关节名
    # （IdealPdActuator 在编译时用关节名命名，见 rewards / wolf_constants）。
    leg_cfg = SceneEntityCfg(
        "robot", joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES, preserve_order=True
    )
    leg_actuator_cfg = SceneEntityCfg(
        "robot", actuator_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES
    )
    cfg.rewards["hip_default"] = RewardTermCfg(
        func=hip_default_l1,
        weight=WOLF_CONFIG.reward.scales.hip_default,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=tuple(
                    WOLF_LEG_JOINT_NAMES[leg][0] for leg in WOLF_LEG_ORDER
                ),
                preserve_order=True,
            ),
            "command_name": WOLF_COMMAND_NAME,
            "y_ref": posture.hip_y_ref,
            "yaw_ref": posture.hip_yaw_ref,
            "y_scale": posture.hip_y_scale,
            "yaw_scale": posture.hip_yaw_scale,
            "min_scale": posture.hip_min_scale,
        },
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=stand_still_leg_l1,
        weight=WOLF_CONFIG.reward.scales.stand_still,
        params={
            "asset_cfg": leg_cfg,
            "command_name": WOLF_COMMAND_NAME,
            "lin_threshold": posture.stand_still_lin_threshold,
            "yaw_threshold": posture.stand_still_yaw_threshold,
        },
    )
    cfg.rewards["run_still"] = RewardTermCfg(
        func=run_still_leg_l1,
        weight=WOLF_CONFIG.reward.scales.run_still,
        params={
            "asset_cfg": leg_cfg,
            "command_name": WOLF_COMMAND_NAME,
            "x_threshold": posture.run_still_x_threshold,
            "y_threshold": posture.run_still_y_threshold,
            "yaw_threshold": posture.run_still_yaw_threshold,
        },
    )
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=envs_mdp.joint_pos_limits,
        weight=WOLF_CONFIG.reward.scales.dof_pos_limits,
        params={"asset_cfg": leg_cfg},
    )
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=envs_mdp.joint_torques_l2,
        weight=WOLF_CONFIG.reward.scales.leg_torques,
        params={"asset_cfg": leg_actuator_cfg},
    )
    assert WOLF_TRACKING_VELOCITY_REWARD_TERM in cfg.rewards
    assert len(cfg.rewards) == 13


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """flat task specialization：plane terrain、无 terrain generator / curriculum。"""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop(WOLF_TERRAIN_CURRICULUM_TERM, None)


def _configure_rough_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough task specialization：curriculum terrain generator（一个 terrain 一列）。

    generator 内部 ``curriculum=True``，因此列数 = terrain 类型数（7），
    ``proportion`` 是 env 分配权重。terrain curriculum（``terrain_levels_vel``）
    的注册在 builder 的 curriculum 阶段处理（仅 rough train）。
    """
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = wolf_rough_terrain_generator_cfg()
    cfg.scene.terrain.max_init_terrain_level = (
        WOLF_CONFIG.terrain.max_init_terrain_level
    )


def _configure_rough_terrain_scan(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough 专用 native RayCastSensorCfg：base_link frame、yaw 对齐、187 rays。

    flat 不注册该 sensor（不增加不必要的地形传感器）。数值来自 WOLF_CONFIG.terrain
    的 sensor 参数（当前与 MjLab velocity baseline / Black rough 的 terrain_scan
    同值）。raw clearance 语义由 native ``envs_mdp.height_scan`` 提供，本文件不做
    二次包装（base_height reward 直接消费 ``base_height_l2_terrain``）。
    """
    t = WOLF_CONFIG.terrain
    sensor = RayCastSensorCfg(
        name=WOLF_TERRAIN_SCAN_SENSOR,
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=t.terrain_scan_size, resolution=t.terrain_scan_resolution),
        max_distance=t.terrain_scan_max_distance,
        exclude_parent_body=True,
        include_geom_groups=(0,),  #Terrain only.
        debug_vis=True,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor,)


def _wheel_vel_term_cfg(dr: DomainRandomizationParams) -> ObservationTermCfg:
    """actor wheel_vel term：DR 开启时用带 per-episode bias 的类 term（仅 actor）。

    critic 构造时显式替换回无 bias 的真实 signed_wheel_velocity。
    """
    wheel_obs = SceneEntityCfg(
        "robot", joint_names=WOLF_WHEEL_JOINT_POLICY_ORDER, preserve_order=True
    )
    if not dr.wheel_obs_bias.enabled:
        return ObservationTermCfg(
            func=signed_wheel_velocity,
            params={"asset_cfg": wheel_obs},
            scale=_WOLF_ACTOR_TERM_SCALE["wheel_vel"],
            noise=UniformNoiseCfg(
                n_min=_WOLF_ACTOR_TERM_NOISE["wheel_vel"][0],
                n_max=_WOLF_ACTOR_TERM_NOISE["wheel_vel"][1],
            ),
        )
    return ObservationTermCfg(
        func=BiasedSignedWheelVelocity,
        params={
            "asset_cfg": wheel_obs,
            "bias_range": dr.wheel_obs_bias.range,
            "enabled": True,
        },
        scale=_WOLF_ACTOR_TERM_SCALE["wheel_vel"],
        noise=UniformNoiseCfg(
            n_min=_WOLF_ACTOR_TERM_NOISE["wheel_vel"][0],
            n_max=_WOLF_ACTOR_TERM_NOISE["wheel_vel"][1],
        ),
    )


def _configure_observations(
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams
) -> None:
    """Wolf PPO actor 53 维 / critic 56 维 observation contract（顺序 / scale / noise）。

    IMU 观测走 Wolf XML 原生 sensor（`robot/imu_ang_vel` / `robot/imu_upvector`，
    数值语义见 §27.5 与 tests/check_wolf_imu.py：gyro = IMU 局部系角速度，
    projected_gravity = -upvector = R_world_imu.T @ (0,0,-1)）。实装 IMU 后观测反映
    IMU site 姿态而非 base_link，部署侧同源。

    leg 观测用 policy 腿顺序（FL→FR→RL→RR，hip→thigh→calf）；轮速观测乘
    forward sign（正值 = 该轮机身 +x 前进方向角速度）。critic 同布局加末尾
    3 维真实 base_lin_vel，且完全无 noise（不与 actor 共享 cfg 对象）。
    """
    actor_terms = cfg.observations["actor"].terms
    # baseline 项清理（显式移除，不留无效引用）。
    actor_terms.pop("base_lin_vel", None)
    actor_terms.pop("base_ang_vel", None)
    actor_terms.pop("projected_gravity", None)
    actor_terms.pop("height_scan", None)

    def _actor_term(
        name: str,
        func,
        params: dict | None = None,
    ) -> tuple[str, ObservationTermCfg]:
        noise_range = _WOLF_ACTOR_TERM_NOISE.get(name)
        term = ObservationTermCfg(
            func=func,
            params=dict(params or {}),
            scale=_WOLF_ACTOR_TERM_SCALE[name],
            noise=(
                UniformNoiseCfg(n_min=noise_range[0], n_max=noise_range[1])
                if noise_range is not None
                else None
            ),
        )
        return name, term

    actor_built = dict(
        (
            _actor_term("command", mdp.generated_commands, {"command_name": WOLF_COMMAND_NAME}),
            _actor_term("base_ang_vel", mdp.builtin_sensor, {"sensor_name": "robot/imu_ang_vel"}),
            _actor_term(
                "projected_gravity",
                mdp.projected_gravity_from_sensor,
                {"sensor_name": "robot/imu_upvector"},
            ),
            _actor_term(
                "leg_joint_pos",
                mdp.joint_pos_rel,
                {"asset_cfg": SceneEntityCfg("robot", joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES, preserve_order=True)},
            ),
            _actor_term(
                "leg_joint_vel",
                mdp.joint_vel_rel,
                {"asset_cfg": SceneEntityCfg("robot", joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES, preserve_order=True)},
            ),
            ("wheel_vel", _wheel_vel_term_cfg(dr)),
            _actor_term("actions", mdp.last_action),
        )
    )
    assert tuple(actor_built) == WOLF_ACTOR_OBS_TERM_ORDER
    cfg.observations["actor"].terms = actor_built
    cfg.observations["actor"].enable_corruption = (
        WOLF_CONFIG.observation.actor_corruption_enabled and WOLF_CONFIG.noise.enabled
    )

    # critic：同布局，全部无 noise（独立构造 cfg 对象），末尾追加真实 base_lin_vel。
    # wheel_vel：若 actor 使用带 bias 的类 term，critic 显式替换回真实值
    # （DR 的轮速观测偏置不得污染 critic privileged observation）。
    critic_built: dict[str, ObservationTermCfg] = {}
    for name, term in actor_built.items():
        func = term.func
        params = dict(term.params)
        if name == "wheel_vel" and dr.wheel_obs_bias.enabled:
            func = signed_wheel_velocity
            params = {"asset_cfg": params["asset_cfg"]}
        critic_built[name] = ObservationTermCfg(
            func=func,
            params=params,
            scale=term.scale,
        )
    critic_built["base_lin_vel"] = ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        params={},
        scale=1.0,  # raw m/s，不缩放
    )
    assert tuple(critic_built) == WOLF_CRITIC_TERM_ORDER
    cfg.observations["critic"].terms = critic_built
    cfg.observations["critic"].enable_corruption = False


def _configure_terminations(cfg: ManagerBasedRlEnvCfg, rough: bool) -> None:
    """终止 contract：time_out + illegal_contact（base_link 对 terrain）。

    baseline 的 fell_over（倾角终止）显式移除；stuck termination 本轮不注册。
    rough 追加 native ``out_of_terrain_bounds``（time_out=True：有限生成地形边界的
    人工截断，不是机器人 physical failure，由 PPO 做 value bootstrap）；flat 不注册。
    """
    cfg.terminations.pop("out_of_terrain_bounds", None)
    if rough:
        cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            time_out=True,
        )
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": WOLF_ILLEGAL_CONTACT_SENSOR,
            "force_threshold": WOLF_CONFIG.termination.illegal_contact_force,
        },
    )


def _dr_event_terms(
    dr: DomainRandomizationParams, rough: bool
) -> dict[str, EventTermCfg]:
    """Wolf DR 事件表（仅显式开启的项注册；dict 顺序 = 同 mode 内应用顺序）。

    顺序敏感点：
    - ``dr_ground_friction``（绝对值写全部机器人 collision geom + 地面 geom 压
      0）必须先于 ``dr_wheel_friction``（乘子，读当前值）；
    - ``dr_wheel_radius`` 必须在 ``reset_base`` 之后（读改当前 root pose 的
      Z 分量做高度补偿，保留 XY / quat / velocity）。
    """
    terms: dict[str, EventTermCfg] = {}

    all_collision_geoms = SceneEntityCfg("robot", geom_names=(".*",))
    wheel_geoms = SceneEntityCfg(
        "robot", geom_names=WOLF_WHEEL_COLLISION_GEOM_NAMES, preserve_order=True
    )
    # 地面 geom（terrain 实体的碰撞 geom；摩擦事件把它们压 0，见 randomization 模块）。
    # flat plane 是单个命名 "terrain" 的 geom；rough generator 的全部 patch geom
    # （含未命名 hfield / box）用 ".*" 显式选中；geom 选择由 randomization 侧
    # fail-loud 防空（ selector 无匹配时报错，不允许静默无效 DR）。
    terrain_geoms = (
        SceneEntityCfg("terrain", geom_names=(".*",))
        if rough
        else SceneEntityCfg("terrain", geom_names=("terrain",))
    )
    base_body = SceneEntityCfg("robot", body_names=("base_link",))
    wheel_bodies = SceneEntityCfg(
        "robot", body_names=tuple(f"{leg}_Link4" for leg in WOLF_LEG_ORDER)
    )
    # Link1-3（非 base、非轮）：link mass / inertia 随机化目标。
    link_bodies = SceneEntityCfg(
        "robot",
        body_names=tuple(
            f"{leg}_Link{i}" for leg in WOLF_LEG_ORDER for i in (1, 2, 3)
        ),
    )
    leg_actuators = SceneEntityCfg(
        "robot",
        actuator_names=tuple(
            f"{leg}_{joint}" for leg in WOLF_LEG_ORDER for joint in ("hip", "thigh", "calf")
        ),
    )
    wheel_actuators = SceneEntityCfg(
        "robot", actuator_names=WOLF_WHEEL_JOINT_NAMES
    )

    # --- 摩擦（reset，per-episode 重采样；轮乘子顺序敏感）---
    # 地面 geom 由事件压 0，保证 wheel-ground contact 摩擦不被地面默认 1.0
    # 经 max() 合成抬高下限（采样范围低段 / 乘子结果可达）。
    if dr.ground_friction_enabled:
        terms["dr_ground_friction"] = EventTermCfg(
            func=randomize_ground_friction,
            mode="reset",
            params={
                "asset_cfg": all_collision_geoms,
                "terrain_cfg": terrain_geoms,
                "friction_range": dr.ground_friction_range,
            },
        )
    if dr.wheel_friction_enabled:
        terms["dr_wheel_friction"] = EventTermCfg(
            func=randomize_wheel_friction_multiplier,
            mode="reset",
            params={
                "scale_range": dr.wheel_friction_scale_range,
                "asset_cfg": wheel_geoms,
                "terrain_cfg": terrain_geoms,
                "from_current_base": dr.ground_friction_enabled,
            },
        )

    # --- 刚体质量 / COM / 惯量（reset，相对 nominal，无累计漂移）---
    if dr.base_mass_enabled:
        terms["dr_base_mass"] = EventTermCfg(
            func=mdp_dr.body_mass,
            mode="reset",
            params={
                "asset_cfg": base_body,
                "operation": "add",
                "ranges": dr.base_mass_range,
            },
        )
    if dr.link_mass_enabled:
        terms["dr_link_mass"] = EventTermCfg(
            func=mdp_dr.body_mass,
            mode="reset",
            params={
                "asset_cfg": link_bodies,
                "operation": "scale",
                "ranges": dr.link_mass_scale_range,
            },
        )
    if dr.wheel_mass_enabled:
        terms["dr_wheel_mass"] = EventTermCfg(
            func=mdp_dr.body_mass,
            mode="reset",
            params={
                "asset_cfg": wheel_bodies,
                "operation": "scale",
                "ranges": dr.wheel_mass_scale_range,
            },
        )
    if dr.link_inertia_enabled:
        terms["dr_link_inertia"] = EventTermCfg(
            func=randomize_body_inertia_scale,
            mode="reset",
            params={
                "ranges": dr.link_inertia_scale_range,
                "asset_cfg": link_bodies,
                "shared_random": False,
            },
        )
    if dr.wheel_inertia_enabled:
        terms["dr_wheel_inertia"] = EventTermCfg(
            func=randomize_body_inertia_scale,
            mode="reset",
            params={
                "ranges": dr.wheel_inertia_scale_range,
                "asset_cfg": wheel_bodies,
                "shared_random": True,
            },
        )
    if dr.base_com_enabled:
        terms["dr_base_com"] = EventTermCfg(
            func=mdp_dr.body_com_offset,
            mode="reset",
            params={
                "asset_cfg": base_body,
                "operation": "add",
                "ranges": {
                    0: dr.base_com_offset_range,
                    1: dr.base_com_offset_range,
                    2: dr.base_com_offset_range,
                },
            },
        )

    # --- 执行器（reset）---
    if dr.kp_enabled or dr.kd_enabled or dr.motor_strength_enabled or dr.hip_motor_strength_enabled:
        terms["dr_leg_pd_gains"] = EventTermCfg(
            func=randomize_leg_pd_gains_strength,
            mode="reset",
            params={
                "kp_range": dr.kp_scale_range if dr.kp_enabled else (1.0, 1.0),
                "kd_range": dr.kd_scale_range if dr.kd_enabled else (1.0, 1.0),
                "motor_strength_range": (
                    dr.motor_strength_range if dr.motor_strength_enabled else (1.0, 1.0)
                ),
                "hip_strength_range": (
                    dr.hip_motor_strength_range if dr.hip_motor_strength_enabled else None
                ),
                "asset_cfg": leg_actuators,
            },
        )
    if dr.wheel_motor_enabled:
        terms["dr_wheel_motor"] = EventTermCfg(
            func=randomize_wheel_motor_strength,
            mode="reset",
            params={
                "strength_range": dr.wheel_motor_strength_range,
                "asset_cfg": wheel_actuators,
            },
        )

    # --- 轮几何（reset，含初始高度补偿；在 reset_base 之后读改当前 root pose Z）---
    if dr.wheel_radius_enabled:
        terms["dr_wheel_radius"] = EventTermCfg(
            func=randomize_wheel_radius_with_height,
            mode="reset",
            params={
                "scale_range": dr.wheel_radius_scale_range,
                "asset_cfg": wheel_geoms,
            },
        )

    # --- reset：initial joint position（multiplicative）已在 _configure_events 中
    # 替换腿 offset reset，不在此重复注册。---

    # --- 外部扰动（interval，全局同步）---
    if dr.push_enabled:
        terms["dr_push_robot"] = EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=dr.push_interval_s,
            is_global_time=True,
            params={
                "velocity_range": {
                    "x": dr.push_velocity_xy,
                    "y": dr.push_velocity_xy,
                },
            },
        )
    if dr.disturbance_enabled:
        disturbance_interval_s = (
            dr.disturbance_interval_policy_steps * WOLF_CONFIG.control.policy_dt
        )
        terms["dr_disturbance"] = EventTermCfg(
            func=envs_mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(disturbance_interval_s, disturbance_interval_s),
            is_global_time=True,
            params={
                "force_range": dr.disturbance_force_range,
                "torque_range": (0.0, 0.0),
                "asset_cfg": base_body,
            },
        )

    return terms


def _configure_common_runtime(cfg: ManagerBasedRlEnvCfg, *, play: bool, rough: bool = False) -> None:
    """task 级通用字段：env 数、episode 长度、physics dt、sim capacity、viewer。"""
    cfg.scene.num_envs = (
        WOLF_CONFIG.env.play_num_envs if play else WOLF_CONFIG.env.train_num_envs
    )
    cfg.episode_length_s = WOLF_CONFIG.env.episode_length_s
    cfg.decimation = WOLF_CONFIG.control.decimation
    cfg.sim.mujoco.timestep = WOLF_CONFIG.control.physics_dt
    # MJWarp per-world capacity（与 Black 同源的候选值，见 wolf_config.SimulationParams）。
    # rough generator 模板 spawn 状态 contact 数高于 plane（>=178），用 rough_nconmax。
    cfg.sim.nconmax = WOLF_CONFIG.simulation.rough_nconmax if rough else WOLF_CONFIG.simulation.nconmax
    cfg.sim.njmax = WOLF_CONFIG.simulation.rough_njmax if rough else WOLF_CONFIG.simulation.njmax
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.8
    cfg.viewer.elevation = -10.0


def _configure_command_curriculum(
    cfg: ManagerBasedRlEnvCfg, stage: Literal["flat", "rough"]
) -> None:
    """性能驱动 forward-speed command curriculum（仅 train；stage = flat | rough）。

    Wolf 本地 CurriculumTerm 实现（curriculums.py），数值全部由 WOLF_CONFIG
    注入（含 max_abs_vx=4.0），不从 BLACK_CONFIG 读取；`robot` / `stage` 元数据
    用于 checkpoint restore 的 provenance / restore mode 判定（跨 robot 拒绝恢复；
    flat→rough 跨 stage 用 range 恢复）。
    """
    cc = WOLF_CONFIG.command.command_curriculum
    cc.validate()
    if not cc.enabled:
        return
    cfg.curriculum[CURRICULUM_TERM_NAME] = CurriculumTermCfg(
        func=WolfForwardSpeedCommandCurriculum,
        params={
            "command_name": WOLF_COMMAND_NAME,
            "reward_term_name": WOLF_TRACKING_VELOCITY_REWARD_TERM,
            "stage": stage,
            "robot": "wolf",
            "curriculum_params": cc,
        },
    )


def _configure_terrain_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough train 启用 terrain curriculum（native ``terrain_levels_vel``）。

    难度推进 / 回退公式由 MjLab native 实现（walked distance 与 commanded velocity
    对比）；command curriculum 由 ``_configure_command_curriculum()`` 在其上追加
    （不覆盖；flat 不注册本 term）。
    """
    cfg.curriculum[WOLF_TERRAIN_CURRICULUM_TERM] = CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
        params={"command_name": WOLF_COMMAND_NAME},
    )


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
    """play 模式：remove illegal_contact，curriculum 清空（nominal 干净观察）。

    play 强制关闭 DR（双保险：builder 路径已不注入，这里再清一次防御）。
    """
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.terminations.pop("illegal_contact", None)
    # 越界终止仅在 rough 注册；play 下按 Black rough 处理同样移除。
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    for name in [k for k in cfg.events if k.startswith("dr_")]:
        cfg.events.pop(name, None)


def _build_wolf_env_cfg(
    play: bool,
    rough: bool,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """Wolf flat / rough 共用装配路径（二者只差 terrain 语义等任务特化）。

    顺序：command / scene+sensors（rough 追加 terrain_scan）/ actions / events /
    rewards（rough 换 base_height 测量方式）→ flat 或 rough terrain →
    observations → terminations（rough 追加 out_of_terrain_bounds）→
    common runtime → curriculum（rough train：terrain_levels + command；
    flat train：command；play：两者都没有）→ play。

    play 模式强制无 DR（nominal dynamics）；train 的 DR 默认取 WOLF_CONFIG.dr
    （DomainRandomizationParams，默认全关），也可显式传入（供 minimal 等
    profile / 验证脚本使用）。
    """
    cfg = make_velocity_env_cfg()

    if play:
        dr = DomainRandomizationParams()
    else:
        dr = WOLF_CONFIG.dr if dr is None else dr
        dr.validate()

    if not rough:
        # 仅 flat（train + play）：编译模板 spawn 高度抬高，消除模板穿地接触
        # 对 MJWarp 容量的硬下限；rough 模板容量需求另行处理（§30.2）。
        _configure_template_spawn_height(cfg)

    _configure_command(cfg)
    _configure_scene_and_sensors(cfg, dr, rough)
    _configure_actions(cfg, dr)
    _configure_events(cfg, dr, rough)
    _configure_rewards(cfg, rough)
    if rough:
        _configure_rough_terrain(cfg)
    else:
        _configure_flat_terrain(cfg)
    _configure_observations(cfg, dr)
    _configure_terminations(cfg, rough)
    _configure_common_runtime(cfg, play=play, rough=rough)
    stage = "rough" if rough else "flat"
    if not play:
        if rough:
            _configure_terrain_curriculum(cfg)
        _configure_command_curriculum(cfg, stage)
    if play:
        _configure_play(cfg)

    return cfg


def wolf_flat_env_cfg(
    play: bool = False,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """Wolf flat-ground PPO task。

    actor 53 维单帧；critic 56 维；terrain 为 plane；DR 默认取 WOLF_CONFIG.dr
    （全关 = nominal dynamics）。
    """
    return _build_wolf_env_cfg(play=play, rough=False, dr=dr)


def wolf_rough_env_cfg(
    play: bool = False,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """Wolf rough-terrain PPO task（flat contract + rough terrain 语义）。

    actor 仍为 53 维单帧、critic 仍为 56 维（不含 height_scan，与 flat 完全同 shape，
    保证 flat→rough 续训直接复用 checkpoint 网络）；差异仅在：terrain generator
    （7 类 / 10 行）、base_height reward 改用 terrain_scan 中央 35 rays clearance、
    增加 out_of_terrain_bounds 截断、train 开启 terrain_levels_vel + command
    curriculum（stage=rough）。play 保留 generator 布局供策略回放，移除非法接触 /
    越界终止。

    DR 默认取 WOLF_CONFIG.dr（全关 = nominal dynamics）；rough 下摩擦 DR 的
    地面 geom 选择为 generator 全部 patch geom（见 _dr_event_terms）。
    """
    return _build_wolf_env_cfg(play=play, rough=True, dr=dr)


def wolf_flat_him_env_cfg(
    play: bool = False,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """``wolf-flat`` + HIM observation / history / terminal contract（wolf/him.py）。

    与 ``wolf_flat_env_cfg`` 的 reward / command / reset / action / termination /
    terrain / robot 完全相同，只增量应用 HIM 契约（wolf/him.py，group 名与
    terminal extras key 均为 HIM 算法接口 contract）：

    - actor group 打开 MjLab 原生 history（``[B, 6, 53]``，oldest → newest）；
    - 新增 ``estimator_velocity`` group（``[B, 3]``，scaled true base lin vel）；
    - 注册 terminal successor target recorder。
    """
    cfg = wolf_flat_env_cfg(play=play, dr=dr)
    configure_him_observations(cfg)
    configure_him_terminal_targets(cfg)
    return cfg


def wolf_rough_him_env_cfg(
    play: bool = False,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """``wolf-rough`` + HIM observation / history / terminal contract（同 flat HIM）。"""
    cfg = wolf_rough_env_cfg(play=play, dr=dr)
    configure_him_observations(cfg)
    configure_him_terminal_targets(cfg)
    return cfg
