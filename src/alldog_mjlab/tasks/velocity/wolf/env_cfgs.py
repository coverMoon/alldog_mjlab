"""Wolf flat velocity task 的 MjLab task assembly（结构对应 black/env_cfgs.py）。

term 顺序 contract、selector 绑定、sensor 装配都显式写在本文件；训练数值在
wolf_config.py，reward 数学复用 black/rewards.py（机器人无关），Wolf 专属的
sign / 16-D action 分组逻辑在本任务的 observations.py / rewards.py。

与 Black task 的关键结构差异：
- actor 角速度 / 投影重力来自 Wolf XML 原生 IMU sensor（robot/imu_*，§27.5）；
- action 为 8 个原生 action term 的交错拼接（每腿 3 pos + 1 wheel，共 16 维）；
- wheel 有独立 obs / reward 通道；
- 终止只有 time_out + base_link 非法接触（无倾角 / 无 stuck）；play 下移除非法接触；
- 无 DR 事件、无 terrain generator / curriculum（flat plane，nominal dynamics）。
"""

from dataclasses import replace
from typing import TYPE_CHECKING

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
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
from alldog_mjlab.tasks.velocity.black.curriculums import (
    CURRICULUM_TERM_NAME,
    ForwardSpeedCommandCurriculum,
)
from alldog_mjlab.tasks.velocity.black.him import (
    configure_him_observations,
    configure_him_terminal_targets,
)
from alldog_mjlab.tasks.velocity.black.rewards import (
    angular_velocity_xy_l2,
    base_height_l2_flat,
    track_angular_velocity_z,
    track_linear_velocity_xy,
    vertical_linear_velocity_l2,
)
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
from alldog_mjlab.tasks.velocity.wolf.rewards import (
    leg_action_rate_l2,
    wheel_action_rate_l2,
)
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
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams
) -> None:
    """robot entity 与非法接触 sensor 装配；移除 baseline 的 terrain / foot 传感器。

    Wolf 用不到 foot site / height scan：轮足结构没有足端摆动相概念，flat plane 的
    base height reward 直接读 world z，critic 也不需要 terrain height。
    DR 的 actuator delay 在 robot cfg 层注入（见 _wolf_robot_cfg）。
    """
    cfg.scene.entities = {"robot": _wolf_robot_cfg(dr)}
    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name not in ("foot_height_scan", "terrain_scan")
    )

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
    cfg: ManagerBasedRlEnvCfg, dr: DomainRandomizationParams
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
        **_dr_event_terms(dr),
    }
    # baseline 的 push / foot_friction / encoder_bias / base_com DR 事件全部不注册
    # （Wolf DR 事件统一在 _dr_event_terms 中显式构造）。


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """Wolf flat v1 reward baseline：显式 8 项（dict 顺序即 logging 顺序）。

    tracking / lin_vel_z / body_ang_vel / orientation / base_height 复用 Black
    已验证实现（HIMLoco 公式，机器人无关）；action rate 按显式 16-D contract 的
    leg / wheel 分组（wolf/rewards.py，见 WOLF_ACTION_TERM_ORDER 的块布局推导）。
    不加入 run_still / 固定步态 / 强制轮地接触等限制高速轮足混合的项。
    """
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
        # 与 HIMLoco `_reward_orientation` 严格一致，故直接用 native。
        "upright": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=WOLF_CONFIG.reward.scales.orientation,
        ),
        "base_height": RewardTermCfg(
            func=base_height_l2_flat,
            weight=WOLF_CONFIG.reward.scales.base_height,
            params={
                "target_height": WOLF_CONFIG.reward.base_height_target,
            },
        ),
        "leg_action_rate": RewardTermCfg(
            func=leg_action_rate_l2,
            weight=WOLF_CONFIG.reward.scales.leg_action_rate,
        ),
        "wheel_action_rate": RewardTermCfg(
            func=wheel_action_rate_l2,
            weight=WOLF_CONFIG.reward.scales.wheel_action_rate,
        ),
    }
    assert WOLF_TRACKING_VELOCITY_REWARD_TERM in cfg.rewards


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """flat task specialization：plane terrain、无 terrain generator / curriculum。"""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)


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


def _configure_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
    """终止 contract：time_out + illegal_contact（base_link 对 terrain）。

    baseline 的 fell_over（倾角终止）与 out_of_terrain_bounds 显式移除；stuck
    termination 本轮不注册。
    """
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": WOLF_ILLEGAL_CONTACT_SENSOR,
            "force_threshold": WOLF_CONFIG.termination.illegal_contact_force,
        },
    )


def _dr_event_terms(dr: DomainRandomizationParams) -> dict[str, EventTermCfg]:
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
    # 地面 geom（plane terrain 的碰撞平面；摩擦事件把它压 0，见 randomization 模块）。
    ground_plane_geoms = SceneEntityCfg("terrain", geom_names=("terrain",))
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
                "terrain_cfg": ground_plane_geoms,
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
                "terrain_cfg": ground_plane_geoms,
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


def _configure_common_runtime(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """task 级通用字段：env 数、episode 长度、physics dt、sim capacity、viewer。"""
    cfg.scene.num_envs = (
        WOLF_CONFIG.env.play_num_envs if play else WOLF_CONFIG.env.train_num_envs
    )
    cfg.episode_length_s = WOLF_CONFIG.env.episode_length_s
    cfg.decimation = WOLF_CONFIG.control.decimation
    cfg.sim.mujoco.timestep = WOLF_CONFIG.control.physics_dt
    # MJWarp per-world capacity（与 Black 同源的候选值，见 wolf_config.SimulationParams）。
    cfg.sim.nconmax = WOLF_CONFIG.simulation.nconmax
    cfg.sim.njmax = WOLF_CONFIG.simulation.njmax
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.8
    cfg.viewer.elevation = -10.0


def _configure_command_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
    """性能驱动 forward-speed command curriculum（仅 train；stage = flat）。

    与 Black 同一 CurriculumTerm 实现（curriculums.py），数值全部由 WOLF_CONFIG
    注入（含 max_abs_vx=4.0），不从 BLACK_CONFIG 读取；`robot` / `stage` 元数据
    用于 checkpoint restore 的 provenance 判定（跨 robot 拒绝恢复）。
    """
    cc = WOLF_CONFIG.command.command_curriculum
    cc.validate()
    cfg.curriculum[CURRICULUM_TERM_NAME] = CurriculumTermCfg(
        func=ForwardSpeedCommandCurriculum,
        params={
            "command_name": WOLF_COMMAND_NAME,
            "reward_term_name": WOLF_TRACKING_VELOCITY_REWARD_TERM,
            "stage": "flat",
            "robot": "wolf",
            "curriculum_params": cc,
        },
    )


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
    """play 模式：remove illegal_contact，curriculum 清空（nominal 干净观察）。

    play 强制关闭 DR（双保险：builder 路径已不注入，这里再清一次防御）。
    """
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.terminations.pop("illegal_contact", None)
    cfg.curriculum = {}
    for name in [k for k in cfg.events if k.startswith("dr_")]:
        cfg.events.pop(name, None)


def _build_wolf_env_cfg(
    play: bool,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """Wolf flat 任务的完整装配路径（train 含 command curriculum，play 干净）。

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

    _configure_command(cfg)
    _configure_scene_and_sensors(cfg, dr)
    _configure_actions(cfg, dr)
    _configure_events(cfg, dr)
    _configure_rewards(cfg)
    _configure_flat_terrain(cfg)
    _configure_observations(cfg, dr)
    _configure_terminations(cfg)
    _configure_common_runtime(cfg, play=play)
    if not play:
        _configure_command_curriculum(cfg)
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
    return _build_wolf_env_cfg(play=play, dr=dr)


def wolf_flat_him_env_cfg(
    play: bool = False,
    dr: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    """``wolf-flat`` + HIM observation / history / terminal contract（复用 black/him.py）。

    与 ``wolf_flat_env_cfg`` 的 reward / command / reset / action / termination /
    terrain / robot 完全相同，只增量应用 HIM 契约（black/him.py 为机器人无关实现，
    group 名与 terminal extras key 均为 HIM 算法接口 contract）：

    - actor group 打开 MjLab 原生 history（``[B, 6, 53]``，oldest → newest）；
    - 新增 ``estimator_velocity`` group（``[B, 3]``，scaled true base lin vel）；
    - 注册 terminal successor target recorder。
    """
    cfg = wolf_flat_env_cfg(play=play, dr=dr)
    configure_him_observations(cfg)
    configure_him_terminal_targets(cfg)
    return cfg
