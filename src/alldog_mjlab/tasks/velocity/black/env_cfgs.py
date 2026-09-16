"""Black flat-ground velocity task 的 MjLab task assembly。

本文件回答的是「task 是怎么组装的」：term 顺序 contract、selector 绑定、
sensor 装配、与 MjLab native 配置的差异。
训练者要调的具体数值在 params.py，reward 数学在 rewards.py，
stateful termination 实现在 terminations.py。
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from alldog_mjlab.robots.black import BLACK_ACTION_SCALE, get_black_robot_cfg
from alldog_mjlab.robots.black.black_constants import (
    BLACK_FOOT_NAMES,
    BLACK_JOINT_NAMES,
)
from alldog_mjlab.tasks.velocity.black import params
from alldog_mjlab.tasks.velocity.black.rewards import (
    angular_velocity_xy_l2,
    base_height_l2_flat,
    dof_acc_l2,
    joint_power_l1,
    track_angular_velocity_z,
    track_linear_velocity_xy,
    vertical_linear_velocity_l2,
)
from alldog_mjlab.tasks.velocity.black.terminations import StuckTermination

# ---------------------------------------------------------------------------
# Interface contracts（不属于训练调参，勿当作超参数阅读）
# ---------------------------------------------------------------------------

# Policy action term 顺序 contract：ActionManager 按 dict 插入顺序切分
# flat policy action，因此必须由此显式顺序驱动构造，不能依赖字面 dict 写法。
# 腿顺序为 FL → FR → RL → RR，每腿 hip → thigh → calf（共 12 维）。
BLACK_ACTION_TERM_ORDER = (
    "joint_pos_fl",
    "joint_pos_fr",
    "joint_pos_rl",
    "joint_pos_rr",
)

# Black PPO actor 单步 observation layout contract（共 45 维）：
#   command → base_ang_vel → projected_gravity → joint_pos → joint_vel → action
# 不含 base_lin_vel / height_scan，也不含 history（单帧）。
BLACK_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Black actor observation 数值 contract：command 三分量缩放分别为
# vx 2.0 / vy 2.0 / wz 0.25，其余为各分量的固定 scale。
BLACK_ACTOR_OBS_SCALE: dict[str, float | tuple[float, ...]] = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}

# 摔倒终止 sensor 身份与 body selector：trunk + 四 thigh 对 terrain。
# 不含 calf / foot / hip。
BLACK_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"
BLACK_ILLEGAL_CONTACT_BODIES = (
    "trunk",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
)

# Command term 名称属于 task wiring：reward / termination / observation 都按名字取它。
BLACK_COMMAND_NAME = "twist"


def _configure_command(cfg: ManagerBasedRlEnvCfg) -> None:
    """Black flat velocity command contract。

    heading command 关闭（v1.6.0 要求 heading_command=False 时 ranges.heading 必须为
    None，否则构建环境时报错），数值范围见 params.py。
    """
    twist_command = cfg.commands[BLACK_COMMAND_NAME]
    assert isinstance(twist_command, UniformVelocityCommandCfg)
    twist_command.resampling_time_range = params.command.resampling_time
    twist_command.heading_command = False
    twist_command.ranges.heading = None
    twist_command.ranges.lin_vel_x = params.command.lin_vel_x
    twist_command.ranges.lin_vel_y = params.command.lin_vel_y
    twist_command.ranges.ang_vel_z = params.command.ang_vel_z


def _configure_scene_and_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
    """Robot entity 与三个接触 / 高度 sensor 的装配。"""
    cfg.scene.entities = {
        "robot": get_black_robot_cfg(),
    }

    foot_names = BLACK_FOOT_NAMES
    foot_geom_names = _foot_geom_names()

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)

            sensor.frame = tuple(
                ObjRef(
                    type="site",
                    name=name,
                    entity="robot",
                )
                for name in foot_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=foot_geom_names,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_contact,)

    # 摔倒终止专用 sensor：trunk / thigh 对 terrain 的接触力历史。
    # 不复用 feet_ground_contact（那是正常行走接触）。
    illegal_ground_contact = ContactSensorCfg(
        name=BLACK_ILLEGAL_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="body",
            pattern=BLACK_ILLEGAL_CONTACT_BODIES,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=params.termination.illegal_contact_history,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (illegal_ground_contact,)


def _configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
    """Black policy action contract：四个单腿 JointPositionAction term。

    flat policy action 按 BLACK_ACTION_TERM_ORDER（FL → FR → RL → RR）拼接，
    每 term 精确绑定对应腿的 hip/thigh/calf 三个关节（每腿 3 维，共 12 维）。
    不使用单个全机器人 term：joint transmission 的 target 顺序在 MjLab v1.6.0
    中恒为 Entity natural order（FL → FR → RR → RL），无法表达 policy order。
    """
    cfg.actions.pop("joint_pos")

    leg_joint_names = {
        leg: tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf"))
        for leg in BLACK_FOOT_NAMES
    }
    action_terms: dict[str, JointPositionActionCfg] = {
        f"joint_pos_{leg.lower()}": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=leg_joint_names[leg],
            scale=BLACK_ACTION_SCALE,
            use_default_offset=True,
        )
        for leg in BLACK_FOOT_NAMES
    }
    assert tuple(action_terms) == BLACK_ACTION_TERM_ORDER
    cfg.actions = action_terms


def _configure_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Reset / domain 相关 event 的 selector 与数值绑定（不新增 DR 行为）。"""
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        geom_names=_foot_geom_names(),
    )

    # Root reset contract：pose 固定为 default initial state，root 六维速度独立均匀
    # 采样。env origin 与 default root state 的叠加由 mdp.reset_root_state_uniform
    # 内部处理。
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = dict(params.reset.root_pose)
    reset_base.params["velocity_range"] = dict(params.reset.root_velocity)

    # Joint reset contract：把单一的全机器人 joint reset 拆为 hip / thigh / calf
    # 三个选择器互不重叠的 native reset event（event 执行顺序不影响结果）。
    base_joint_reset = cfg.events.pop("reset_robot_joints")
    assert base_joint_reset.func is reset_joints_by_offset
    for joint_group, position_range in params.reset.joint_position.items():
        cfg.events[f"reset_{joint_group}_joints"] = replace(
            base_joint_reset,
            params={
                "position_range": position_range,
                "velocity_range": params.reset.joint_velocity,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=tuple(
                        f"{leg}_{joint_group}_joint" for leg in BLACK_FOOT_NAMES
                    ),
                ),
            },
        )
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=("trunk",),
    )


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """Black flat v1 reward baseline：显式构造最终 10 项。

    reward baseline 已正式冻结，因此不再「继承 MjLab baseline 再零散 patch」：
    这里直接给出完整 dict，dict 顺序即 logging / 调参表的稳定顺序。

    结构来源：
      - tracking / lin_vel_z / body_ang_vel / upright / base_height /
        dof_acc / joint_power / action_rate / smoothness 的公式取自
        InternRobotics/HIMLoco 官方 Go1 baseline；
      - base height target 等机器人相关数值取自 Black。
    native 严格等价的公式直接用 mjlab 函数（flat_orientation_l2 / action_rate_l2 /
    action_acc_l2），其余在 black/rewards.py 实现。

    dt 缩放由 RewardManager 统一处理，reward 函数只返回 raw magnitude。
    """
    cfg.rewards = {
        # Command tracking（HIMLoco：exp(-error / sigma)）。
        "track_linear_velocity": RewardTermCfg(
            func=track_linear_velocity_xy,
            weight=params.reward.tracking_linear,
            params={
                "command_name": BLACK_COMMAND_NAME,
                "sigma": params.reward.tracking_sigma,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=track_angular_velocity_z,
            weight=params.reward.tracking_angular,
            params={
                "command_name": BLACK_COMMAND_NAME,
                "sigma": params.reward.tracking_sigma,
            },
        ),
        # Base stability。
        "lin_vel_z": RewardTermCfg(
            func=vertical_linear_velocity_l2,
            weight=params.reward.lin_vel_z,
        ),
        "body_ang_vel": RewardTermCfg(
            func=angular_velocity_xy_l2,
            weight=params.reward.ang_vel_xy,
        ),
        # 与 HIMLoco `_reward_orientation` 严格一致，故直接用 native。
        "upright": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=params.reward.orientation,
        ),
        "base_height": RewardTermCfg(
            func=base_height_l2_flat,
            weight=params.reward.base_height,
            params={
                "target_height": params.reward.base_height_target,
            },
        ),
        # Regularization。
        "dof_acc": RewardTermCfg(
            func=dof_acc_l2,
            weight=params.reward.dof_acc,
        ),
        "joint_power": RewardTermCfg(
            func=joint_power_l1,
            weight=params.reward.joint_power,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=params.reward.action_rate,
        ),
        # HIMLoco `smoothness` 的二阶动作差分与 native action_acc_l2 严格一致。
        "smoothness": RewardTermCfg(
            func=mdp.action_acc_l2,
            weight=params.reward.smoothness,
        ),
    }


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """flat task specialization：plane terrain、无 terrain generator / scan / curriculum。"""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    cfg.curriculum.pop("terrain_levels", None)


def _configure_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """Black PPO actor 45 维单步 observation contract：内容、顺序、scale 与 noise。"""
    actor_terms = cfg.observations["actor"].terms
    # 移除 MjLab velocity 默认中不属于 Black actor contract 的项。
    actor_terms.pop("height_scan", None)
    actor_terms.pop("base_lin_vel")
    # joint_pos / joint_vel 必须使用 policy joint order（FL → FR → RL → RR）。
    # 每个 term 用独立的 SceneEntityCfg 实例；并用 replace 生成独立 term cfg，
    # 避免改动与 critic 共享的 ObservationTermCfg 对象（其 func/noise/biased 等均保留）。
    for term_name in ("joint_pos", "joint_vel"):
        term_cfg = actor_terms[term_name]
        actor_terms[term_name] = replace(
            term_cfg,
            params={
                **term_cfg.params,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=BLACK_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )
    # 由显式 contract 重建 term 顺序，不依赖基础配置的原始插入顺序。
    cfg.observations["actor"].terms = {
        name: actor_terms[name] for name in BLACK_ACTOR_OBS_TERM_ORDER
    }
    # Black actor 数值 contract：固定 scale 与 raw noise 幅值。
    # 同样用 replace 生成独立 term cfg，避免通过共享对象改动 critic term。
    # term → noise 的映射属于 observation wiring，因此留在这里而不是参数对象内部。
    actor_noise = {
        "base_ang_vel": params.observation_noise.base_ang_vel,
        "projected_gravity": params.observation_noise.projected_gravity,
        "joint_pos": params.observation_noise.joint_pos,
        "joint_vel": params.observation_noise.joint_vel,
    }
    for term_name in BLACK_ACTOR_OBS_TERM_ORDER:
        noise_range = actor_noise.get(term_name)
        cfg.observations["actor"].terms[term_name] = replace(
            actor_terms[term_name],
            scale=BLACK_ACTOR_OBS_SCALE[term_name],
            noise=(
                UniformNoiseCfg(n_min=noise_range[0], n_max=noise_range[1])
                if noise_range is not None
                else None
            ),
        )
    cfg.observations["critic"].terms.pop("height_scan", None)


def _configure_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
    """终止 contract：time_out + illegal_contact + stuck。"""
    # 摔倒终止用 trunk / thigh 触地替代 orientation-based 的 fell_over。
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": BLACK_ILLEGAL_CONTACT_SENSOR,
            "force_threshold": params.termination.illegal_contact_force,
        },
    )
    # Stuck termination：沿 planar command 方向无 progress 的连续时长超过阈值。
    # stateful，因此用 class-based term 而非 mdp 函数。
    cfg.terminations["stuck"] = TerminationTermCfg(
        func=StuckTermination,
        params={
            "command_name": BLACK_COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": params.termination.stuck_command_threshold,
            "velocity_threshold": params.termination.stuck_velocity_threshold,
            "grace_s": params.termination.stuck_grace,
            "timeout_s": params.termination.stuck_timeout,
        },
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)


def _configure_common_runtime(cfg: ManagerBasedRlEnvCfg) -> None:
    """不属于上面各分组的少量 task 级字段。"""
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
    """play 模式的既有差异：只放开 episode 长度、关闭 corruption / push / curriculum。"""
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}


def _foot_geom_names() -> tuple[str, ...]:
    """四足 collision geom 名（由 policy 腿顺序 contract 驱动）。"""
    return tuple(f"{name}_foot_collision" for name in BLACK_FOOT_NAMES)


def black_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground velocity task for Black."""

    cfg = make_velocity_env_cfg()

    _configure_command(cfg)
    _configure_scene_and_sensors(cfg)
    _configure_actions(cfg)
    _configure_events(cfg)
    _configure_rewards(cfg)
    _configure_flat_terrain(cfg)
    _configure_observations(cfg)
    _configure_terminations(cfg)
    _configure_common_runtime(cfg)

    if play:
        _configure_play(cfg)

    return cfg
