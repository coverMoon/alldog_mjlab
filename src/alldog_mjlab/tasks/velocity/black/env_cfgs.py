"""Velocity task configurations for Black."""

from dataclasses import replace
import math

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
from alldog_mjlab.tasks.velocity.black.rewards import (
    track_angular_velocity_z,
    track_linear_velocity_xy,
)
from alldog_mjlab.tasks.velocity.black.terminations import StuckTermination

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

# 各分量的 raw noise 幅值。MjLab v1.6.0 pipeline 为
# compute → noise → clip → scale，故这里写加在 raw 值上的噪声：
# 进入 policy 的最终幅值 = raw noise × scale，不要写成已乘过 scale 的值。
BLACK_ACTOR_OBS_NOISE: dict[str, tuple[float, float]] = {
    "base_ang_vel": (-0.3, 0.3),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos": (-0.08, 0.08),
    "joint_vel": (-2.0, 2.0),
}

# 摔倒终止 contract：trunk 或任一 thigh 与 terrain 接触，且接触力 > 1.0 N。
# 不含 calf / foot / hip。history_length 取一个 control step 内的 physics substep 数
# （sim dt 0.005 × decimation 4 = 0.02 s），用于捕获 step 中途出现的碰撞。
BLACK_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"
BLACK_ILLEGAL_CONTACT_BODIES = (
    "trunk",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
)
BLACK_ILLEGAL_CONTACT_FORCE_THRESHOLD = 1.0
BLACK_ILLEGAL_CONTACT_HISTORY = 4

# Root reset contract：pose 不随机（x/y/z/roll/pitch/yaw 均为 0 offset，
# root 高度直接取 INIT_STATE.pos 的 0.45 m），root 六维速度独立均匀采样 [-0.5, 0.5]。
# key 名称固定为 MjLab v1.6.0 的 SE(3) 轴名（velocity 也用 x/y/z/roll/pitch/yaw）。
BLACK_ROOT_RESET_POSE_RANGE: dict[str, tuple[float, float]] = {}
BLACK_ROOT_RESET_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}

# Joint reset contract：以 default joint pose 为均值的对称 offset 采样，joint 速度不随机。
# 三个分组的 offset 范围均完全落在 MjLab soft joint limits 内（不依赖 clamp）：
#   hip   保持 default（offset 0）；
#   thigh 保持 ±0.4007 = 0.8014 × 0.5（default magnitude 的一半）；
#   calf  取 ±0.5945，使左右 calf 的 support（default ± 0.5945）都不超出 soft limits。
BLACK_JOINT_RESET_POSITION_RANGE: dict[str, tuple[float, float]] = {
    "hip": (0.0, 0.0),
    "thigh": (-0.4007, 0.4007),
    "calf": (-0.5945, 0.5945),
}
BLACK_JOINT_RESET_VELOCITY_RANGE: tuple[float, float] = (0.0, 0.0)

# Stuck termination contract：planar command 有效（norm > 0.2 m/s）且沿该指令方向的
# progress speed 持续低于 0.05 m/s 时终止，grace 期内不计时。计时以 control step
# 时长累计，阈值单位为秒。
BLACK_STUCK_TIMEOUT_S = 4.0
BLACK_STUCK_VELOCITY_THRESHOLD = 0.05
BLACK_STUCK_COMMAND_THRESHOLD = 0.2
BLACK_STUCK_GRACE_S = 1.0

# Tracking reward contract：指数形式的速度跟踪，denominator 直接使用 legacy 的
# tracking_sigma（不是 sigma²）。linear 只含 vx/vy 误差，angular 只含 yaw 误差。
BLACK_TRACKING_SIGMA = 0.25
BLACK_TRACKING_LINEAR_WEIGHT = 2.0
BLACK_TRACKING_ANGULAR_WEIGHT = 1.5


def black_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground velocity task for Black."""

    cfg = make_velocity_env_cfg()

    # Black flat velocity command contract：heading command 关闭、
    # resampling 固定 10.0 s、vx/vy ∈ [-1, 1] m/s、wz ∈ [-pi, pi] rad/s。
    # v1.6.0 要求：heading_command=False 时 ranges.heading 必须为 None，否则构建环境时报错。
    twist_command = cfg.commands["twist"]
    assert isinstance(twist_command, UniformVelocityCommandCfg)
    twist_command.resampling_time_range = (10.0, 10.0)
    twist_command.heading_command = False
    twist_command.ranges.heading = None
    twist_command.ranges.lin_vel_x = (-1.0, 1.0)
    twist_command.ranges.lin_vel_y = (-1.0, 1.0)
    twist_command.ranges.ang_vel_z = (-math.pi, math.pi)

    cfg.scene.entities = {
        "robot": get_black_robot_cfg(),
    }

    foot_names = BLACK_FOOT_NAMES
    foot_geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

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
        history_length=BLACK_ILLEGAL_CONTACT_HISTORY,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (illegal_ground_contact,)

    cfg.actions.pop("joint_pos")
    # Black policy action contract：四个单腿 JointPositionAction term，
    # flat policy action 按 BLACK_ACTION_TERM_ORDER（FL → FR → RL → RR）拼接，
    # 每 term 精确绑定对应腿的 hip/thigh/calf 三个关节（每腿 3 维，共 12 维）。
    # 不使用单个全机器人 term：joint transmission 的 target 顺序在 MjLab v1.6.0
    # 中恒为 Entity natural order（FL → FR → RR → RL），无法表达 policy order。
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

    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        geom_names=foot_geom_names,
    )

    # Root reset contract：pose 固定为 default initial state（不随机 x/y/z/roll/pitch/yaw），
    # root 六维速度独立均匀采样 [-0.5, 0.5]。env origin 与 default root state 的叠加
    # 由 mdp.reset_root_state_uniform 内部处理。
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = dict(BLACK_ROOT_RESET_POSE_RANGE)
    reset_base.params["velocity_range"] = dict(BLACK_ROOT_RESET_VELOCITY_RANGE)

    # Joint reset contract：把单一的全机器人 joint reset 拆为 hip / thigh / calf
    # 三个选择器互不重叠的 native reset event（event 执行顺序不影响结果）。
    base_joint_reset = cfg.events.pop("reset_robot_joints")
    assert base_joint_reset.func is reset_joints_by_offset
    for joint_group, position_range in BLACK_JOINT_RESET_POSITION_RANGE.items():
        cfg.events[f"reset_{joint_group}_joints"] = replace(
            base_joint_reset,
            params={
                "position_range": position_range,
                "velocity_range": BLACK_JOINT_RESET_VELOCITY_RANGE,
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

    # Tracking reward contract：替换 native track_* 的 func/weight/params
    # （native 会把 v_z² / ω_xy² 并入同一个 exponential，与 Black contract 不等价）。
    # term name 保持 MjLab 原生名称，dt 缩放由 RewardManager 统一处理。
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=track_linear_velocity_xy,
        weight=BLACK_TRACKING_LINEAR_WEIGHT,
        params={
            "command_name": "twist",
            "sigma": BLACK_TRACKING_SIGMA,
        },
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
        func=track_angular_velocity_z,
        weight=BLACK_TRACKING_ANGULAR_WEIGHT,
        params={
            "command_name": "twist",
            "sigma": BLACK_TRACKING_SIGMA,
        },
    )

    cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=("trunk",),
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=("trunk",),
    )

    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"] = SceneEntityCfg(
            "robot",
            site_names=foot_names,
        )

    cfg.rewards["pose"].params["std_standing"] = {
        r".*_hip_joint": 0.05,
        r".*_thigh_joint": 0.05,
        r".*_calf_joint": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*_hip_joint": 0.3,
        r".*_thigh_joint": 0.3,
        r".*_calf_joint": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*_hip_joint": 0.3,
        r".*_thigh_joint": 0.3,
        r".*_calf_joint": 0.6,
    }

    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    # Black PPO actor 45 维单步 observation contract：内容与 layout 显式重建。
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
    for term_name in BLACK_ACTOR_OBS_TERM_ORDER:
        noise_range = BLACK_ACTOR_OBS_NOISE.get(term_name)
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
    # 摔倒终止：用 trunk / thigh 触地替代 orientation-based 的 fell_over，
    # 与 time_out（20.0 s）共同构成训练终止 contract。
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": BLACK_ILLEGAL_CONTACT_SENSOR,
            "force_threshold": BLACK_ILLEGAL_CONTACT_FORCE_THRESHOLD,
        },
    )
    # Stuck termination：沿 planar command 方向无 progress 的连续时长超过阈值。
    # stateful，因此用 class-based term 而非 mdp 函数。
    cfg.terminations["stuck"] = TerminationTermCfg(
        func=StuckTermination,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": BLACK_STUCK_COMMAND_THRESHOLD,
            "velocity_threshold": BLACK_STUCK_VELOCITY_THRESHOLD,
            "grace_s": BLACK_STUCK_GRACE_S,
            "timeout_s": BLACK_STUCK_TIMEOUT_S,
        },
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg
