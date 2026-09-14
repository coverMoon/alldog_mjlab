"""Velocity task configurations for Black."""

from dataclasses import replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from nwwolf_mjlab.robots.black import BLACK_ACTION_SCALE, get_black_robot_cfg
from nwwolf_mjlab.robots.black.black_constants import (
    BLACK_FOOT_NAMES,
    BLACK_JOINT_NAMES,
)

# Policy action term 顺序 contract：ActionManager 按 dict 插入顺序切分
# flat policy action，因此必须由此显式顺序驱动构造，不能依赖字面 dict 写法。
# 对齐旧 super-dog Black 运行时 dof order：FL → FR → RL → RR，每腿 3 维。
BLACK_ACTION_TERM_ORDER = (
    "joint_pos_fl",
    "joint_pos_fr",
    "joint_pos_rl",
    "joint_pos_rr",
)

# Black PPO actor 单步 observation layout contract（对齐旧 super-dog Black
# num_one_step_observations = 45）：
#   command → base_ang_vel → projected_gravity → joint_pos → joint_vel → action
# 不含 base_lin_vel / height_scan，也不含 history（仍为单帧）。
BLACK_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Black actor observation 数值 contract（对齐旧 super-dog Black 的固定
# obs_scales / noise_scales）：command 用 (lin_vel, lin_vel, ang_vel) 缩放，
# 其余为各分量的 obs_scales。
BLACK_ACTOR_OBS_SCALE: dict[str, float | tuple[float, ...]] = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}

# 旧 legged_gym 的 raw noise_scales（noise_level = 1.0）。
# MjLab v1.6.0 pipeline 为 compute → noise → clip → scale，因此噪声进入
# policy 的幅值 = raw noise × scale，与旧 (value + raw_noise) × scale 一致；
# 不得写成已乘过 scale 的最终幅值。
BLACK_ACTOR_OBS_NOISE: dict[str, tuple[float, float]] = {
    "base_ang_vel": (-0.3, 0.3),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos": (-0.08, 0.08),
    "joint_vel": (-2.0, 2.0),
}


def black_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground velocity task for Black."""

    cfg = make_velocity_env_cfg()

    # Black flat 初始 velocity command contract（对齐旧 super-dog BlackCfg.commands）：
    # heading command 关闭、resampling 固定 10.0 s、vx/vy ∈ [-1, 1] m/s、wz ∈ [-pi, pi] rad/s。
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
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=("trunk",),
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
    # 移除 MjLab velocity 默认中不属于旧 Black actor 的项。
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
    # Black actor 数值 contract：固定 scale + 旧 raw noise_scales。
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
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg
