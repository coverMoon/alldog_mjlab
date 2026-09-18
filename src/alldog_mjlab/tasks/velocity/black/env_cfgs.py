"""Black flat-ground velocity task 的 MjLab task assembly。

本文件回答的是「task 是怎么组装的」：term 顺序 contract、selector 绑定、
sensor 装配、与 MjLab native 配置的差异。
训练者要调的具体数值在 params.py，reward 数学在 rewards.py，
stateful termination 实现在 terminations.py。
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.envs.mdp import dr
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
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
    base_height_l2_terrain,
    dof_acc_l2,
    joint_power_l1,
    track_angular_velocity_z,
    track_linear_velocity_xy,
    vertical_linear_velocity_l2,
)
from alldog_mjlab.tasks.velocity.black.terminations import StuckTermination
from alldog_mjlab.tasks.velocity.black.terrain import black_rough_terrain_generator_cfg

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

# Black critic（privileged）observation 顺序 contract。flat 为 MjLab velocity baseline
# 的 privileged 派生（72 维）；rough 只在末尾追加 terrain height scan（+187 = 259 维）。
# 显式重建顺序，不依赖 native `critic_terms = {**actor_terms, ...}` 的 dict 顺序。
BLACK_FLAT_CRITIC_TERM_ORDER = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
)
BLACK_ROUGH_CRITIC_TERM_ORDER = BLACK_FLAT_CRITIC_TERM_ORDER + ("height_scan",)

# Terrain height scan sensor 身份 contract。sensor 对象复用 MjLab v1.6 native
# `terrain_scan`（GridPatternCfg(size=(1.6, 1.0), resolution=0.1) = 17 x 11 = 187 rays，
# max_distance 5.0 m，ray_alignment="yaw"），这里只固定 frame body 与数值语义。
BLACK_TERRAIN_SCAN_SENSOR = "terrain_scan"
BLACK_TERRAIN_SCAN_BODY = "trunk"
BLACK_TERRAIN_SCAN_MAX_DISTANCE = 5.0


def _configure_command(cfg: ManagerBasedRlEnvCfg) -> None:
    """Black flat v1 command contract：固定范围 + MjLab native sampler。

    - 范围与重采样间隔来自 params.command，训练全程固定（不迁移任何 command
      curriculum，`command_vel` 在本函数里移除）；
    - heading command 关闭（v1.6.0 要求 heading_command=False 时 ranges.heading 必须
      为 None，否则构建环境时报错）；`rel_heading_envs` 在 heading 关闭时不生效，
      显式写 0 以免被误读为启用；
    - standing / forward-only / world-frame 比例为 native sampler 契约，显式冻结。
    """
    twist_command = cfg.commands[BLACK_COMMAND_NAME]
    assert isinstance(twist_command, UniformVelocityCommandCfg)
    twist_command.resampling_time_range = params.command.resampling_time
    twist_command.ranges.lin_vel_x = params.command.lin_vel_x
    twist_command.ranges.lin_vel_y = params.command.lin_vel_y
    twist_command.ranges.ang_vel_z = params.command.ang_vel_z
    twist_command.heading_command = False
    twist_command.rel_heading_envs = 0.0
    twist_command.ranges.heading = None
    twist_command.rel_standing_envs = params.command.standing_fraction
    twist_command.rel_forward_envs = params.command.forward_fraction
    twist_command.rel_world_envs = params.command.world_fraction
    twist_command.init_velocity_prob = params.command.init_velocity_prob
    # MjLab velocity baseline 自带 staged velocity curriculum；Black flat v1 不迁移
    # 任何 command curriculum，范围从训练开始到结束保持不变。
    cfg.curriculum.pop("command_vel", None)


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
    """Reset 与 domain randomization 的 event contract。

    dict 顺序即下表，reset 在前、DR 在后：

        reset_base / reset_hip_joints / reset_thigh_joints / reset_calf_joints
        foot_friction / payload_mass / base_com / pd_gains / encoder_bias / push_robot

    数值全部来自 params（`_configure_play()` 会移除整组 DR）。
    """
    dr_params = params.domain_randomization
    base_events = dict(cfg.events)
    trunk_cfg = SceneEntityCfg("robot", body_names=("trunk",))

    # ---- reset contract ----
    # pose 固定为 default initial state，root 六维速度独立均匀采样；env origin 与
    # default root state 的叠加由 mdp.reset_root_state_uniform 内部处理。
    reset_base = replace(
        base_events["reset_base"],
        params={
            "pose_range": dict(params.reset.root_pose),
            "velocity_range": dict(params.reset.root_velocity),
        },
    )
    # 把单一的全机器人 joint reset 拆为 hip / thigh / calf 三个选择器互不重叠的
    # native reset event（event 执行顺序不影响结果）。
    base_joint_reset = base_events["reset_robot_joints"]
    assert base_joint_reset.func is reset_joints_by_offset
    joint_resets = {
        f"reset_{joint_group}_joints": replace(
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
        for joint_group, position_range in params.reset.joint_position.items()
    }

    cfg.events = {
        "reset_base": reset_base,
        **joint_resets,
        # ---- domain randomization contract ----
        # 四只脚的切向摩擦：同一 env 内共享一个采样，不同 env 独立。
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=_foot_geom_names()
                ),
                "operation": "abs",
                "ranges": dr_params.friction,
                "shared_random": True,
            },
        ),
        # 躯干 payload：在 nominal trunk mass 上做 add。与 legacy 一致，只加质量
        # 不改惯量（mjlab dr.body_mass 会就此给出 UserWarning，是本 baseline 的预期语义）。
        "payload_mass": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "operation": "add",
                "ranges": dr_params.payload_mass,
            },
        ),
        # 躯干 COM offset（相对 nominal，不设 absolute COM）。
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "operation": "add",
                "ranges": dict(dr_params.com_offset),
            },
        ),
        # PD 增益缩放：12 个 IdealPd actuator，每个 episode reset 重新采样。
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": dr_params.kp_scale,
                "kd_range": dr_params.kd_scale,
                "operation": "scale",
            },
        ),
        # 固定 encoder calibration bias（startup 采样一次，episode reset 不重采）。
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": dr_params.encoder_bias,
            },
        ),
        # 周期性 xy 速度增量扰动（native push 为 add increment，非直接赋值）。
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=dr_params.push_interval,
            params={"velocity_range": dict(dr_params.push_velocity)},
        ),
    }


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


def _configure_rough_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough 只把 ``base_height`` 换成 local terrain-relative 语义。

    只替换 func / params：reward key 仍为 ``base_height``、weight 仍为
    ``params.reward.base_height``、顺序仍为第 6 项，因此 rough 与 flat 的 reward 表
    除该项测量方式（world z vs local terrain clearance）外完全一致。
    其余 9 项 func / weight / params 均不变。
    """
    term = cfg.rewards["base_height"]
    assert term.func is base_height_l2_flat
    cfg.rewards["base_height"] = replace(
        term,
        func=base_height_l2_terrain,
        params={
            "target_height": params.reward.base_height_target,
            "sensor_name": BLACK_TERRAIN_SCAN_SENSOR,
        },
    )


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """flat task specialization：plane terrain、无 terrain generator / scan / curriculum。

    `terrain_scan` sensor 在这里从 scene 移除（critic 侧的 height_scan term 由
    `_configure_observations()` 移除），因此 flat 不承担 raycast 开销。
    """
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != BLACK_TERRAIN_SCAN_SENSOR
    )
    cfg.curriculum.pop("terrain_levels", None)


def _configure_rough_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough task specialization：curriculum terrain generator（一个 terrain 一列）。

    覆盖 `_configure_flat_terrain()` 的 plane / 无 generator；generator 内部
    `curriculum=True`，因此列数 = terrain 类型数（5），`proportion` 是 env 分配权重。

    `terrain_scan` 复用 MjLab v1.6 native sensor（geometry / ray_alignment /
    max_distance 全部保持 native），只把 frame 绑到 trunk（与 native rough task 相同），
    供 `_configure_rough_privileged_observation()` 的 critic height scan 使用；
    flat specialization 会移除这个 sensor，因此两者不会同时存在。
    """
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = black_rough_terrain_generator_cfg()
    cfg.scene.terrain.max_init_terrain_level = params.terrain.max_init_terrain_level

    terrain_scans = [
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name == BLACK_TERRAIN_SCAN_SENSOR
    ]
    assert len(terrain_scans) == 1, terrain_scans
    terrain_scan = terrain_scans[0]
    assert isinstance(terrain_scan, RayCastSensorCfg)
    assert isinstance(terrain_scan.frame, ObjRef)
    assert terrain_scan.max_distance == BLACK_TERRAIN_SCAN_MAX_DISTANCE
    terrain_scan.frame.name = BLACK_TERRAIN_SCAN_BODY


def _configure_rough_privileged_observation(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough critic 在 flat 72 维之后追加 187 维 terrain height scan。

    数值语义为 native `envs_mdp.height_scan()`：raw = sensor frame z - terrain hit z
    （offset 0），scale = 1 / max_distance = 0.2，无 noise / 无 clip；不做 legacy 的
    `clip(root_z - 0.5 - terrain_z, -1, 1) * 5` 与 height noise（intentional
    difference，见 MIGRATION.md）。actor 不含 height_scan（仍 45 维）。
    """
    critic_terms = cfg.observations["critic"].terms
    assert tuple(critic_terms) == BLACK_FLAT_CRITIC_TERM_ORDER, tuple(critic_terms)
    critic_terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": BLACK_TERRAIN_SCAN_SENSOR},
        scale=1.0 / BLACK_TERRAIN_SCAN_MAX_DISTANCE,
    )
    assert tuple(critic_terms) == BLACK_ROUGH_CRITIC_TERM_ORDER, tuple(critic_terms)


def _configure_terrain_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough v1 只启用 terrain curriculum（native `terrain_levels_vel`）。

    难度推进 / 回退公式由 MjLab native 实现（walked distance 与 command x
    max_episode_length_s x 0.5 对比），与 legacy `_update_terrain_curriculum()` 一致；
    command curriculum 仍关闭（见 `_configure_command()`）。
    """
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": BLACK_COMMAND_NAME},
        ),
    }


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
    # critic（privileged）observation：flat v1 不含 height_scan，且 term 顺序由常量
    # 显式重建，不依赖 native `critic_terms = {**actor_terms, ...}` 的 dict 顺序。
    # rough 的 height_scan 由 `_configure_rough_privileged_observation()` 追加。
    critic_terms = cfg.observations["critic"].terms
    critic_terms.pop("height_scan", None)
    cfg.observations["critic"].terms = {
        name: critic_terms[name] for name in BLACK_FLAT_CRITIC_TERM_ORDER
    }


def _configure_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
    """终止 contract：time_out + illegal_contact + stuck（flat / rough 共用）。

    ``out_of_terrain_bounds`` 在这里移除（MjLab velocity baseline 自带它）：flat 不注册
    该 term，rough 由 `_configure_rough_terminations()` 在末尾追加。
    """
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


def _configure_rough_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
    """rough 在末尾追加 native ``out_of_terrain_bounds`` safety truncation。

    OOB 是有限生成 terrain 造成的 artificial truncation，不是机器人自身的 physical
    failure，因此沿用 MjLab v1.6 native baseline 的 ``time_out=True``（PPO 会对该
    env 做 value bootstrap，而非当作 terminal failure）；``margin`` 用 native
    default 0.3 m，不显式传入（本轮没有 Black-specific tuning evidence）。

    flat 不注册该 term；rough play 由 `_configure_play()` 移除。
    """
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=mdp.out_of_terrain_bounds,
        time_out=True,
    )


def _configure_common_runtime(cfg: ManagerBasedRlEnvCfg) -> None:
    """不属于上面各分组的少量 task 级字段。"""
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
    """play 模式：nominal physics（移除整组 DR），只保留 reset events。

    ``out_of_terrain_bounds`` 也在 play 下移除（与 MjLab native rough play 一致）；
    其余 termination（illegal_contact / stuck / time_out）不变。
    """
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.terminations.pop("out_of_terrain_bounds", None)
    for event_name in (
        "foot_friction",
        "payload_mass",
        "base_com",
        "pd_gains",
        "encoder_bias",
        "push_robot",
    ):
        cfg.events.pop(event_name, None)
    cfg.curriculum = {}


def _foot_geom_names() -> tuple[str, ...]:
    """四足 collision geom 名（由 policy 腿顺序 contract 驱动）。"""
    return tuple(f"{name}_foot_collision" for name in BLACK_FOOT_NAMES)


def _build_black_env_cfg(play: bool, rough: bool) -> ManagerBasedRlEnvCfg:
    """flat / rough 共用装配路径：二者只差 terrain specialization 与 critic height scan。

    顺序：command / scene+sensors / actions / events / rewards（rough 再覆盖 base_height）
    → flat 或 rough terrain → observations（rough 追加 critic height scan）
    → terminations（rough 追加 out_of_terrain_bounds）/ common runtime
    → terrain curriculum（仅 rough 训练）→ play。

    rough 专门化保留 native `terrain_scan` sensor（flat 专门化会移除它），因此不再
    需要从 flat 配置事后恢复 sensor，也不复制 MjLab baseline 的 sensor / observation
    dict。action / observation / reward / reset / termination / DR / command 的装配
    与已冻结 contract 完全一致。
    """
    cfg = make_velocity_env_cfg()

    _configure_command(cfg)
    _configure_scene_and_sensors(cfg)
    _configure_actions(cfg)
    _configure_events(cfg)
    _configure_rewards(cfg)
    if rough:
        _configure_rough_rewards(cfg)
        _configure_rough_terrain(cfg)
    else:
        _configure_flat_terrain(cfg)
    _configure_observations(cfg)
    if rough:
        _configure_rough_privileged_observation(cfg)
    _configure_terminations(cfg)
    if rough:
        _configure_rough_terminations(cfg)
    _configure_common_runtime(cfg)
    if rough and not play:
        _configure_terrain_curriculum(cfg)
    if play:
        _configure_play(cfg)

    return cfg


def black_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground velocity task for Black.

    actor 45 维单帧；critic 72 维（MjLab-derived privileged baseline，不含
    height_scan）；terrain 为 plane，scene 里没有 terrain_scan。
    """
    return _build_black_env_cfg(play=play, rough=False)


def black_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the rough-terrain velocity task for Black（flat contract + rough terrain）。

    actor 仍为 45 维单帧（不含 height_scan）；critic 在 flat 的 72 维之后追加 187 维
    terrain height scan，共 259 维。terrain 装配见 `_configure_rough_terrain()`。

    play 模式沿用 flat play contract（无 DR / 无 corruption / episode 极长 /
    `curriculum = {}`），并保留同一 generator 布局与 env 分配比例，因此 play 看到的
    terrain 分布与训练一致（与 MjLab native rough task 在 play 下切成 random 小网格
    的做法不同，见 MIGRATION.md）。observation schema 不因 play 改变：actor 45 /
    critic 259 / terrain_scan 均与训练相同（inference policy 不使用 critic）。
    """
    return _build_black_env_cfg(play=play, rough=True)
