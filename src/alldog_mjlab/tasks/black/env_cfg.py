"""Black flat/rough PPO/HIM environment, rewards and interface contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mdp_dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from alldog_mjlab.robots.black import get_black_robot_cfg
from alldog_mjlab.robots.black.black_constants import (
    BLACK_FOOT_NAMES,
    BLACK_JOINT_NAMES,
)

from .mdp import rewards
from .mdp.curriculums import (
    CURRICULUM_TERM_NAME,
    CommandCurriculumParams,
    ForwardSpeedCommandCurriculum,
)
from .mdp.events import DomainRandomizationParams
from .mdp.him import configure_him_observations, configure_him_terminal_targets
from .mdp.terminations import StuckTermination
from .mdp.terrain import BlackRoughSlopeTerrainCfg

COMMAND_CURRICULUM = CommandCurriculumParams(
    **{
        "enabled": True,
        "initial_lin_vel_x": (-1.0, 1.0),
        "max_abs_vx": 2.0,
        "step": 0.1,
        "threshold_low": 0.7,
        "threshold_offset": 0.1,
        "ema_alpha": 0.2,
        "required_passes": 2,
        "buffer_min": 256,
        "min_low_count": 8,
        "min_high_count": 4,
        "low_speed_min": 0.2,
        "low_high_split_ratio": 0.6,
    }
)
DEFAULT_DR = DomainRandomizationParams()


BLACK_ACTION_TERM_ORDER = (
    "joint_pos_fl",
    "joint_pos_fr",
    "joint_pos_rl",
    "joint_pos_rr",
)

BLACK_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)

BLACK_ACTOR_OBS_SCALE: dict[str, float | tuple[float, ...]] = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}

BLACK_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"

BLACK_COMMAND_NAME = "twist"

BLACK_TRACKING_VELOCITY_REWARD_TERM = "tracking_lin_vel"

BLACK_BASE_CRITIC_TERM_ORDER = (
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

BLACK_CRITIC_TERM_ORDER = BLACK_BASE_CRITIC_TERM_ORDER + ("height_scan",)

BLACK_TERRAIN_SCAN_SENSOR = "terrain_scan"

BLACK_TERRAIN_SCAN_BODY = "trunk"

BLACK_TERRAIN_SCAN_MAX_DISTANCE = 5.0


def black_rough_terrain_generator_cfg() -> TerrainGeneratorCfg:
    proportions = {
        "flat": 0.1,
        "smooth_slope_up": 0.05,
        "smooth_slope_down": 0.05,
        "rough_slope": 0.1,
        "discrete_obstacles": 0.2,
        "stairs_up": 0.25,
        "stairs_down": 0.25,
    }
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        difficulty_range=(0.0, 0.9),
        add_lights=True,
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=proportions["flat"]),
            "smooth_slope_up": HfPyramidSlopedTerrainCfg(
                proportion=proportions["smooth_slope_up"],
                slope_range=(0.0, 0.7),
                platform_width=3.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                inverted=False,
            ),
            "smooth_slope_down": HfPyramidSlopedTerrainCfg(
                proportion=proportions["smooth_slope_down"],
                slope_range=(0.0, 0.7),
                platform_width=3.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                inverted=True,
            ),
            "rough_slope": BlackRoughSlopeTerrainCfg(
                proportion=proportions["rough_slope"],
                slope_range=(0.0, 0.7),
                noise_base=0.015,
                noise_gain=0.1,
                noise_step=0.005,
                noise_downsample=0.2,
                base_thickness_ratio=1.0,
                platform_width=3.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
            ),
            "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                proportion=proportions["discrete_obstacles"],
                obstacle_height_mode="choice",
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=(0.06, 0.26),
                num_obstacles=20,
                platform_width=3.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                square_obstacles=False,
            ),
            "stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=proportions["stairs_up"],
                step_height_range=(0.05, 0.05 + 0.18),
                step_width=0.3,
                platform_width=3.0,
            ),
            "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=proportions["stairs_down"],
                step_height_range=(0.05, 0.05 + 0.18),
                step_width=0.3,
                platform_width=3.0,
            ),
        },
    )


def _actions(dr: DomainRandomizationParams) -> dict:
    leg_joint_names = {
        leg: tuple((f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")))
        for leg in BLACK_FOOT_NAMES
    }
    action_terms: dict[str, JointPositionActionCfg] = {
        f"joint_pos_{leg.lower()}": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=leg_joint_names[leg],
            scale=0.25,
            use_default_offset=True,
        )
        for leg in BLACK_FOOT_NAMES
    }
    assert tuple(action_terms) == BLACK_ACTION_TERM_ORDER
    return action_terms


def _events(dr_params: DomainRandomizationParams, rough: bool, play: bool):
    base_events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform, mode="reset"
        ),
        "reset_robot_joints": EventTermCfg(func=reset_joints_by_offset, mode="reset"),
    }
    trunk_cfg = SceneEntityCfg("robot", body_names=("trunk",))
    reset_base = replace(
        base_events["reset_base"],
        params={
            "pose_range": dict({}),
            "velocity_range": dict(
                {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                }
            ),
        },
    )
    base_joint_reset = base_events["reset_robot_joints"]
    assert base_joint_reset.func is reset_joints_by_offset
    joint_resets = {
        f"reset_{joint_group}_joints": replace(
            base_joint_reset,
            params={
                "position_range": position_range,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=tuple(
                        (f"{leg}_{joint_group}_joint" for leg in BLACK_FOOT_NAMES)
                    ),
                ),
            },
        )
        for joint_group, position_range in {
            "hip": (0.0, 0.0),
            "thigh": (-0.4007, 0.4007),
            "calf": (-0.5945, 0.5945),
        }.items()
    }
    events = {
        "reset_base": reset_base,
        **joint_resets,
        "foot_friction": EventTermCfg(
            func=mdp_dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    geom_names=tuple(
                        f"{leg}_foot_collision" for leg in BLACK_FOOT_NAMES
                    ),
                ),
                "operation": "abs",
                "ranges": dr_params.friction,
                "shared_random": True,
            },
        ),
        "payload_mass": EventTermCfg(
            func=mdp_dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "operation": "add",
                "ranges": dr_params.payload_mass,
            },
        ),
        "base_com": EventTermCfg(
            func=mdp_dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "operation": "add",
                "ranges": dict(dr_params.com_offset),
            },
        ),
        "pd_gains": EventTermCfg(
            func=mdp_dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": dr_params.kp_scale,
                "kd_range": dr_params.kd_scale,
                "operation": "scale",
            },
        ),
        "encoder_bias": EventTermCfg(
            func=mdp_dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": dr_params.encoder_bias,
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=dr_params.push_interval,
            params={"velocity_range": dict(dr_params.push_velocity)},
        ),
    }
    enabled = {
        "foot_friction": dr_params.friction_enabled,
        "payload_mass": dr_params.payload_enabled,
        "base_com": dr_params.com_enabled,
        "pd_gains": dr_params.pd_gain_enabled,
        "encoder_bias": dr_params.encoder_bias_enabled,
        "push_robot": dr_params.push_enabled,
    }
    events = {
        name: term
        for name, term in events.items()
        if name not in enabled or enabled[name]
    }
    return {
        name: term
        for name, term in events.items()
        if not play or name.startswith("reset_")
    }


def _observations(dr: DomainRandomizationParams, play: bool):
    actor_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": BLACK_COMMAND_NAME},
            scale=2.0,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
        ),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "biased": True,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BLACK_JOINT_NAMES, preserve_order=True
                ),
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BLACK_JOINT_NAMES, preserve_order=True
                )
            },
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    noise = {
        "base_ang_vel": (-0.3, 0.3),
        "projected_gravity": (-0.05, 0.05),
        "joint_pos": (-0.08, 0.08),
        "joint_vel": (-2.0, 2.0),
    }
    for name, term in actor_terms.items():
        term.scale = BLACK_ACTOR_OBS_SCALE[name]
        if name in noise:
            term.noise = UniformNoiseCfg(n_min=noise[name][0], n_max=noise[name][1])
    critic_joint_names = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    )
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=critic_joint_names, preserve_order=True
                )
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=critic_joint_names, preserve_order=True
                )
            },
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": BLACK_COMMAND_NAME}
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": BLACK_TERRAIN_SCAN_SENSOR},
            scale=0.2,
        ),
    }
    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True, enable_corruption=not play
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False
        ),
    }


def make_env_cfg(
    rough: bool = False,
    him_enabled: bool = False,
    play: bool = False,
    dr_profile: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    dr_profile = deepcopy(DEFAULT_DR) if play or dr_profile is None else dr_profile

    sensors = [
        RayCastSensorCfg(
            name=BLACK_TERRAIN_SCAN_SENSOR,
            frame=ObjRef(type="body", name="trunk", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
            max_distance=5.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=True,
        ),
        TerrainHeightSensorCfg(
            name="foot_height_scan",
            frame=tuple(
                ObjRef(type="site", name=leg, entity="robot")
                for leg in BLACK_FOOT_NAMES
            ),
            pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
            ray_alignment="yaw",
            max_distance=1.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=True,
            viz=TerrainHeightSensorCfg.VizCfg(
                show_rays=True,
                hit_color=(1.0, 0.0, 1.0, 0.8),
                hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=tuple(f"{leg}_foot_collision" for leg in BLACK_FOOT_NAMES),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name=BLACK_ILLEGAL_CONTACT_SENSOR,
            primary=ContactMatch(
                mode="body",
                pattern=("trunk", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    ]

    terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
    if not play:
        terminations["illegal_contact"] = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_name": BLACK_ILLEGAL_CONTACT_SENSOR,
                "force_threshold": 1.0,
            },
        )
        if rough:
            terminations["out_of_terrain_bounds"] = TerminationTermCfg(
                func=mdp.out_of_terrain_bounds, time_out=True
            )
    terminations["stuck"] = TerminationTermCfg(
        func=StuckTermination,
        params={
            "command_name": BLACK_COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": 0.2,
            "velocity_threshold": 0.05,
            "grace_s": 1.0,
            "timeout_s": 4.0,
        },
    )
    curriculum = {}
    if not play:
        COMMAND_CURRICULUM.validate()
        if rough:
            curriculum["terrain_levels"] = CurriculumTermCfg(
                func=mdp.terrain_levels_vel, params={"command_name": BLACK_COMMAND_NAME}
            )
        if COMMAND_CURRICULUM.enabled:
            curriculum[CURRICULUM_TERM_NAME] = CurriculumTermCfg(
                func=ForwardSpeedCommandCurriculum,
                params={
                    "command_name": BLACK_COMMAND_NAME,
                    "reward_term_name": "tracking_lin_vel",
                    "stage": "rough" if rough else "flat",
                    "robot": "black",
                    "curriculum_params": COMMAND_CURRICULUM,
                },
            )
    metrics = {"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)}
    sim_cls = SimulationCfg
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": get_black_robot_cfg()},
            terrain=TerrainEntityCfg(
                terrain_type="generator" if rough else "plane",
                terrain_generator=black_rough_terrain_generator_cfg()
                if rough
                else None,
                max_init_terrain_level=5,
            ),
            sensors=tuple(sensors),
            num_envs=1 if play else 4096,
            extent=2.0,
        ),
        observations=_observations(dr_profile, play),
        actions=_actions(dr_profile),
        events=_events(dr_profile, rough, play),
        commands={
            BLACK_COMMAND_NAME: UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(10.0, 10.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-3.141592653589793, 3.141592653589793),
                    heading=None,
                ),
                heading_command=False,
                heading_control_stiffness=0.5,
                rel_heading_envs=0.0,
                rel_standing_envs=0.1,
                rel_forward_envs=0.2,
                rel_world_envs=0.0,
                init_velocity_prob=0.0,
                debug_vis=True,
            )
        },
        rewards={
            "tracking_lin_vel": RewardTermCfg(
                func=rewards.track_linear_velocity_xy,
                weight=1.0,
                params={"command_name": BLACK_COMMAND_NAME, "sigma": 0.25},
            ),
            "tracking_ang_vel": RewardTermCfg(
                func=rewards.track_angular_velocity_z,
                weight=0.5,
                params={"command_name": BLACK_COMMAND_NAME, "sigma": 0.25},
            ),
            "lin_vel_z": RewardTermCfg(
                func=rewards.vertical_linear_velocity_l2, weight=-2.0
            ),
            "ang_vel_xy": RewardTermCfg(
                func=rewards.angular_velocity_xy_l2, weight=-0.05
            ),
            "orientation": RewardTermCfg(
                func=envs_mdp.flat_orientation_l2, weight=-0.2
            ),
            "base_height": RewardTermCfg(
                func=rewards.base_height_l2_terrain
                if rough
                else rewards.base_height_l2_flat,
                weight=-1.0,
                params={
                    "target_height": 0.43,
                    **({"sensor_name": BLACK_TERRAIN_SCAN_SENSOR} if rough else {}),
                },
            ),
            "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
            "joint_power": RewardTermCfg(
                func=rewards.joint_power_l1,
                weight=-2e-5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=BLACK_JOINT_NAMES, preserve_order=True
                    )
                },
            ),
        },
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="trunk",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=sim_cls(
            nconmax=128 if rough else 35,
            njmax=256,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=int(1e9) if play else 20.0,
        scale_rewards_by_dt=True,
    )
    if him_enabled:
        configure_him_observations(cfg)
        configure_him_terminal_targets(cfg)
    return cfg


def black_flat_env_cfg(play: bool = False, dr: DomainRandomizationParams | None = None):
    return make_env_cfg(play=play, dr_profile=dr)


def black_rough_env_cfg(
    play: bool = False, dr: DomainRandomizationParams | None = None
):
    return make_env_cfg(rough=True, play=play, dr_profile=dr)


def black_flat_him_env_cfg(
    play: bool = False, dr: DomainRandomizationParams | None = None
):
    return make_env_cfg(him_enabled=True, play=play, dr_profile=dr)


def black_rough_him_env_cfg(
    play: bool = False, dr: DomainRandomizationParams | None = None
):
    return make_env_cfg(rough=True, him_enabled=True, play=play, dr_profile=dr)
