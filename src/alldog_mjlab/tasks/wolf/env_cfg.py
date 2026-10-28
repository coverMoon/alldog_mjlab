"""Wolf flat/rough PPO/HIM environment, rewards and interface contract."""

from __future__ import annotations

import copy
from copy import deepcopy
from dataclasses import dataclass, replace

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mdp_dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
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

from alldog_mjlab.robots.wolf import get_wolf_robot_cfg
from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_JOINT_NAMES,
    WOLF_LEG_ORDER,
    WOLF_POLICY_JOINT_NAMES,
    WOLF_WHEEL_BODY_NAMES,
    WOLF_WHEEL_COLLISION_GEOM_NAMES,
    WOLF_WHEEL_FORWARD_SIGN,
    WOLF_WHEEL_JOINT_NAMES,
)

from .mdp import rewards
from .mdp.curriculums import (
    CURRICULUM_TERM_NAME,
    CommandCurriculumParams,
    WolfForwardSpeedCommandCurriculum,
)
from .mdp.events import (
    BiasedSignedWheelVelocity,
    CalfBacklashPositionActionCfg,
    DomainRandomizationParams,
    ScaledBiasedWheelVelocityActionCfg,
    randomize_body_inertia_scale,
    randomize_ground_friction,
    randomize_leg_pd_gains_strength,
    randomize_wheel_friction_multiplier,
    randomize_wheel_motor_strength,
    randomize_wheel_radius_with_height,
    reset_joints_by_default_scale,
)
from .mdp.him import configure_him_observations, configure_him_terminal_targets
from .mdp.observations import signed_wheel_velocity
from .mdp.terrain import WolfRoughSlopeTerrainCfg

COMMAND_CURRICULUM = CommandCurriculumParams(
    **{
        "enabled": True,
        "initial_lin_vel_x": (-1.0, 1.0),
        "max_abs_vx": 3.0,
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


_WF_HFIELD_OVERFLOW_BIT = 1 << 5

_WF_ITERATIONS_OVERFLOW_BIT = 1 << 9

_WF_LS_ITERATIONS_OVERFLOW_BIT = 1 << 10

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

WOLF_ACTION_DIM = sum(
    (
        _WOLF_ACTION_TERM_DIMS[
            "joint_pos" if name.startswith("joint_pos") else "wheel_vel"
        ]
        for name in WOLF_ACTION_TERM_ORDER
    )
)

assert WOLF_ACTION_DIM == 16

WOLF_WHEEL_ACTION_CHANNELS = (3, 7, 11, 15)

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

WOLF_CRITIC_TERM_ORDER = WOLF_ACTOR_OBS_TERM_ORDER + ("base_lin_vel",)

WOLF_CRITIC_DIM = WOLF_ACTOR_OBS_DIM + 3

assert WOLF_CRITIC_DIM == 56

_WOLF_ACTOR_TERM_SCALE = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "leg_joint_pos": 1.0,
    "leg_joint_vel": 0.05,
    "wheel_vel": 0.05,
    "actions": 1.0,
}

_WOLF_ACTOR_TERM_NOISE = {
    "base_ang_vel": (-0.3, 0.3),
    "projected_gravity": (-0.05, 0.05),
    "leg_joint_pos": (-0.08, 0.08),
    "leg_joint_vel": (-2.0, 2.0),
    "wheel_vel": (-2.0, 2.0),
}

WOLF_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"

WOLF_WHEEL_FORCE_CONTACT_SENSOR = "wheel_force_contact"

WOLF_TERRAIN_SCAN_SENSOR = "terrain_scan"

WOLF_TERRAIN_CURRICULUM_TERM = "terrain_levels"

WOLF_TEMPLATE_ROOT_BODY = "robot/base_link"

WOLF_COMMAND_NAME = "twist"

WOLF_TRACKING_VELOCITY_REWARD_TERM = "tracking_lin_vel"

WOLF_LEG_ACTION_JOINT_NAMES = {
    leg: tuple(WOLF_LEG_JOINT_NAMES[leg][:3]) for leg in WOLF_LEG_ORDER
}

WOLF_WHEEL_JOINT_NAME = {leg: (WOLF_LEG_JOINT_NAMES[leg][3],) for leg in WOLF_LEG_ORDER}

WOLF_LEGS_FL_FIRST_JOINT_NAMES = tuple(
    (name for leg in WOLF_LEG_ORDER for name in WOLF_LEG_ACTION_JOINT_NAMES[leg])
)

WOLF_WHEEL_JOINT_POLICY_ORDER = tuple((f"{leg}_foot" for leg in WOLF_LEG_ORDER))


def metric_overflow_hfield(env: "ManagerBasedRlEnv") -> torch.Tensor:
    bits = env.sim.data.overflow[:]
    return (bits & _WF_HFIELD_OVERFLOW_BIT != 0).to(
        device=bits.device, dtype=torch.float32
    )


def metric_overflow_other(env: "ManagerBasedRlEnv") -> torch.Tensor:
    bits = env.sim.data.overflow[:]
    return (bits & ~_WF_HFIELD_OVERFLOW_BIT != 0).to(
        device=bits.device, dtype=torch.float32
    )


def metric_overflow_solver_iterations(env: "ManagerBasedRlEnv") -> torch.Tensor:
    bits = env.sim.data.overflow[:]
    return (bits & _WF_ITERATIONS_OVERFLOW_BIT != 0).to(
        device=bits.device, dtype=torch.float32
    )


def metric_overflow_ls_iterations(env: "ManagerBasedRlEnv") -> torch.Tensor:
    bits = env.sim.data.overflow[:]
    return (bits & _WF_LS_ITERATIONS_OVERFLOW_BIT != 0).to(
        device=bits.device, dtype=torch.float32
    )


@dataclass(kw_only=True)
class WolfRoughSimulationCfg(SimulationCfg):
    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = False


def _write_template_root_height(spec: "mujoco.MjSpec", root_z: float) -> None:
    body = next((b for b in spec.bodies if b.name == WOLF_TEMPLATE_ROOT_BODY), None)
    if body is None:
        raise ValueError(
            f"Wolf 模板 spec_fn：找不到根 body {WOLF_TEMPLATE_ROOT_BODY!r}，spec 结构与预期不符，fail-loud"
        )
    free_joints = [
        j for j in spec.joints if int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise ValueError(
            f"Wolf 模板 spec_fn：根 body 下自由关节数量应为 1，实际 {len(free_joints)}（joints={[j.name for j in free_joints]}），fail-loud"
        )
    body.pos[:] = [0.0, 0.0, root_z]


def _wolf_flat_template_spec_fn(spec: "mujoco.MjSpec") -> None:
    _write_template_root_height(spec, 0.45)


def _wolf_rough_template_spec_fn(spec: "mujoco.MjSpec") -> None:
    _write_template_root_height(spec, 10.0)


def _wolf_robot_cfg(dr: DomainRandomizationParams) -> "EntityCfg":
    robot_cfg = get_wolf_robot_cfg()
    if not (dr.leg_delay_enabled or dr.wheel_delay_enabled):
        return robot_cfg
    robot_cfg = copy.deepcopy(robot_cfg)
    decimation = 4
    for act in robot_cfg.articulation.actuators:
        joint_name = act.target_names_expr[0]
        if joint_name in WOLF_WHEEL_JOINT_NAMES:
            if dr.wheel_delay_enabled:
                max_lag = dr.wheel_max_delay_steps * decimation
                act.delay_min_lag = 0
                act.delay_max_lag = max_lag
                act.delay_update_period = decimation
                act.delay_per_env_phase = False
        elif dr.leg_delay_enabled:
            max_lag = dr.leg_max_delay_steps * decimation
            act.delay_min_lag = 0
            act.delay_max_lag = max_lag
            act.delay_update_period = decimation
            act.delay_per_env_phase = False
    return robot_cfg


def _wheel_vel_term_cfg(dr: DomainRandomizationParams) -> ObservationTermCfg:
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


WOLF_COLLISION_SENSOR = "body_ground_contact"
WOLF_COLLISION_BODY_NAMES = ("base_link",) + tuple(
    f"{leg}_Link{index}" for leg in WOLF_LEG_ORDER for index in (2, 3)
)


def wolf_rough_terrain_generator_cfg() -> TerrainGeneratorCfg:
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
                inverted=True,
            ),
            "smooth_slope_down": HfPyramidSlopedTerrainCfg(
                proportion=proportions["smooth_slope_down"],
                slope_range=(0.0, 0.7),
                platform_width=3.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                inverted=False,
            ),
            "rough_slope": WolfRoughSlopeTerrainCfg(
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
            "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
                proportion=proportions["stairs_up"],
                step_height_range=(0.05, 0.05 + 0.28),
                step_width=0.3,
                platform_width=3.0,
            ),
            "stairs_down": BoxPyramidStairsTerrainCfg(
                proportion=proportions["stairs_down"],
                step_height_range=(0.05, 0.05 + 0.28),
                step_width=0.3,
                platform_width=3.0,
            ),
        },
    )


def _actions(dr: DomainRandomizationParams) -> dict:
    backlash_on = dr.calf_backlash.enabled
    wheel_target_on = (
        dr.wheel_target.vel_ref_scale_enabled or dr.wheel_target.vel_ref_bias_enabled
    )

    def _position_cfg(leg: str) -> JointPositionActionCfg:
        base = dict(
            entity_name="robot",
            actuator_names=WOLF_LEG_ACTION_JOINT_NAMES[leg],
            scale=0.25,
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
            scale={WOLF_LEG_JOINT_NAMES[leg][3]: 10.0 * WOLF_WHEEL_FORWARD_SIGN[leg]},
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
    return action_terms


def _events(dr: DomainRandomizationParams, rough: bool, play: bool):
    base_events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform, mode="reset"
        ),
        "reset_robot_joints": EventTermCfg(func=reset_joints_by_offset, mode="reset"),
    }
    base_joint_reset = base_events["reset_robot_joints"]
    assert base_joint_reset.func is reset_joints_by_offset
    reset_base = replace(
        base_events["reset_base"],
        params={
            "pose_range": dict({}),
            "velocity_range": dict(
                {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.3, 0.3),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.5, 0.5),
                }
            ),
        },
    )
    if dr.initial_joint_pos_enabled:
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
            )
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
                        joint_names=tuple(
                            (f"{leg}_{joint_group}" for leg in WOLF_LEG_ORDER)
                        ),
                    ),
                },
            )
            for joint_group, position_range in {
                "hip": (0.0, 0.0),
                "thigh": (-0.3, 0.3),
                "calf": (-0.3, 0.3),
            }.items()
        }
    wheel_reset = replace(
        base_joint_reset,
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=WOLF_WHEEL_JOINT_POLICY_ORDER
            ),
        },
    )
    events = {
        "reset_base": reset_base,
        **leg_resets,
        "reset_wheel_joints": wheel_reset,
        **_dr_event_terms(dr, rough),
    }
    return {
        name: term
        for name, term in events.items()
        if not play or name.startswith("reset_")
    }


def _dr_event_terms(
    dr: DomainRandomizationParams, rough: bool
) -> dict[str, EventTermCfg]:
    terms: dict[str, EventTermCfg] = {}
    all_collision_geoms = SceneEntityCfg("robot", geom_names=(".*",))
    wheel_geoms = SceneEntityCfg(
        "robot", geom_names=WOLF_WHEEL_COLLISION_GEOM_NAMES, preserve_order=True
    )
    terrain_geoms = (
        SceneEntityCfg("terrain", geom_names=(".*",))
        if rough
        else SceneEntityCfg("terrain", geom_names=("terrain",))
    )
    base_body = SceneEntityCfg("robot", body_names=("base_link",))
    wheel_bodies = SceneEntityCfg("robot", body_names=WOLF_WHEEL_BODY_NAMES)
    link_bodies = SceneEntityCfg(
        "robot",
        body_names=tuple(
            (f"{leg}_Link{i}" for leg in WOLF_LEG_ORDER for i in (1, 2, 3))
        ),
    )
    leg_actuators = SceneEntityCfg(
        "robot",
        actuator_names=tuple(
            (
                f"{leg}_{joint}"
                for leg in WOLF_LEG_ORDER
                for joint in ("hip", "thigh", "calf")
            )
        ),
    )
    wheel_actuators = SceneEntityCfg("robot", actuator_names=WOLF_WHEEL_JOINT_NAMES)
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
    if (
        dr.kp_enabled
        or dr.kd_enabled
        or dr.motor_strength_enabled
        or dr.hip_motor_strength_enabled
    ):
        terms["dr_leg_pd_gains"] = EventTermCfg(
            func=randomize_leg_pd_gains_strength,
            mode="reset",
            params={
                "kp_range": dr.kp_scale_range if dr.kp_enabled else (1.0, 1.0),
                "kd_range": dr.kd_scale_range if dr.kd_enabled else (1.0, 1.0),
                "motor_strength_range": dr.motor_strength_range
                if dr.motor_strength_enabled
                else (1.0, 1.0),
                "hip_strength_range": dr.hip_motor_strength_range
                if dr.hip_motor_strength_enabled
                else None,
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
    if dr.wheel_radius_enabled:
        terms["dr_wheel_radius"] = EventTermCfg(
            func=randomize_wheel_radius_with_height,
            mode="reset",
            params={
                "scale_range": dr.wheel_radius_scale_range,
                "asset_cfg": wheel_geoms,
            },
        )
    if dr.push_enabled:
        terms["dr_push_robot"] = EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=dr.push_interval_s,
            is_global_time=True,
            params={
                "velocity_range": {"x": dr.push_velocity_xy, "y": dr.push_velocity_xy}
            },
        )
    if dr.disturbance_enabled:
        disturbance_interval_s = dr.disturbance_interval_policy_steps * 0.02
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


def _observations(dr: DomainRandomizationParams, play: bool):
    def _actor_term(
        name: str, func, params: dict | None = None
    ) -> tuple[str, ObservationTermCfg]:
        noise_range = _WOLF_ACTOR_TERM_NOISE.get(name)
        term = ObservationTermCfg(
            func=func,
            params=dict(params or {}),
            scale=_WOLF_ACTOR_TERM_SCALE[name],
            noise=UniformNoiseCfg(n_min=noise_range[0], n_max=noise_range[1])
            if noise_range is not None
            else None,
        )
        return (name, term)

    actor_built = dict(
        (
            _actor_term(
                "command", mdp.generated_commands, {"command_name": WOLF_COMMAND_NAME}
            ),
            _actor_term(
                "base_ang_vel", mdp.builtin_sensor, {"sensor_name": "robot/imu_ang_vel"}
            ),
            _actor_term(
                "projected_gravity",
                mdp.projected_gravity_from_sensor,
                {"sensor_name": "robot/imu_upvector"},
            ),
            _actor_term(
                "leg_joint_pos",
                mdp.joint_pos_rel,
                {
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    )
                },
            ),
            _actor_term(
                "leg_joint_vel",
                mdp.joint_vel_rel,
                {
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    )
                },
            ),
            ("wheel_vel", _wheel_vel_term_cfg(dr)),
            _actor_term("actions", mdp.last_action),
        )
    )
    assert tuple(actor_built) == WOLF_ACTOR_OBS_TERM_ORDER
    critic_built: dict[str, ObservationTermCfg] = {}
    for name, term in actor_built.items():
        func = term.func
        params = dict(term.params)
        if name == "wheel_vel" and dr.wheel_obs_bias.enabled:
            func = signed_wheel_velocity
            params = {"asset_cfg": params["asset_cfg"]}
        critic_built[name] = ObservationTermCfg(
            func=func, params=params, scale=term.scale
        )
    critic_built["base_lin_vel"] = ObservationTermCfg(
        func=envs_mdp.base_lin_vel, params={}, scale=1.0
    )
    assert tuple(critic_built) == WOLF_CRITIC_TERM_ORDER
    return {
        "actor": ObservationGroupCfg(
            terms=actor_built, concatenate_terms=True, enable_corruption=not play
        ),
        "critic": ObservationGroupCfg(
            terms=critic_built, concatenate_terms=True, enable_corruption=False
        ),
    }


def make_env_cfg(
    rough: bool = False,
    him_enabled: bool = False,
    play: bool = False,
    dr_profile: DomainRandomizationParams | None = None,
) -> ManagerBasedRlEnvCfg:
    dr_profile = deepcopy(DEFAULT_DR) if play or dr_profile is None else dr_profile
    dr_profile.validate()
    sensors = [
        ContactSensorCfg(
            name=WOLF_ILLEGAL_CONTACT_SENSOR,
            primary=ContactMatch(mode="body", pattern=("base_link",), entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name=WOLF_COLLISION_SENSOR,
            primary=ContactMatch(
                mode="body", pattern=WOLF_COLLISION_BODY_NAMES, entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name=WOLF_WHEEL_FORCE_CONTACT_SENSOR,
            primary=ContactMatch(
                mode="geom", pattern=WOLF_WHEEL_COLLISION_GEOM_NAMES, entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    ]
    if rough:
        sensors.append(
            RayCastSensorCfg(
                name=WOLF_TERRAIN_SCAN_SENSOR,
                frame=ObjRef(type="body", name="base_link", entity="robot"),
                ray_alignment="yaw",
                pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
                max_distance=5.0,
                exclude_parent_body=True,
                include_geom_groups=(0,),
                debug_vis=True,
            )
        )

    terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
    if not play:
        terminations["illegal_contact"] = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": WOLF_ILLEGAL_CONTACT_SENSOR, "force_threshold": 1.0},
        )
        if rough:
            terminations["out_of_terrain_bounds"] = TerminationTermCfg(
                func=mdp.out_of_terrain_bounds, time_out=True
            )
    curriculum = {}
    if not play:
        COMMAND_CURRICULUM.validate()
        if rough:
            curriculum["terrain_levels"] = CurriculumTermCfg(
                func=mdp.terrain_levels_vel, params={"command_name": WOLF_COMMAND_NAME}
            )
        if COMMAND_CURRICULUM.enabled:
            curriculum[CURRICULUM_TERM_NAME] = CurriculumTermCfg(
                func=WolfForwardSpeedCommandCurriculum,
                params={
                    "command_name": WOLF_COMMAND_NAME,
                    "reward_term_name": "tracking_lin_vel",
                    "stage": "rough" if rough else "flat",
                    "robot": "wolf",
                    "curriculum_params": COMMAND_CURRICULUM,
                },
            )
    metrics = {"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)}
    if rough and not play:
        metrics.update(
            {
                "overflow_hfield": MetricsTermCfg(
                    func=metric_overflow_hfield, reduce="max"
                ),
                "overflow_other": MetricsTermCfg(
                    func=metric_overflow_other, reduce="max"
                ),
                "overflow_solver_iterations": MetricsTermCfg(
                    func=metric_overflow_solver_iterations, reduce="max"
                ),
                "overflow_ls_iterations": MetricsTermCfg(
                    func=metric_overflow_ls_iterations, reduce="max"
                ),
            }
        )
    sim_cls = (
        WolfRoughSimulationCfg if rough and not play and not True else SimulationCfg
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": _wolf_robot_cfg(dr_profile)},
            terrain=TerrainEntityCfg(
                terrain_type="generator" if rough else "plane",
                terrain_generator=wolf_rough_terrain_generator_cfg() if rough else None,
                max_init_terrain_level=5,
            ),
            sensors=tuple(sensors),
            num_envs=1 if play else 4096,
            extent=2.0,
            spec_fn=_wolf_rough_template_spec_fn
            if rough
            else _wolf_flat_template_spec_fn,
        ),
        observations=_observations(dr_profile, play),
        actions=_actions(dr_profile),
        events=_events(dr_profile, rough, play),
        commands={
            WOLF_COMMAND_NAME: UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(10.0, 10.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-3.14, 3.14),
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
                weight=1.5,
                params={"command_name": WOLF_COMMAND_NAME, "sigma": 0.25},
            ),
            "tracking_ang_vel": RewardTermCfg(
                func=rewards.track_angular_velocity_z,
                weight=0.75,
                params={"command_name": WOLF_COMMAND_NAME, "sigma": 0.25},
            ),
            "lin_vel_z": RewardTermCfg(
                func=rewards.vertical_linear_velocity_l2, weight=-1.0
            ),
            "ang_vel_xy": RewardTermCfg(
                func=rewards.angular_velocity_xy_l2, weight=-0.05
            ),
            "orientation": RewardTermCfg(
                func=envs_mdp.flat_orientation_l2, weight=-0.5
            ),
            "base_height": RewardTermCfg(
                func=rewards.base_height_l2_terrain
                if rough
                else rewards.base_height_l2_flat,
                weight=-10.0,
                params={
                    "target_height": 0.4,
                    **({"sensor_name": WOLF_TERRAIN_SCAN_SENSOR} if rough else {}),
                },
            ),
            "hip_default": RewardTermCfg(
                func=rewards.hip_default,
                weight=-0.5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=tuple(f"{leg}_hip" for leg in WOLF_LEG_ORDER),
                        preserve_order=True,
                    )
                },
            ),
            "stand_still": RewardTermCfg(
                func=rewards.stand_still,
                weight=-0.5,
                params={
                    "command_name": WOLF_COMMAND_NAME,
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    ),
                    "command_threshold": 0.1,
                },
            ),
            "collision": RewardTermCfg(
                func=rewards.collision,
                weight=-1.0,
                params={
                    "sensor_name": WOLF_COLLISION_SENSOR,
                    "body_names": WOLF_COLLISION_BODY_NAMES,
                    "force_threshold": 0.1,
                },
            ),
            "feet_stumble": RewardTermCfg(
                func=rewards.feet_stumble,
                weight=-0.1,
                params={
                    "sensor_name": WOLF_WHEEL_FORCE_CONTACT_SENSOR,
                    "geom_names": WOLF_WHEEL_COLLISION_GEOM_NAMES,
                    "horizontal_ratio": 3.0,
                },
            ),
            "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
            "torques": RewardTermCfg(
                func=envs_mdp.joint_torques_l2,
                weight=-5e-4,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        actuator_names=WOLF_POLICY_JOINT_NAMES,
                        preserve_order=True,
                    )
                },
            ),
            "dof_vel": RewardTermCfg(
                func=envs_mdp.joint_vel_l2,
                weight=-1e-7,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    )
                },
            ),
            "dof_acc": RewardTermCfg(
                func=envs_mdp.joint_acc_l2,
                weight=-1e-7,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_POLICY_JOINT_NAMES,
                        preserve_order=True,
                    )
                },
            ),
            "run_still": RewardTermCfg(
                func=rewards.run_still,
                weight=-0.05,
                params={
                    "command_name": WOLF_COMMAND_NAME,
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=WOLF_LEGS_FL_FIRST_JOINT_NAMES,
                        preserve_order=True,
                    ),
                    "command_threshold": 0.1,
                },
            ),
        },
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.8,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=sim_cls(
            nconmax=128 if rough else 64,
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


def wolf_flat_env_cfg(play: bool = False, dr: DomainRandomizationParams | None = None):
    return make_env_cfg(play=play, dr_profile=dr)


def wolf_rough_env_cfg(play: bool = False, dr: DomainRandomizationParams | None = None):
    return make_env_cfg(rough=True, play=play, dr_profile=dr)


def wolf_flat_him_env_cfg(
    play: bool = False, dr: DomainRandomizationParams | None = None
):
    return make_env_cfg(him_enabled=True, play=play, dr_profile=dr)


def wolf_rough_him_env_cfg(
    play: bool = False, dr: DomainRandomizationParams | None = None
):
    return make_env_cfg(rough=True, him_enabled=True, play=play, dr_profile=dr)
