"""Velocity task configurations for Black."""

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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from nwwolf_mjlab.robots.black import BLACK_ACTION_SCALE, get_black_robot_cfg
from nwwolf_mjlab.robots.black.black_constants import BLACK_FOOT_NAMES


def black_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground velocity task for Black."""

    cfg = make_velocity_env_cfg()

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

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = BLACK_ACTION_SCALE

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
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg
