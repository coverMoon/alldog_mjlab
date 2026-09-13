"""Independent flat and rough HIM tasks for Black."""
from dataclasses import dataclass
import torch
from mjlab.envs import mdp as builtin
from mjlab.envs.mdp import dr
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch, TerrainHeightSensorCfg, RayCastSensorCfg, ObjRef, GridPatternCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from ..env_cfgs import black_flat_env_cfg
from .control import robot_cfg, DelayedPositionActionCfg, LEGS
from .source_config import BlackCfg
from . import mdp
from .events import BlackCommandCfg, CommandCurriculum, reset_joints, perturbation, randomize_inertia, randomize_gains
from .terrain import terrain_cfg


@dataclass
class SourceGrid(GridPatternCfg):
    """Source uses ij ordering; body scan has independent x/y spacing."""
    nx: int = 17
    ny: int = 11

    def generate_rays(self, mj_model, device):
        x = torch.linspace(-self.size[0]/2, self.size[0]/2, self.nx, device=device)
        y = torch.linspace(-self.size[1]/2, self.size[1]/2, self.ny, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        offsets = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx).flatten()), -1)
        directions = torch.zeros_like(offsets)
        directions[:, 2] = -1
        return offsets, directions


def black_him_env_cfg(play=False, rough=False):
    cfg = black_flat_env_cfg(play=False)
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.num_envs = 4096
    cfg.sim.mujoco.timestep = .005
    cfg.decimation = 4
    cfg.episode_length_s = 20.
    cfg.sim.nconmax = 128
    cfg.scene.sensors = (
        ContactSensorCfg(name="body_contact", primary=ContactMatch(mode="body", pattern=("trunk", ".*_hip", ".*_thigh", ".*_calf", ".*_foot"), entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1),
        RayCastSensorCfg(name="terrain_scan", frame=ObjRef(type="body", name="trunk", entity="robot"), ray_alignment="yaw", pattern=SourceGrid(size=(1.6, 1.)), max_distance=5., include_geom_groups=(0,), exclude_parent_body=True),
        RayCastSensorCfg(name="body_height_scan", frame=ObjRef(type="body", name="trunk", entity="robot"), ray_alignment="yaw", pattern=SourceGrid(size=(.6, .36), nx=11, ny=9), max_distance=5., include_geom_groups=(0,), exclude_parent_body=True),
        TerrainHeightSensorCfg(name="foot_height_scan", frame=tuple(ObjRef(type="site", name=leg, entity="robot") for leg in LEGS), pattern=SourceGrid(size=(0.,0.), nx=1, ny=1), max_distance=3., include_geom_groups=(0,), exclude_parent_body=True),
    )
    cfg.actions = {"joint_pos": DelayedPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=.25, max_lag=0 if play else 2)}
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={"history": ObservationTermCfg(func=mdp.actor_observation)}, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms={"privileged": ObservationTermCfg(func=mdp.critic_observation)}),
    }
    cfg.commands = {"twist": BlackCommandCfg(entity_name="robot", resampling_time_range=(10.,10.), heading_command=False, rel_heading_envs=0., rel_standing_envs=0., rel_forward_envs=0., ranges=BlackCommandCfg.Ranges(lin_vel_x=(-1.,1.), lin_vel_y=(-1.,1.), ang_vel_z=(-3.14,3.14), heading=None))}
    weights = {k:v for k,v in vars(BlackCfg.rewards.scales).items() if not k.startswith('_') and v != 0}
    cfg.rewards = {name: RewardTermCfg(func=mdp.reward, weight=weight, params={"name":name}) for name,weight in weights.items()}
    cfg.terminations = {"time_out": TerminationTermCfg(func=builtin.time_out, time_out=True), "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact), "stuck": TerminationTermCfg(func=mdp.stuck)}
    cfg.metrics = {}
    cfg.events = {
        "reset_base": EventTermCfg(func=builtin.reset_root_state_uniform, mode="reset", params={"pose_range":({"x":(-1.,1.), "y":(-1.,1.)} if rough else {}), "velocity_range":{k:(-.5,.5) for k in ("x","y","z","roll","pitch","yaw")}}),
        "reset_joints": EventTermCfg(func=reset_joints, mode="reset"),
        "reset_history": EventTermCfg(func=mdp.reset_state, mode="reset"),
        "friction": EventTermCfg(func=dr.geom_friction, mode="startup", params={"asset_cfg":SceneEntityCfg("robot", geom_names=("collision_.*",)), "operation":"abs", "ranges":(.3,1.35), "shared_random":True}),
        "payload": EventTermCfg(func=dr.body_mass, mode="startup", params={"asset_cfg":SceneEntityCfg("robot", body_names=("trunk",)), "operation":"add", "ranges":(-2.,4.)}),
        "link_mass": EventTermCfg(func=dr.body_mass, mode="startup", params={"asset_cfg":SceneEntityCfg("robot", body_names=(".*_hip", ".*_thigh", ".*_calf", ".*_foot")), "operation":"scale", "ranges":(.75,1.25)}),
        "base_com": EventTermCfg(func=dr.body_com_offset, mode="startup", params={"asset_cfg":SceneEntityCfg("robot", body_names=("trunk",)), "operation":"add", "ranges":(-.05,.05)}),
        "inertia": EventTermCfg(func=randomize_inertia, mode="startup", params={"asset_cfg":SceneEntityCfg("robot", body_names=("trunk", ".*_hip", ".*_thigh", ".*_calf", ".*_foot"))}),
        "gains": EventTermCfg(func=randomize_gains, mode="reset"),
        "push": EventTermCfg(func=builtin.push_by_setting_velocity, mode="interval", interval_range_s=(30.,30.), params={"velocity_range":{"x":(-2.5,2.5),"y":(-2.5,2.5)}}),
        "disturbance": EventTermCfg(func=perturbation, mode="step"),
    }
    cfg.curriculum = {"command_vel": CurriculumTermCfg(func=CommandCurriculum)}
    cfg.recorders = {"him_terminal": RecorderTermCfg(func=mdp.TerminalObservations)}
    if rough:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = terrain_cfg()
        cfg.scene.terrain.max_init_terrain_level = 5
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=velocity_mdp.terrain_levels_vel, params={"command_name":"twist"})
    if play:
        cfg.scene.num_envs = 1
        cfg.curriculum = {}
        cfg.events = {k:v for k,v in cfg.events.items() if k in ("reset_base","reset_joints","reset_history")}
    return cfg
