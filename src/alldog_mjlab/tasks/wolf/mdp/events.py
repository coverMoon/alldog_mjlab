"""Wolf domain randomization parameters, events and control perturbations."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import (
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from alldog_mjlab.robots.wolf.wolf_constants import WOLF_WHEEL_RADIUS
from alldog_mjlab.tasks.wolf.mdp.observations import signed_wheel_velocity


@dataclass(frozen=True)
class CalfBacklashParams:
    enabled: bool = False
    width_range: tuple[float, float] = (0.005, 0.035)
    min_kp_scale: float = 0.12
    engage_start: float = 0.6
    leak: float = 0.02


@dataclass(frozen=True)
class WheelTargetBiasParams:
    vel_ref_scale_enabled: bool = False
    vel_ref_scale_range: tuple[float, float] = (0.9, 1.1)
    vel_ref_bias_enabled: bool = False
    vel_ref_bias_range: tuple[float, float] = (-0.3, 0.3)


@dataclass(frozen=True)
class WheelObsBiasParams:
    enabled: bool = False
    range: tuple[float, float] = (-0.5, 0.5)


@dataclass(frozen=True)
class DomainRandomizationParams:
    base_mass_enabled: bool = True
    base_mass_range: tuple[float, float] = (-1.0, 2.0)
    base_com_enabled: bool = True
    base_com_offset_range: tuple[float, float] = (-0.05, 0.05)
    link_mass_enabled: bool = True
    link_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    link_inertia_enabled: bool = True
    link_inertia_scale_range: tuple[float, float] = (0.9, 1.1)
    wheel_mass_enabled: bool = True
    wheel_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    wheel_inertia_enabled: bool = True
    wheel_inertia_scale_range: tuple[float, float] = (0.8, 1.2)
    ground_friction_enabled: bool = True
    ground_friction_range: tuple[float, float] = (0.25, 1.25)
    wheel_friction_enabled: bool = True
    wheel_friction_scale_range: tuple[float, float] = (0.4, 1.0)
    kp_enabled: bool = True
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_enabled: bool = True
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    motor_strength_enabled: bool = True
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    hip_motor_strength_enabled: bool = False
    hip_motor_strength_range: tuple[float, float] = (0.8, 1.05)
    calf_backlash: CalfBacklashParams = field(default_factory=CalfBacklashParams)
    wheel_motor_enabled: bool = True
    wheel_motor_strength_range: tuple[float, float] = (0.8, 1.2)
    wheel_target: WheelTargetBiasParams = field(default_factory=WheelTargetBiasParams)
    wheel_obs_bias: WheelObsBiasParams = field(default_factory=WheelObsBiasParams)
    wheel_radius_enabled: bool = True
    wheel_radius_scale_range: tuple[float, float] = (0.9, 1.1)
    leg_delay_enabled: bool = True
    leg_max_delay_steps: int = 3
    wheel_delay_enabled: bool = True
    wheel_max_delay_steps: int = 4
    push_enabled: bool = True
    push_interval_s: tuple[float, float] = (15.0, 15.0)
    push_velocity_xy: tuple[float, float] = (-1.0, 1.0)
    disturbance_enabled: bool = True
    disturbance_force_range: tuple[float, float] = (-35.0, 35.0)
    disturbance_interval_policy_steps: int = 8
    initial_joint_pos_enabled: bool = True
    initial_joint_pos_range: tuple[float, float] = (0.5, 1.5)

    def validate(self) -> None:
        for name in (
            "base_mass_range",
            "base_com_offset_range",
            "link_mass_scale_range",
            "link_inertia_scale_range",
            "wheel_mass_scale_range",
            "wheel_inertia_scale_range",
            "ground_friction_range",
            "wheel_friction_scale_range",
            "kp_scale_range",
            "kd_scale_range",
            "motor_strength_range",
            "hip_motor_strength_range",
            "wheel_motor_strength_range",
            "wheel_radius_scale_range",
            "initial_joint_pos_range",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"dr.{name} 区间反序: {lo} > {hi}")
        for name in (
            "leg_max_delay_steps",
            "wheel_max_delay_steps",
            "disturbance_interval_policy_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"dr.{name} 必须非负")


def _resolve_env_ids(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    return env_ids.to(env.device, dtype=torch.int)


def _press_terrain_friction(
    env: ManagerBasedRlEnv, terrain_cfg: SceneEntityCfg, env_ids: torch.Tensor
) -> None:
    terrain = env.scene[terrain_cfg.name]
    env_grid, geom_grid = _resolve_geom_grid(env, terrain, terrain_cfg, env_ids)
    if geom_grid.numel() == 0:
        raise ValueError(
            f"terrain friction selector matched no geoms for entity {terrain_cfg.name!r} (geom_names={terrain_cfg.geom_names}); 摩擦 DR 将不会生效，禁止静默无效"
        )
    env.sim.model.geom_friction[env_grid, geom_grid, 0] = 0.0


def _resolve_geom_grid(
    env: ManagerBasedRlEnv,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mjlab.envs.mdp.dr._core import _get_entity_indices

    env_ids = env_ids.to(env.device, dtype=torch.long)
    entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "geom", False).to(
        dtype=torch.long
    )
    env_grid, geom_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
    return (env_grid, geom_grid)


@requires_model_fields("geom_friction")
def randomize_ground_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    terrain_cfg: SceneEntityCfg,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    env_grid, geom_grid = _resolve_geom_grid(env, asset, asset_cfg, env_ids)
    sampled = sample_uniform(
        friction_range[0], friction_range[1], (len(env_ids), 1), device=env.device
    )
    env.sim.model.geom_friction[env_grid, geom_grid, 0] = sampled
    _press_terrain_friction(env, terrain_cfg, env_ids)


@requires_model_fields("geom_friction")
def randomize_wheel_friction_multiplier(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    terrain_cfg: SceneEntityCfg,
    from_current_base: bool,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    env_grid, geom_grid = _resolve_geom_grid(env, asset, asset_cfg, env_ids)
    if from_current_base:
        base = env.sim.model.geom_friction[env_grid, geom_grid, 0].clone()
    else:
        base = env.sim.get_default_field("geom_friction")[geom_grid, 0]
    scale = sample_uniform(
        scale_range[0], scale_range[1], (len(env_ids), 1), device=env.device
    )
    env.sim.model.geom_friction[env_grid, geom_grid, 0] = base * scale
    _press_terrain_friction(env, terrain_cfg, env_ids)


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const_0)
def randomize_body_inertia_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    shared_random: bool = False,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    from mjlab.envs.mdp.dr._core import _get_entity_indices

    entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "body", False).to(
        dtype=torch.long
    )
    env_ids = env_ids.to(env.device, dtype=torch.long)
    env_grid, body_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
    defaults = env.sim.get_default_field("body_inertia")
    base = defaults[body_grid]
    if shared_random:
        scale = sample_uniform(
            ranges[0], ranges[1], (*base.shape[:-1], 1), device=env.device
        )
    else:
        scale = sample_uniform(ranges[0], ranges[1], base.shape, device=env.device)
    env.sim.model.body_inertia[env_grid, body_grid] = base * scale


def _actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    actuators: list[IdealPdActuator],
    kp_scale: torch.Tensor,
    kd_scale: torch.Tensor,
) -> None:
    for act_idx, act in enumerate(actuators):
        assert isinstance(act, IdealPdActuator), type(act)
        kp = act.default_stiffness[env_ids] * kp_scale[:, act_idx : act_idx + 1]
        kd = act.default_damping[env_ids] * kd_scale[:, act_idx : act_idx + 1]
        act.set_gains(env_ids, kp=kp, kd=kd)


def randomize_leg_pd_gains_strength(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    motor_strength_range: tuple[float, float],
    hip_strength_range: tuple[float, float] | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    n_envs = len(env_ids)
    n_act = len(actuators)
    kp_samples = sample_uniform(
        kp_range[0], kp_range[1], (n_envs, n_act), device=env.device
    )
    kd_samples = sample_uniform(
        kd_range[0], kd_range[1], (n_envs, n_act), device=env.device
    )
    strength = sample_uniform(
        motor_strength_range[0], motor_strength_range[1], (n_envs, 1), device=env.device
    )
    kp_scale = kp_samples * strength
    kd_scale = kd_samples * strength
    if hip_strength_range is not None:
        hip_mask = torch.tensor(
            [all(("hip" in name for name in act.target_names)) for act in actuators],
            device=env.device,
            dtype=torch.float32,
        )
        if bool(hip_mask.any()):
            hip_strength = sample_uniform(
                hip_strength_range[0],
                hip_strength_range[1],
                (n_envs, n_act),
                device=env.device,
            )
            factor = 1.0 + hip_mask * (hip_strength - 1.0)
            kp_scale = kp_scale * factor
            kd_scale = kd_scale * factor
    _actuator_gains(env, env_ids, actuators, kp_scale, kd_scale)


def randomize_wheel_motor_strength(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    n_envs = len(env_ids)
    n_act = len(actuators)
    strength = sample_uniform(
        strength_range[0], strength_range[1], (n_envs, n_act), device=env.device
    )
    _actuator_gains(env, env_ids, actuators, strength, strength)


def reset_joints_by_default_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    soft_limits = asset.data.soft_joint_pos_limits
    assert soft_limits is not None
    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    scale = sample_uniform(scale_range[0], scale_range[1], joint_pos.shape, env.device)
    joint_pos = joint_pos * scale
    limits = soft_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        torch.zeros_like(joint_pos),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def randomize_wheel_radius_with_height(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    dr.geom_size(
        env,
        env_ids,
        ranges={0: scale_range},
        asset_cfg=asset_cfg,
        distribution="uniform",
        operation="scale",
        shared_random=True,
    )
    asset = env.scene[asset_cfg.name]
    env_grid, geom_grid = _resolve_geom_grid(env, asset, asset_cfg, env_ids)
    r_new = env.sim.model.geom_size[env_grid, geom_grid, 0]
    assert bool(torch.allclose(r_new, r_new[:, :1], rtol=0, atol=0)), r_new
    env_ids_long = env_ids.to(env.device, dtype=torch.long)
    pose = env.sim.data.qpos[
        env_ids_long[:, None], asset.indexing.free_joint_q_adr
    ].clone()
    pose[:, 2] += r_new[:, 0] - WOLF_WHEEL_RADIUS
    asset.write_root_link_pose_to_sim(pose, env_ids=env_ids_long)


@dataclass(kw_only=True)
class CalfBacklashPositionActionCfg(JointPositionActionCfg):
    backlash_enabled: bool = False
    width_range: tuple[float, float] = (0.005, 0.035)
    min_kp_scale: float = 0.12
    engage_start: float = 0.6
    leak: float = 0.02

    def build(self, env: ManagerBasedRlEnv) -> "CalfBacklashPositionAction":
        return CalfBacklashPositionAction(self, env)


class CalfBacklashPositionAction(JointPositionAction):
    def __init__(self, cfg: CalfBacklashPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._backlash_cfg = cfg
        self._calf_slots = [
            i for i, name in enumerate(self._target_names) if name.endswith("_calf")
        ]
        n_calf = len(self._calf_slots)
        device = self.device
        self._widths = torch.zeros(self.num_envs, n_calf, device=device)
        self._effective = torch.zeros_like(self._widths)
        self._last_targets = torch.zeros_like(self._widths)
        self._dirs = torch.zeros_like(self._widths)
        if n_calf == 0:
            raise ValueError("CalfBacklashPositionAction 需要 term 内含 calf 关节")
        default = self._entity.data.default_joint_pos
        assert default is not None
        calf_joint_ids = [self._target_ids[i] for i in self._calf_slots]
        init = default[:, calf_joint_ids]
        self._effective[:] = init
        self._last_targets[:] = init

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        cfg = self._backlash_cfg
        ids = slice(None) if env_ids is None else env_ids
        if cfg.backlash_enabled:
            self._widths[ids] = sample_uniform(
                cfg.width_range[0],
                cfg.width_range[1],
                self._widths[ids].shape,
                device=self.device,
            )
        else:
            self._widths[ids] = 0.0
        default = self._entity.data.default_joint_pos
        assert default is not None
        calf_joint_ids = torch.tensor(
            [self._target_ids[i] for i in self._calf_slots], device=self.device
        )
        init = (
            default[ids][:, calf_joint_ids]
            if not isinstance(ids, slice)
            else default[:, calf_joint_ids]
        )
        self._effective[ids] = init
        self._last_targets[ids] = init
        self._dirs[ids] = 0.0

    def apply_actions(self) -> None:
        cfg = self._backlash_cfg
        if not cfg.backlash_enabled:
            super().apply_actions()
            return
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._processed_actions - encoder_bias
        calf_targets = target[:, self._calf_slots]
        q = self._entity.data.joint_pos[:, self._target_ids][:, self._calf_slots]
        widths = self._widths.clamp(min=1e-06)
        if cfg.leak > 0.0:
            self._effective += cfg.leak * (calf_targets - self._effective)
        self._effective = torch.clamp(
            self._effective, min=calf_targets - widths, max=calf_targets + widths
        )
        gap_ratio = torch.clamp(
            torch.abs(calf_targets - self._effective) / widths, min=0.0, max=1.0
        )
        engage_span = max(1.0 - cfg.engage_start, 1e-06)
        engage = torch.clamp(
            (gap_ratio - cfg.engage_start) / engage_span, min=0.0, max=1.0
        )
        kp_scale = cfg.min_kp_scale + (1.0 - cfg.min_kp_scale) * engage
        delta_sign = torch.sign(calf_targets - self._last_targets)
        self._last_targets = calf_targets.clone()
        self._dirs = torch.where(delta_sign != 0.0, delta_sign, self._dirs)
        effective_target = q + kp_scale * (self._effective - q)
        target[:, self._calf_slots] = effective_target
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class ScaledBiasedWheelVelocityActionCfg(JointVelocityActionCfg):
    vel_ref_scale_enabled: bool = False
    vel_ref_scale_range: tuple[float, float] = (0.9, 1.1)
    vel_ref_bias_enabled: bool = False
    vel_ref_bias_range: tuple[float, float] = (0.0, 0.0)

    def build(self, env: ManagerBasedRlEnv) -> "ScaledBiasedWheelVelocityAction":
        return ScaledBiasedWheelVelocityAction(self, env)


class ScaledBiasedWheelVelocityAction(JointVelocityAction):
    def __init__(self, cfg: ScaledBiasedWheelVelocityActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._wheel_cfg = cfg
        self._scales = torch.ones(self.num_envs, self.action_dim, device=self.device)
        self._biases = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def _resample(self, env_ids: torch.Tensor | slice) -> None:
        cfg = self._wheel_cfg
        shape = self._scales[env_ids].shape
        if cfg.vel_ref_scale_enabled:
            self._scales[env_ids] = sample_uniform(
                cfg.vel_ref_scale_range[0],
                cfg.vel_ref_scale_range[1],
                shape,
                device=self.device,
            )
        else:
            self._scales[env_ids] = 1.0
        if cfg.vel_ref_bias_enabled:
            self._biases[env_ids] = sample_uniform(
                cfg.vel_ref_bias_range[0],
                cfg.vel_ref_bias_range[1],
                shape,
                device=self.device,
            )
        else:
            self._biases[env_ids] = 0.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._resample(slice(None) if env_ids is None else env_ids)

    def apply_actions(self) -> None:
        cfg = self._wheel_cfg
        if not (cfg.vel_ref_scale_enabled or cfg.vel_ref_bias_enabled):
            super().apply_actions()
            return
        target = self._processed_actions * self._scales + self._biases
        self._entity.set_joint_velocity_target(target, joint_ids=self._target_ids)


class BiasedSignedWheelVelocity:
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._bias_range: tuple[float, float] = cfg.params["bias_range"]
        self._enabled: bool = cfg.params["enabled"]
        self._env = env
        self._bias: torch.Tensor | None = None

    def _ensure_buffer(self, num_envs: int, n_wheels: int, device) -> None:
        if self._bias is None or self._bias.shape != (num_envs, n_wheels):
            self._bias = torch.zeros(num_envs, n_wheels, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._bias is None:
            self._ensure_buffer(self._env.num_envs, 4, self._env.device)
        ids = slice(None) if env_ids is None else env_ids
        if self._enabled:
            shape = self._bias[ids].shape
            self._bias[ids] = sample_uniform(
                self._bias_range[0],
                self._bias_range[1],
                shape,
                device=self._bias.device,
            )
        else:
            self._bias[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        bias_range: tuple[float, float],
        enabled: bool,
    ) -> torch.Tensor:
        signed = signed_wheel_velocity(env, asset_cfg)
        self._ensure_buffer(signed.shape[0], signed.shape[1], signed.device)
        return signed + self._bias
