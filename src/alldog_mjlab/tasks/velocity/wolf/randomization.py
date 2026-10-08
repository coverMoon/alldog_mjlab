"""Wolf flat 任务 Domain Randomization 的自定义实现（MjLab v1.6.0 原生机制的补充）。

设计原则（见 .ai/MIGRATION.md §29）：

- 优先使用 MjLab v1.6.0 原生 ``dr.*`` / ``mdp.*``；本模块只实现原生机制
  无法直接表达的部分，全部相对 nominal（compile-time default）参数显式建立
  状态，禁止跨 episode 累计漂移。
- 轮地摩擦：MuJoCo 接触摩擦按 geom pair 取 max 结合（地面 geom 恒为默认
  1.0），因此「全局摩擦 + 轮摩擦乘子」必须直接以绝对值写轮 geom，而不是
  依赖 pair 组合；轮摩擦事件读取前序 ground_friction 事件写入的当前值再乘
  per-env 乘子（event dict 顺序 = 应用顺序）。
- 执行器：保留 IdealPdActuator 控制律，Kp/Kd/motor strength 组合缩放通过
  ``set_gains`` 相对 default 建立；轮子 Kp 恒为 0，任何缩放都不会引入
  position stiffness。
- calf backlash：在 action term 层实现旧版 ``play`` 模式状态机；输出目标
  ``q + kp_scale * (effective_target - q)`` 与旧版
  ``τ = Kp * kp_scale * (effective - q)`` 逐数学等价（每 physics substep
  更新一次状态，与旧版每 substep 调用 _compute_torques 的时序一致）。
- 轮 target scaling / bias：在 action term 层作用于 velocity target（per
  episode 常数，因此与 actuator 命令延迟的先后可交换，数值等价）。
- 轮观测 bias：只作用于 actor 可观测通道（独立 term 实例持有状态并在
  reset 重采样）；critic 观测构造时显式替换回无 bias 的真实 term。
"""

from __future__ import annotations

from dataclasses import dataclass

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
from alldog_mjlab.tasks.velocity.wolf.observations import signed_wheel_velocity


# ---------------------------------------------------------------------------
# Event：轮摩擦乘子（依赖前序 ground_friction 事件写入的绝对值）
# ---------------------------------------------------------------------------


def _resolve_env_ids(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    return env_ids.to(env.device, dtype=torch.int)

def _resolve_geom_grid(
    env: ManagerBasedRlEnv,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 asset_cfg 的 geom 选择构造 (env_grid, model_geom_grid) 索引（均 long）。

    geom id 经 ``asset.indexing.geom_ids`` 映射为 per-world 展开字段中的 model
    geom id（entity-local id 不能直接用作字段行索引）。
    """
    from mjlab.envs.mdp.dr._core import _get_entity_indices

    env_ids = env_ids.to(env.device, dtype=torch.long)
    entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "geom", False).to(
        dtype=torch.long
    )
    env_grid, geom_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
    return env_grid, geom_grid


@requires_model_fields("geom_friction")
def randomize_wheel_friction_multiplier(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    from_current_base: bool,
) -> None:
    """轮摩擦 = 本 episode 基础摩擦 × per-env 采样乘子。

    MuJoCo 接触摩擦按 geom pair 取 max 结合，乘法组合无法经 pair 表达，故直接
    改写轮 geom 的切向摩擦绝对值。

    ``from_current_base=True``（ground_friction 同开）：读前序 ground_friction
    事件写入的当前值作为基础摩擦（event dict 顺序保证）；
    ``from_current_base=False``（仅轮摩擦开启）：相对 compile-time default
    摩擦建立，避免跨 episode 陈旧值累计。
    """
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    env_grid, geom_grid = _resolve_geom_grid(env, asset, asset_cfg, env_ids)
    if from_current_base:
        base = env.sim.model.geom_friction[env_grid, geom_grid, 0].clone()
    else:
        base = env.sim.get_default_field("geom_friction")[geom_grid, 0]
    scale = sample_uniform(
        scale_range[0], scale_range[1], base.shape, device=env.device
    )
    env.sim.model.geom_friction[env_grid, geom_grid, 0] = base * scale


# ---------------------------------------------------------------------------
# Event：body inertia 相对 default 的乘法缩放（原生 dr 无 body_inertia 函数）
# ---------------------------------------------------------------------------


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const_0)
def randomize_body_inertia_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    shared_random: bool = False,
) -> None:
    """按 body 相对 nominal（default）惯量张量主矩做乘法缩放（三轴同乘）。

    与原生 ``dr.body_mass``（只改质量）配套使用，表达旧版
    wheel_inertia / link_inertia 的独立惯量缩放语义。
    """
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    from mjlab.envs.mdp.dr._core import _get_entity_indices

    entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "body", False).to(
        dtype=torch.long
    )
    env_ids = env_ids.to(env.device, dtype=torch.long)
    env_grid, body_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")

    defaults = env.sim.get_default_field("body_inertia")  # (nbody, 3) 编译期默认
    base = defaults[body_grid]  # (n_envs, n_bodies, 3)
    if shared_random:
        scale = sample_uniform(
            ranges[0], ranges[1], (*base.shape[:-1], 1), device=env.device
        )
    else:
        scale = sample_uniform(ranges[0], ranges[1], base.shape, device=env.device)
    env.sim.model.body_inertia[env_grid, body_grid] = base * scale


# ---------------------------------------------------------------------------
# Event：腿部 PD gains × motor strength 组合缩放（IdealPdActuator）
# ---------------------------------------------------------------------------


def _actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    actuators: list[IdealPdActuator],
    kp_scale: torch.Tensor,
    kd_scale: torch.Tensor,
) -> None:
    """把 (n_envs, n_actuators) 的组合缩放写到每 actuator 的 gains。

    kp_scale / kd_scale 是已经组合好全部因子的最终缩放（相对 default）。
    """
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
    """腿部（hip/thigh/calf）actuator 的 Kp / Kd / motor strength 组合缩放。

    语义对应 legacy BlackW：τ = s_global * (kp*kp_f*e − kd*kd_f*v)，hip 再乘
    per-hip strength s_hip。组合到 actuator 等效增益：
    ``kp_eff = default_kp * kp_f * s_global [* s_hip]``、kd 同理。
    所有因子相对 compile-time default 采样，per-episode 重采样不漂移。
    """
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    n_envs = len(env_ids)
    n_act = len(actuators)

    kp_samples = sample_uniform(kp_range[0], kp_range[1], (n_envs, n_act), device=env.device)
    kd_samples = sample_uniform(kd_range[0], kd_range[1], (n_envs, n_act), device=env.device)
    # 旧版 motor strength 是 per-env 标量（乘全部 12 腿关节力矩）。
    strength = sample_uniform(
        motor_strength_range[0], motor_strength_range[1], (n_envs, 1), device=env.device
    )
    kp_scale = kp_samples * strength
    kd_scale = kd_samples * strength

    if hip_strength_range is not None:
        hip_mask = torch.tensor(
            [all("hip" in name for name in act.target_names) for act in actuators],
            device=env.device,
            dtype=torch.float32,
        )
        if bool(hip_mask.any()):
            # per-env per-hip 独立采样（hip actuator 各自一列）。
            hip_strength = sample_uniform(
                hip_strength_range[0], hip_strength_range[1], (n_envs, n_act), device=env.device
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
    """轮 actuator（Velocity PD：Kp=0 / Kd=1）的电机强度缩放。

    只缩放 Kd（τ = Kd_eff * (vel_target − dq)）；Kp 恒为 default 0，缩放后
    仍为 0，不会意外引入 position stiffness。控制模式（velocity PD）不变。
    """
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    n_envs = len(env_ids)
    n_act = len(actuators)
    strength = sample_uniform(
        strength_range[0], strength_range[1], (n_envs, n_act), device=env.device
    )
    _actuator_gains(env, env_ids, actuators, strength, strength)


# ---------------------------------------------------------------------------
# Event：initial joint position（multiplicative，旧版 _reset_dofs 语义）
# ---------------------------------------------------------------------------


def reset_joints_by_default_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """关节位置 = default × U(scale_range)（multiplicative，旧版语义）。

    与 Wolf nominal 的 offset reset 不同：仅在 DR 开启时由 env_cfgs 用本函数
    替换腿关节 offset reset 事件。轮关节 default 为 0，乘法后仍为 0。
    位置 clamp 到 soft joint limits（与 native reset_joints_by_offset 一致）。
    """
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


# ---------------------------------------------------------------------------
# Event：轮碰撞半径缩放 + 初始高度补偿
# ---------------------------------------------------------------------------


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def randomize_wheel_radius_with_height(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    nominal_root_z: float,
) -> None:
    """轮碰撞圆柱半径相对 nominal 乘法缩放，并把 root z 抬高到新几何接触高度。

    - 半径用原生 ``dr.geom_size``（scale 相对 default，不累计；写后同步刷新
      geom_rbound / geom_aabb，broadphase 一致）；轮 body 有显式 inertial，
      几何变化不会暗中修改质量。
    - 初始高度补偿：root z = nominal_root_z + (r_new − r_nominal)，避免半径
      变化后 reset 轮地初始穿透（写 pos，不改 quat / 速度）。
    """
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
    r_new = env.sim.model.geom_size[env_grid, geom_grid, 0]  # (n, 4)
    # shared_random=True ⇒ 同 env 内 4 轮半径一致；取第一列做高度补偿。
    assert bool(torch.allclose(r_new, r_new[:, :1], rtol=0, atol=0)), r_new
    z = nominal_root_z + (r_new[:, :1] - WOLF_WHEEL_RADIUS)
    origin = env.scene.env_origins[env_ids]
    pose = torch.cat(
        [
            torch.cat(
                [origin[:, :2], z], dim=-1
            ),
            torch.zeros(len(env_ids), 4, device=env.device),
        ],
        dim=-1,
    )
    pose[:, 3] = 1.0  # 单位四元数（与 nominal reset 一致）
    asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Action term：calf backlash（旧版 play 模式）
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CalfBacklashPositionActionCfg(JointPositionActionCfg):
    """带 calf backlash 状态机的腿部位置 action cfg（term 维度 / 顺序不变）。"""

    backlash_enabled: bool = False
    width_range: tuple[float, float] = (0.005, 0.035)
    min_kp_scale: float = 0.12
    engage_start: float = 0.6
    leak: float = 0.02

    def build(self, env: ManagerBasedRlEnv) -> "CalfBacklashPositionAction":
        return CalfBacklashPositionAction(self, env)


class CalfBacklashPositionAction(JointPositionAction):
    """calf 通道的 backlash ``play`` 模式（legacy BlackW 语义）。

    每 physics substep（apply_actions 调用频率）更新一次状态机；非 calf 通道
    与父类行为完全一致。enabled=False 时本 term 与原生 JointPositionAction
    等价（无残余状态）。
    """

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
                cfg.width_range[0], cfg.width_range[1],
                self._widths[ids].shape, device=self.device,
            )
        else:
            self._widths[ids] = 0.0
        default = self._entity.data.default_joint_pos
        assert default is not None
        calf_joint_ids = torch.tensor(
            [self._target_ids[i] for i in self._calf_slots], device=self.device
        )
        init = default[ids][:, calf_joint_ids] if not isinstance(ids, slice) else default[:, calf_joint_ids]
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
        widths = self._widths.clamp(min=1e-6)
        if cfg.leak > 0.0:
            self._effective += cfg.leak * (calf_targets - self._effective)
        self._effective = torch.clamp(
            self._effective, min=calf_targets - widths, max=calf_targets + widths
        )
        gap_ratio = torch.clamp(
            torch.abs(calf_targets - self._effective) / widths, min=0.0, max=1.0
        )
        engage_span = max(1.0 - cfg.engage_start, 1e-6)
        engage = torch.clamp((gap_ratio - cfg.engage_start) / engage_span, min=0.0, max=1.0)
        kp_scale = cfg.min_kp_scale + (1.0 - cfg.min_kp_scale) * engage
        delta_sign = torch.sign(calf_targets - self._last_targets)
        self._last_targets = calf_targets.clone()
        self._dirs = torch.where(delta_sign != 0.0, delta_sign, self._dirs)

        # τ = Kp * kp_scale * (effective − q) 的精确等价表达（actuator Kp 不变）。
        effective_target = q + kp_scale * (self._effective - q)
        target[:, self._calf_slots] = effective_target
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


# ---------------------------------------------------------------------------
# Action term：轮 velocity target 的 scaling / bias
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ScaledBiasedWheelVelocityActionCfg(JointVelocityActionCfg):
    """带 per-episode target scaling / bias 的轮速度 action cfg。"""

    vel_ref_scale_enabled: bool = False
    vel_ref_scale_range: tuple[float, float] = (0.9, 1.1)
    vel_ref_bias_enabled: bool = False
    vel_ref_bias_range: tuple[float, float] = (0.0, 0.0)

    def build(self, env: ManagerBasedRlEnv) -> "ScaledBiasedWheelVelocityAction":
        return ScaledBiasedWheelVelocityAction(self, env)


class ScaledBiasedWheelVelocityAction(JointVelocityAction):
    """dq_target = processed * vel_ref_scale + vel_ref_bias（每 episode 重采样）。

    两个因子均为 per-episode 常数，与 actuator 命令延迟可交换（延迟的是
    常数缩放后的 target，与延迟 raw 后再缩放数值相同）。
    enabled 均为 False 时与原生 JointVelocityAction 等价。
    """

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
                cfg.vel_ref_scale_range[0], cfg.vel_ref_scale_range[1], shape, device=self.device
            )
        else:
            self._scales[env_ids] = 1.0
        if cfg.vel_ref_bias_enabled:
            self._biases[env_ids] = sample_uniform(
                cfg.vel_ref_bias_range[0], cfg.vel_ref_bias_range[1], shape, device=self.device
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


# ---------------------------------------------------------------------------
# Observation term：带 per-episode bias 的 signed wheel velocity（仅 actor）
# ---------------------------------------------------------------------------


class BiasedSignedWheelVelocity:
    """signed wheel velocity + per-episode 观测偏置（只影响策略可观测通道）。

    状态为本 term 实例私有，reset 时重采样；真实 joint_vel / reward / PD
    feedback / critic 观测均不受影响（critic 构造时显式使用无 bias 的
    ``signed_wheel_velocity``）。
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._bias_range: tuple[float, float] = cfg.params["bias_range"]
        self._enabled: bool = cfg.params["enabled"]
        self._env = env
        self._bias: torch.Tensor | None = None  # 首次 __call__ / reset 时按轮数初始化
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
                self._bias_range[0], self._bias_range[1], shape, device=self._bias.device
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
        """签名与 cfg.params 一致（manager 以关键字传参）；状态在 __init__ / reset。"""
        signed = signed_wheel_velocity(env, asset_cfg)
        self._ensure_buffer(signed.shape[0], signed.shape[1], signed.device)
        return signed + self._bias
