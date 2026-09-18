"""Black 专用的 velocity task reward math。

只有 MjLab v1.6.0 native 没有严格等价 term 的公式才在这里实现；native 严格等价的
项（orientation L2、action rate、action 二阶差分）直接使用 mjlab 自带函数。

当前 task-local：
  track_linear_velocity_xy     HIMLoco tracking（sigma 作 denominator，非 sigma²）
  track_angular_velocity_z     HIMLoco yaw tracking（同上）
  vertical_linear_velocity_l2  body-frame root v_z²
  angular_velocity_xy_l2       body-frame root ω_x² + ω_y²（native 读 world frame）
  base_height_l2_flat          flat task 的 world-z root 高度 L2
  base_height_l2_terrain       rough task 的 local terrain-relative root 高度 L2
  joint_power_l1               |q̇| · |τ|（native 无 joint power）
  dof_acc_l2                   control step 关节速度有限差分（native joint_acc_l2 用 qacc）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# rough base-height footprint：legacy BlackEnv 的 99 点 footprint（0.60 x 0.36 m）在
# native terrain_scan（0.1 m 网格）上的最近近似，即中央 x ∈ [-0.3, 0.3]、
# y ∈ [-0.2, 0.2] 的 7 x 5 = 35 条 ray。命名常量代替手写 indices。
_BASE_HEIGHT_FOOTPRINT_X = 0.3
_BASE_HEIGHT_FOOTPRINT_Y = 0.2
_BASE_HEIGHT_FOOTPRINT_NUM_RAYS = 35
# float32 网格值（如 0.30000001192）略大于十进制边界，用 tolerance 判定归属。
_BASE_HEIGHT_FOOTPRINT_TOLERANCE = 1e-4


def track_linear_velocity_xy(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """只跟踪 commanded planar 线速度的指数奖励。

    ``reward = exp(-Σ(v_cmd_xy - v_xy)² / sigma)``，body-frame root 线速度，
    竖直分量 v_z 不参与。``sigma`` 即 legacy ``tracking_sigma``。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return torch.exp(-error / sigma)


def track_angular_velocity_z(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """只跟踪 commanded yaw 角速度的指数奖励。

    ``reward = exp(-(w_cmd_z - w_z)² / sigma)``，body-frame root 角速度，
    roll / pitch 分量 ω_x、ω_y 不参与。``sigma`` 即 legacy ``tracking_sigma``。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-error / sigma)


def vertical_linear_velocity_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 body-frame root 竖直线速度：``v_z²``。

    black-flat 采用 legacy 的 terrain-level-0 分支（仅 ``v_z²``，无地形系数）；
    legacy ``terrain_levels > 0 → ×0.1`` 属于 black-rough 阶段，不在本 task 引入。
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def angular_velocity_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 body-frame root 的 roll / pitch 角速度：``ω_x² + ω_y²``。

    MjLab native ``body_angular_velocity_penalty`` 读的是 world-frame body 角速度，
    与 body-frame root 角速度不是同一 contract，因此这里单独实现。
    """
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(torch.square(angular_velocity_xy), dim=1)


def base_height_l2_flat(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 root 高度偏离期望值：``(root_link_pos_w.z - target_height)²``。

    命名带 ``flat``：当前 task 的地面是 world z = 0 的 plane，因此 world z 就是
    离地高度，与 HIMLoco ``_get_base_heights()`` 在 ``mesh_type == 'plane'`` 下的
    分支（直接返回 ``root_states[:, 2]``）一致，不需要地形采样。

    rough terrain 的 ground-relative 高度语义属于 ``black-rough`` 阶段，
    本 task 不引入任何 height sensor / terrain 依赖。
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.square(base_height - target_height)


class base_height_l2_terrain(ManagerTermBase):
    """rough task 的 local terrain-relative root 高度 L2。

    native ``height_scan()`` 的 raw 输出已经是每条 ray 的局部离地高度
    （``trunk_z - terrain_hit_z``，offset 0），因此 base height 就是 footprint 内 ray
    的均值：``reward = (mean(raw_footprint) - target_height)²``。与
    ``base_height_l2_flat`` 的唯一差别是测量方式（world z vs local clearance），
    kernel / target / weight 均相同。

    footprint 由 native ``terrain_scan`` 的中央区域表达（见模块常量）。索引在
    ``__init__`` 中从 sensor pattern 的真实 offsets 推导（不手写 magic indices），
    并校验冻结边界。

    与 legacy 的 intentional difference：legacy 用 11 x 9 = 99 点 footprint
    （0.60 x 0.36 m、dx 0.06 / dy 0.045）加 heightfield 3-cell min 采样与 L1 kernel；
    本 round 用 35 条 native ray 的直接命中值，L2 kernel 不变（见 MIGRATION.md）。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        sensor_name: str = cfg.params["sensor_name"]
        pattern = env.scene[sensor_name].cfg.pattern
        offsets, _ = pattern.generate_rays(None, str(self.device))
        tolerance = _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        mask = (offsets[:, 0].abs() <= _BASE_HEIGHT_FOOTPRINT_X + tolerance) & (
            offsets[:, 1].abs() <= _BASE_HEIGHT_FOOTPRINT_Y + tolerance
        )
        self._indices = mask.nonzero(as_tuple=False).squeeze(-1)
        selected = offsets[self._indices]
        assert selected.shape[0] == _BASE_HEIGHT_FOOTPRINT_NUM_RAYS, selected.shape
        assert (
            abs(float(selected[:, 0].abs().max()) - _BASE_HEIGHT_FOOTPRINT_X)
            <= _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        ), float(selected[:, 0].abs().max())
        assert (
            abs(float(selected[:, 1].abs().max()) - _BASE_HEIGHT_FOOTPRINT_Y)
            <= _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        ), float(selected[:, 1].abs().max())

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float,
        sensor_name: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs  # footprint 索引已在 __init__ 中解析。
        raw_scan = envs_mdp.height_scan(env, sensor_name=sensor_name)
        base_height = raw_scan[:, self._indices].mean(dim=1)
        return torch.square(base_height - target_height)


def joint_power_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚关节机械功率：``Σ |q̇| · |τ|``。

    使用 ``data.qfrc_actuator``（映射到关节空间的执行器力矩，PD actuator 下即
    PD 律算出的关节力矩），与 legacy ``self.torques`` 语义一致；不用
    ``actuator_force``（actuation space，经传动比折算）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    applied_torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=1)


class dof_acc_l2(ManagerTermBase):
    """惩罚关节加速度：``Σ ((q̇_prev - q̇) / step_dt)²``。

    HIMLoco ``dof_acc`` 用的是 **control step 之间的关节速度有限差分**，而 MjLab
    native ``joint_acc_l2`` 读的是 MuJoCo 瞬时 ``qacc``，两者不等价，因此需要
    per-env 保存上一 control step 的关节速度。

    legacy 中 ``last_dof_vel`` 在 ``compute_reward()`` 之后才更新，因此 reward 看到
    的正是「上一 control step 的速度」；这里用同样的语义。episode reset 时把缓存
    重置为当前速度（首个 step 的差分为 0）。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        self._asset: Entity = env.scene[asset_cfg.name]
        self._joint_ids = asset_cfg.joint_ids
        self._last_joint_vel = self._asset.data.joint_vel[:, self._joint_ids].clone()

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self._last_joint_vel[env_ids] = self._asset.data.joint_vel[
            :, self._joint_ids
        ][env_ids]

    def __call__(self, env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
        del kwargs  # 参数已在 __init__ 中读取。
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        acceleration = (self._last_joint_vel - joint_vel) / env.step_dt
        self._last_joint_vel[:] = joint_vel
        return torch.sum(torch.square(acceleration), dim=1)
