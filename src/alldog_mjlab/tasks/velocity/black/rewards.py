"""Black 专用的 velocity task reward functions。

MjLab v1.6.0 的 native tracking 把 v_z²（linear）与 ω_xy²（angular）并入同一个
exponential，与 Black 的 tracking contract 不等价，因此这两个 term 使用
task-local 实现。其余 reward 继续使用 MjLab native function。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
    与 legacy 的 body-frame root 角速度不是同一 contract，因此这里单独实现。
    """
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(torch.square(angular_velocity_xy), dim=1)


def orientation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚身体倾斜：``|g_x^b| + |g_y^b|``（body-frame projected gravity 的 L1）。

    legacy 的 terrain-adaptive pitch scaling 在 2026-07-03 reward baseline 中已关闭
    （``terrain_adaptive.enabled = False``，decay scale 恒为 1），因此这里就是
    roll / pitch 对称的纯 L1 惩罚，不含地形依赖，也不用 Euler 角。

    MjLab native ``upright`` 是 ``exp(-Σg_xy²/std²)`` 的正奖励，与 legacy 的
    L1 惩罚形式不同，无法用 weight / std 组合等价，所以单独实现。
    """
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return torch.sum(torch.abs(projected_gravity[:, :2]), dim=1)


def base_height_l1_flat(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 root 高度偏离期望值：``|root_link_pos_w.z - target_height|``。

    命名带 ``flat``：当前 task 的地面是 world z = 0 的 plane，因此 world z 就是
    离地高度，与 legacy ``_get_base_heights()`` 在 ``mesh_type == 'plane'`` 下的
    分支（直接返回 ``root_states[:, 2]``）一致，不需要地形采样。

    legacy rough terrain 的 ground-relative 高度语义属于 ``black-rough`` 阶段，
    本 task 不引入任何 height sensor / terrain 依赖，也不为此预留抽象。
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.abs(base_height - target_height)
