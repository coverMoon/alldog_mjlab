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
