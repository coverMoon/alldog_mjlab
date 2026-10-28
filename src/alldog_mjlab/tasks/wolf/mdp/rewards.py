"""Wolf custom reward kernels; return unweighted per-environment values."""

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
_BASE_HEIGHT_FOOTPRINT_X = 0.3
_BASE_HEIGHT_FOOTPRINT_Y = 0.2
_BASE_HEIGHT_FOOTPRINT_NUM_RAYS = 35
_BASE_HEIGHT_FOOTPRINT_TOLERANCE = 0.0001


def track_linear_velocity_xy(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
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
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-error / sigma)


def vertical_linear_velocity_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def angular_velocity_xy_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(torch.square(angular_velocity_xy), dim=1)


def base_height_l2_flat(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.square(base_height - target_height)


class base_height_l2_terrain(ManagerTermBase):
    """Squared height error using the mean clearance of the central 35 terrain rays."""

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
        del kwargs
        raw_scan = envs_mdp.height_scan(env, sensor_name=sensor_name)
        base_height = raw_scan[:, self._indices].mean(dim=1)
        return torch.square(base_height - target_height)


def hip_default(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return error.square().sum(dim=1)


def stand_still(
    env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    command = env.command_manager.get_command(command_name)
    return error.abs().sum(dim=1) * (command[:, :2].norm(dim=1) < command_threshold)


def run_still(
    env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    command = env.command_manager.get_command(command_name)
    return error.abs().sum(dim=1) * (command[:, :2].norm(dim=1) > command_threshold)


def collision(
    env, sensor_name: str, body_names: tuple[str, ...], force_threshold: float
) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if tuple(sensor.primary_names) != tuple(body_names):
        raise ValueError(
            f"collision body mapping mismatch: {sensor.primary_names} != {body_names}"
        )
    force = sensor.data.force
    assert force is not None and force.shape == (env.num_envs, len(body_names), 3), (
        force
    )
    return (force.norm(dim=-1) > force_threshold).float().sum(dim=1)


def feet_stumble(
    env, sensor_name: str, geom_names: tuple[str, ...], horizontal_ratio: float
) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if tuple(sensor.primary_names) != tuple(geom_names):
        raise ValueError(
            f"wheel contact mapping mismatch: {sensor.primary_names} != {geom_names}"
        )
    assert sensor.cfg.reduce == "netforce", (
        "feet_stumble requires world-frame net force"
    )
    force = sensor.data.force
    assert force is not None and force.shape == (env.num_envs, len(geom_names), 3), (
        force
    )
    return (
        (force[..., :2].norm(dim=-1) > horizontal_ratio * force[..., 2].abs())
        .any(dim=1)
        .float()
    )
