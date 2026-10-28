"""Wolf observations interfaces."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_ORDER,
    WOLF_WHEEL_FORWARD_SIGN,
)


def signed_wheel_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    expected_wheel_joints: tuple[str, ...] = tuple(
        (f"{leg}_foot" for leg in WOLF_LEG_ORDER)
    )
    assert tuple(asset_cfg.joint_names) == expected_wheel_joints, (
        f"wheel observation order mismatch: {asset_cfg.joint_names} vs {expected_wheel_joints}"
    )
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    signs = torch.tensor(
        [WOLF_WHEEL_FORWARD_SIGN[leg] for leg in WOLF_LEG_ORDER],
        device=joint_vel.device,
        dtype=joint_vel.dtype,
    )
    return joint_vel * signs
