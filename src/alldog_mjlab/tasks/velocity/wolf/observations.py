"""Wolf task-local 的 observation math（MjLab native 没有等价 term 的部分）。

`base_ang_vel` / `projected_gravity` 走 MjLab v1.6 native 原生函数
（`envs_mdp.builtin_sensor` / `envs_mdp.projected_gravity_from_sensor`，见 §27.5）；
这里只实现带轮子 forward sign 的 signed wheel velocity。
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_ORDER,
    WOLF_WHEEL_FORWARD_SIGN,
)


def signed_wheel_velocity(
    env,  # ManagerBasedRlEnv
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """带 forward sign 的轮角速度（body-frame policy 语义）。

    ``output = wheel_joint_angular_velocity * forward_sign``：正值表示该轮沿机身
    +x 前进方向的角速度（符号契约来自 wolf_constants.WOLF_WHEEL_FORWARD_SIGN，
    已由 FK + 闭环滚动验证）。轮列顺序由 ``asset_cfg``（preserve_order=True）
    决定；在此显式校验名字顺序就是 policy 轮顺序（FL→FR→RL→RR），不靠
    MJCF natural order 巧合。
    """
    expected_wheel_joints: tuple[str, ...] = tuple(f"{leg}_foot" for leg in WOLF_LEG_ORDER)
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
