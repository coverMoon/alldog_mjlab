"""Wolf 任务的 reward 函数入口（task local，不 import `black.rewards`）。

所有 Wolf reward term 的 func 都定义在本文件，由 env_cfgs.py 统一注册
（wolf_config.py 管参数 / rewards.py 管函数 / env_cfgs.py 管装配）。

分组（按 env_cfgs 注册顺序）：

1. 速度跟踪（分轴）：``track_linear_velocity_x`` / ``track_linear_velocity_y``
   / ``track_angular_velocity_z``（指数核，body-frame）；
2. 罚项与姿态：``vertical_linear_velocity_l2`` / ``angular_velocity_xy_l2`` /
   ``orientation_l1``（body-frame 项目均单独实现，不用 native world-frame 版）；
3. base_height：``base_height_l2_flat``（world-z，flat）/
   ``base_height_l2_terrain``（terrain-relative footprint 均值，rough；
   Black rough 同算法：native terrain_scan 中央 7 x 5 = 35 rays clearance）；
4. action rate：``leg_action_rate_l2`` / ``wheel_action_rate_l2``（按 16-D
   显式 action contract 分组，native `action_rate_l2` 对全部 16 维求和无法分组）；
5. 姿态（BlackW 经验迁移）：``hip_default_l1`` / ``stand_still_leg_l1`` /
   ``run_still_leg_l1``；
6. native 薄包装（不复制 MjLab 数学）：``dof_pos_limits`` /
   ``leg_torques_l2``（直接转发 ``mjlab.envs.mdp`` 原生实现）；
7. rough 专属：``wheel_force_lift``（轮水平接触力二值门控 × 向上竖直速度，仅
   wolf-rough / wolf-rough-him 注册）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_JOINT_NAMES,
    WOLF_LEG_ORDER,
    WOLF_WHEEL_COLLISION_GEOM_NAMES,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# rough base-height footprint：Black rough 已验证的中央 7 x 5 = 35 条 ray
# （x ∈ [-0.3, 0.3]、y ∈ [-0.2, 0.2]）；边界值来自 WOLF_CONFIG.terrain 的
# footprint 参数（Wolf 层配置，数值当前与 Black 相同）。
_FOOTPRINT_TOLERANCE = 1e-4


def track_linear_velocity_x(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """只跟踪 commanded X 线速度的指数奖励（单轴）。"""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    error = torch.square(command[:, 0] - actual[:, 0])
    return torch.exp(-error / sigma)


def track_linear_velocity_y(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """只跟踪 commanded Y 线速度的指数奖励（单轴）。"""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    error = torch.square(command[:, 1] - actual[:, 1])
    return torch.exp(-error / sigma)
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

    ``reward = exp(-(w_cmd_z - w_z)² / sigma)``，body-frame root 角速度。
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
    """惩罚 body-frame root 竖直线速度：``v_z²``。"""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def angular_velocity_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 body-frame root 的 roll / pitch 角速度：``ω_x² + ω_y²``。

    MjLab native ``body_angular_velocity_penalty`` 读 world-frame body 角速度，
    与 body-frame root 角速度不是同一 contract，因此单独实现。
    """
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(torch.square(angular_velocity_xy), dim=1)


def base_height_l2_flat(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 root 高度偏离期望值：``(root_link_pos_w.z - target_height)²``（flat）。

    flat 任务的地面是 world z = 0 的 plane，world z 即离地高度，不需要地形采样。
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.square(base_height - target_height)


class base_height_l2_terrain(ManagerTermBase):
    """rough task 的 local terrain-relative root 高度 L2。

    native ``height_scan()`` 的 raw 输出是每条 ray 的局部离地高度
    （frame z - terrain hit z），因此 base height 取 footprint 内 ray 的均值：
    ``reward = (mean(raw_footprint) - target_height)²``。与
    ``base_height_l2_flat`` 的唯一差别是测量方式（world z vs local clearance），
    kernel / target / weight 均相同。

    footprint = native ``terrain_scan`` 中央 7 x 5 = 35 条 ray（x 半宽 0.3 m /
    y 半长 0.2 m，来自 WOLF_CONFIG.terrain 的 footprint 参数）；索引在
    ``__init__`` 从 sensor pattern 的真实 offsets 推导，不手写 magic indices。
    """

    _NUM_RAYS = 35  # 7 x 5

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        from alldog_mjlab.tasks.velocity.wolf.wolf_config import WOLF_CONFIG

        super().__init__(env)
        sensor_name: str = cfg.params["sensor_name"]
        footprint_x = WOLF_CONFIG.terrain.base_height_footprint_x
        footprint_y = WOLF_CONFIG.terrain.base_height_footprint_y
        pattern = env.scene[sensor_name].cfg.pattern
        offsets, _ = pattern.generate_rays(None, str(self.device))
        mask = (offsets[:, 0].abs() <= footprint_x + _FOOTPRINT_TOLERANCE) & (
            offsets[:, 1].abs() <= footprint_y + _FOOTPRINT_TOLERANCE
        )
        self._indices = mask.nonzero(as_tuple=False).squeeze(-1)
        selected = offsets[self._indices]
        assert selected.shape[0] == self._NUM_RAYS, selected.shape
        assert (
            abs(float(selected[:, 0].abs().max()) - footprint_x)
            <= _FOOTPRINT_TOLERANCE
        ), float(selected[:, 0].abs().max())
        assert (
            abs(float(selected[:, 1].abs().max()) - footprint_y)
            <= _FOOTPRINT_TOLERANCE
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

# ---------------------------------------------------------------------------
# 16-D raw action 通道分组 contract。
#
# ActionManager 按 cfg.actions 插入顺序拼接 combined raw action。Wolf action
# term 顺序为每腿交错（joint_pos_X 3 维 + wheel_vel_X 1 维），即每腿 4 维 block：
#
#     [FL_hip FL_thigh FL_calf FL_wheel | FR ... | RL ... | RR ...]
#
# leg block 含 4 个 policy 关节位（hip/thigh/calf/foot），其中前 3 个是位置
# 通道、最后 1 个是轮通道。通道下标由上述插入顺序直接推导，并在测试中用真实
# env 的 action term 绑定交叉验证（tests/check_wolf_task.py）。
# ---------------------------------------------------------------------------

_BLOCK_DIM = len(WOLF_LEG_JOINT_NAMES["FL"])  # hip/thigh/calf/foot = 4
_LEG_CHANNELS = tuple(
    leg * _BLOCK_DIM + joint
    for leg in range(len(WOLF_LEG_ORDER))
    for joint in range(_BLOCK_DIM - 1)  # 前 3 个是腿位置通道
)
_WHEEL_CHANNELS = tuple(
    leg * _BLOCK_DIM + (_BLOCK_DIM - 1) for leg in range(len(WOLF_LEG_ORDER))
)
assert _LEG_CHANNELS == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14), _LEG_CHANNELS
assert _WHEEL_CHANNELS == (3, 7, 11, 15), _WHEEL_CHANNELS


def leg_action_rate_l2(env) -> torch.Tensor:
    """腿部 12 通道的 raw action 变化率 L2（native action_rate 的分组版）。"""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    diff = action[:, list(_LEG_CHANNELS)] - prev_action[:, list(_LEG_CHANNELS)]
    return torch.sum(torch.square(diff), dim=1)


def wheel_action_rate_l2(env) -> torch.Tensor:
    """轮部 4 通道的 raw action 变化率 L2（native action_rate 的分组版）。"""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    diff = action[:, list(_WHEEL_CHANNELS)] - prev_action[:, list(_WHEEL_CHANNELS)]
    return torch.sum(torch.square(diff), dim=1)


# ---------------------------------------------------------------------------
# Flat v1 姿态奖励（BlackW 经验迁移；仅 wolf-flat / wolf-flat-him 注册）。
#
# 关节均用显式名称选择（WOLF_LEG_JOINT_NAMES，不依赖 MJCF natural order）；
# default 位置读 entity 的 ``default_joint_pos``（EntityCfg INIT_STATE，与 reset
# 分布同源）。
# ---------------------------------------------------------------------------

# 四腿 hip 关节名（FL/FR/RL/RR 顺序，仅用于求和，顺序无语义）。
_HIP_JOINT_NAMES = tuple(
    WOLF_LEG_JOINT_NAMES[leg][0]  # 每腿第 1 个 = hip
    for leg in WOLF_LEG_ORDER
)
# 12 个腿部位置关节（hip/thigh/calf，不含轮关节 foot）。
_LEG_JOINT_NAMES = tuple(
    WOLF_LEG_JOINT_NAMES[leg][joint]
    for leg in WOLF_LEG_ORDER
    for joint in range(3)
)


def orientation_l1(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """机身姿态 L1：|projected_gravity_x| + |projected_gravity_y|。

    BlackW `_reward_orientation` 的无地形自适应版（指令相关衰减由 hip_default
    承担，姿态项本身恒定权重；flat / rough 四任务统一使用）。
    """
    asset: "Entity" = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1
    )


def hip_default_l1(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    y_ref: float,
    yaw_ref: float,
    y_scale: float,
    yaw_scale: float,
    min_scale: float,
) -> torch.Tensor:
    """四腿 hip 相对 default 的 L1 偏差 × 指令相关衰减。

    alpha = clamp(1 - y_scale*min(|cmd_y|/y_ref, 1) - yaw_scale*min(|cmd_yaw|/
    yaw_ref, 1), min_scale, 1)：侧向平移 / 原地转向机动时放宽 hip 回中要求
    （BlackW `_reward_hip_default` 同构，参数来自 WOLF_CONFIG.reward.posture）。
    """
    asset: "Entity" = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    hip_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    alpha = 1.0
    alpha = alpha - y_scale * torch.clamp(
        torch.abs(command[:, 1]) / max(y_ref, 1e-6), max=1.0
    )
    alpha = alpha - yaw_scale * torch.clamp(
        torch.abs(command[:, 2]) / max(yaw_ref, 1e-6), max=1.0
    )
    alpha = torch.clamp(alpha, min=min_scale, max=1.0)
    return hip_error * alpha


def stand_still_leg_l1(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    lin_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """静止门控的 12 个腿关节（hip/thigh/calf，无轮）回中 L1 惩罚。

    仅在 norm(cmd_xy) < lin_threshold 且 |cmd_yaw| < yaw_threshold 同时满足时
    开启（BlackW `_reward_stand_still` 同构；轮关节显式排除——轮静止时由
    wheel 通道自身表达，不在此重复约束）。
    """
    asset: "Entity" = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    command = env.command_manager.get_command(command_name)
    lin_stand = torch.norm(command[:, :2], dim=1) < lin_threshold
    yaw_stand = torch.abs(command[:, 2]) < yaw_threshold
    stand_mask = (lin_stand & yaw_stand).to(torch.float32)
    leg_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return leg_error * stand_mask


def run_still_leg_l1(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    x_threshold: float,
    y_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """直线行走门控的 12 个腿关节（hip/thigh/calf，无轮）回中 L1 惩罚。

    门控 = |cmd_x| > x_threshold 且 |cmd_y| < y_threshold 且 |cmd_yaw| <
    yaw_threshold（严格不等号；BlackW `_reward_run_still` 同构，阈值
    0.1 / 0.1 / 0.15 与 legacy 一致）：直线行走时要求腿回中，抑制行走中
    关节长期偏离 default。
    """
    asset: "Entity" = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    command = env.command_manager.get_command(command_name)
    active = (
        (torch.abs(command[:, 0]) > x_threshold)
        & (torch.abs(command[:, 1]) < y_threshold)
        & (torch.abs(command[:, 2]) < yaw_threshold)
    ).to(torch.float32)
    leg_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return leg_error * active


# ---------------------------------------------------------------------------
# MjLab native 薄包装（不复制数学实现；统一 Wolf reward 函数入口由本文件提供，
# 使 env_cfgs.py 的 reward 注册全部指向 task local 名称）。
# ---------------------------------------------------------------------------


def wheel_force_lift(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    min_horizontal_force: float,
) -> torch.Tensor:
    """轮子水平接触力门控 × 向上竖直速度的提升奖励（仅 rough 注册）。

    reward = Σ_{i∈{FL,FR,RL,RR}} 1[‖f_xy^i‖ ≥ min_horizontal_force] · max(v_z^i, 0)

    - ``f``：轮-terrain 接触力的世界系 net force（ContactSensor
      ``reduce="netforce"``：MjLab/mujoco_warp 对每个 contact 做
      ``contact_frame.T @ force`` 后求和，恒为世界系；符号约定为 primary
      →secondary，即轮压向地面方向为负 z，本公式只用水平模长，与符号无关）；
      因此 ``f`` 前两维的 norm 就是世界水平接触力幅值（若误当 contact-frame
      读，在障碍面（法向不竖直）上会得到错误量值，见 MIGRATION §30.6）；
    - 门控：水平接触力幅值达到 ``min_horizontal_force``（单位 N）时取 1，否则 0。
      门控把“轮正在水平顶住障碍”与“平地静态噪声 / 弱接触”区分开，且同一
      向上速度在 10 N 与 200 N 水平力下得分相同（不再随力大小线性放大；
      旧的 ``‖f_xy‖ · v_z`` 力乘速度语义已废弃）；
    - ``v_z``：轮刚体（Link4）世界系线速度竖直分量，clamp min=0（仅向上运动
      计入，向下 / 静止不产生奖励，自然限制奖励方向）；
    - 脱离接触后 net force 归零 → 门控 0，奖励自然归零。

    顺序 contract：接触力沿 ``sensor.primary_names``（不依赖 MJCF natural
    order），逐 call 显式断言为 FL/FR/RL/RR 的轮碰撞 geom 顺序；轮部速度索引
    由 ``asset_cfg``（``preserve_order=True`` 的 Link4 body 名）解析。
    """
    sensor = env.scene[sensor_name]
    assert list(sensor.primary_names) == list(WOLF_WHEEL_COLLISION_GEOM_NAMES), (
        f"wheel contact sensor primary order {sensor.primary_names} != "
        f"policy wheel order {WOLF_WHEEL_COLLISION_GEOM_NAMES}"
    )
    contact_force = sensor.data.force  # [B, P=4, 3]，netforce → world frame
    assert contact_force is not None
    horizontal_force = torch.norm(contact_force[..., :2], dim=-1)  # [B, 4]
    contact_gate = (horizontal_force >= min_horizontal_force).to(horizontal_force.dtype)
    asset: Entity = env.scene[asset_cfg.name]
    wheel_velocity_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]  # [B, 4, 3]
    upward_velocity = torch.clamp(wheel_velocity_w[..., 2], min=0.0)  # [B, 4]
    return torch.sum(contact_gate * upward_velocity, dim=1)


def dof_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """关节软限位越界惩罚（MjLab native `joint_pos_limits` 的 Wolf-local 转发）。"""
    return envs_mdp.joint_pos_limits(env, asset_cfg=asset_cfg)


def leg_torques_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """关节力矩 L2 惩罚（MjLab native `joint_torques_l2` 的 Wolf-local 转发）。

    选择器由 ``asset_cfg`` 控制：注册时显式传入 12 个腿部 actuator 名称
    （actuator 名 = 目标关节名），轮 actuator 不参与。
    """
    return envs_mdp.joint_torques_l2(env, asset_cfg=asset_cfg)
