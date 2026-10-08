"""Wolf flat 任务专属的 reward math。

tracking / 罚项公式全部复用 `black/rewards.py` 的机器人无关实现；这里只实现
按 16-D 显式 action contract 分组的 leg / wheel action rate（native
`action_rate_l2` 对全部 16 维求和，无法按腿 / 轮分组）。
"""

from __future__ import annotations

import torch

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_LEG_JOINT_NAMES,
    WOLF_LEG_ORDER,
)

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
