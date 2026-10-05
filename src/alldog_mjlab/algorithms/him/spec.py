"""HIM algorithm 的维度与接口 contract。

生产 HIM 算法不依赖机器人名或 task 名，也不出现 Black 固定维度（270 / 238 / 45 / 12）。
所有尺寸都通过 ``HIMSpec`` 从 task-side specification 或 env observation shape 得到，
环境侧 group 名与 terminal extras key 通过 ``HIMInterface`` 注入。

维度关系：

```text
history_dim              = history_length * single_frame_dim
target_encoder_input_dim = (single_frame_dim - command_dim) + velocity_dim
actor_input_dim          = single_frame_dim + velocity_dim + latent_dim
```
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class HIMSpec:
    """HIM 算法维度 contract（由 task-side spec / obs shape 填充）。"""

    single_frame_dim: int
    history_length: int
    command_dim: int
    velocity_dim: int
    action_dim: int
    latent_dim: int

    @property
    def history_dim(self) -> int:
        """source encoder 输入维度（history_length × single_frame_dim）。"""
        return self.history_length * self.single_frame_dim

    @property
    def target_encoder_input_dim(self) -> int:
        """target encoder 输入维度（当前 frame 去掉 command + velocity）。"""
        return (self.single_frame_dim - self.command_dim) + self.velocity_dim

    @property
    def actor_input_dim(self) -> int:
        """actor MLP 输入维度（current frame + estimated velocity + latent）。"""
        return self.single_frame_dim + self.velocity_dim + self.latent_dim


@dataclass(frozen=True)
class HIMInterface:
    """HIM 算法与 task-side observation / terminal contract 的接口名。"""

    history_group: str
    """提供 actor history（``[B, history_length, single_frame_dim]``）的 obs group。"""

    velocity_group: str
    """提供 scaled true base linear velocity（``[B, velocity_dim]``）的 obs group。"""

    terminal_ids_key: str
    """``env.extras`` 中 done env id 的 key。"""

    terminal_frame_key: str
    """``env.extras`` 中 terminal successor current actor frame 的 key。"""

    terminal_velocity_key: str
    """``env.extras`` 中 terminal successor estimator velocity 的 key。"""


def him_spec_from_obs(
    obs: TensorDict,
    *,
    history_group: str,
    velocity_group: str,
    command_dim: int,
    latent_dim: int,
    action_dim: int,
) -> HIMSpec:
    """从 env observation 的 runtime shape 推导 HIM 维度 contract。"""
    history = obs[history_group]
    if history.ndim != 3:
        raise ValueError(
            f"HIM history group '{history_group}' must be [B, history_length, single_frame_dim], "
            f"got shape {tuple(history.shape)}."
        )
    velocity = obs[velocity_group]
    if velocity.ndim != 2:
        raise ValueError(
            f"HIM velocity group '{velocity_group}' must be [B, velocity_dim], "
            f"got shape {tuple(velocity.shape)}."
        )
    spec = HIMSpec(
        single_frame_dim=history.shape[-1],
        history_length=history.shape[-2],
        command_dim=command_dim,
        velocity_dim=velocity.shape[-1],
        action_dim=action_dim,
        latent_dim=latent_dim,
    )
    if spec.command_dim >= spec.single_frame_dim:
        raise ValueError(f"command_dim {spec.command_dim} must be < single_frame_dim {spec.single_frame_dim}.")
    return spec


def canonical_history(history: torch.Tensor) -> torch.Tensor:
    """MjLab task-side history → official HIM canonical history。

    MjLab 原生顺序为 oldest → newest（``[B, H, F]``）；official HIM / legacy
    deployment 期望 frame-major 且 newest → oldest（``[B, H × F]``）。
    """
    return history.flip(1).flatten(1)


def estimator_target_input(
    frame: torch.Tensor, velocity: torch.Tensor, spec: HIMSpec
) -> torch.Tensor:
    """构造 target encoder 逻辑输入：frame 去掉 command + scaled true base lin vel。"""
    return torch.cat((frame[..., spec.command_dim :], velocity), dim=-1)
