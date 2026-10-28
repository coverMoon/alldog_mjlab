"""Robot-independent HIM spec interfaces and computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class HIMSpec:
    single_frame_dim: int
    history_length: int
    command_dim: int
    velocity_dim: int
    action_dim: int
    latent_dim: int

    @property
    def history_dim(self) -> int:
        return self.history_length * self.single_frame_dim

    @property
    def target_encoder_input_dim(self) -> int:
        return self.single_frame_dim - self.command_dim + self.velocity_dim

    @property
    def actor_input_dim(self) -> int:
        return self.single_frame_dim + self.velocity_dim + self.latent_dim


@dataclass(frozen=True)
class HIMInterface:
    history_group: str
    velocity_group: str
    terminal_ids_key: str
    terminal_frame_key: str
    terminal_velocity_key: str


def him_spec_from_obs(
    obs: TensorDict,
    *,
    history_group: str,
    velocity_group: str,
    command_dim: int,
    latent_dim: int,
    action_dim: int,
) -> HIMSpec:
    history = obs[history_group]
    if history.ndim != 3:
        raise ValueError(
            f"HIM history group '{history_group}' must be [B, history_length, single_frame_dim], got shape {tuple(history.shape)}."
        )
    velocity = obs[velocity_group]
    if velocity.ndim != 2:
        raise ValueError(
            f"HIM velocity group '{velocity_group}' must be [B, velocity_dim], got shape {tuple(velocity.shape)}."
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
        raise ValueError(
            f"command_dim {spec.command_dim} must be < single_frame_dim {spec.single_frame_dim}."
        )
    return spec


def canonical_history(history: torch.Tensor) -> torch.Tensor:
    return history.flip(1).flatten(1)


def estimator_target_input(
    frame: torch.Tensor, velocity: torch.Tensor, spec: HIMSpec
) -> torch.Tensor:
    return torch.cat((frame[..., spec.command_dim :], velocity), dim=-1)
