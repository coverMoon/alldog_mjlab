"""Robot-independent HIM policy interfaces and computation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from .estimator import HIMEstimator
from .spec import HIMSpec, canonical_history


class _HimActorJit(nn.Module):
    def __init__(self, policy: HIMPolicy) -> None:
        super().__init__()
        spec = policy.spec
        self.frame_dim: int = spec.single_frame_dim
        self.velocity_dim: int = spec.velocity_dim
        self.history_dim: int = spec.history_dim
        self.encoder = copy.deepcopy(policy.estimator.encoder)
        self.mlp = copy.deepcopy(policy.mlp)
        if policy.distribution is not None:
            self.deterministic_output = (
                policy.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, canonical_history: torch.Tensor) -> torch.Tensor:
        if (
            canonical_history.dim() != 2
            or canonical_history.shape[-1] != self.history_dim
        ):
            raise ValueError(
                "HIM actor input must be float32 [B, %d] canonical history (frame-major, newest→oldest)."
                % self.history_dim
            )
        current = canonical_history[:, : self.frame_dim]
        encoded = self.encoder(canonical_history)
        velocity = encoded[:, : self.velocity_dim]
        latent = F.normalize(encoded[:, self.velocity_dim :], dim=-1, p=2.0)
        return self.deterministic_output(
            self.mlp(torch.cat((current, velocity, latent), dim=-1))
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class HIMPolicy(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        spec: HIMSpec,
        history_group: str,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        target_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        num_prototypes: int = 32,
        temperature: float = 3.0,
        estimator_learning_rate: float = 0.001,
        estimator_max_grad_norm: float = 10.0,
        estimator_optimizer: str = "adam",
    ) -> None:
        super().__init__()
        del obs, obs_groups, obs_set
        if obs_normalization:
            raise NotImplementedError(
                "HIM actor observation normalization is intentionally not implemented: HIM actor observation normalization is disabled by the environment contract."
            )
        if output_dim != spec.action_dim:
            raise ValueError(
                f"output_dim {output_dim} != spec.action_dim {spec.action_dim}."
            )
        self.spec = spec
        self.history_group = history_group
        self.estimator = HIMEstimator(
            spec,
            encoder_hidden_dims=encoder_hidden_dims,
            target_encoder_hidden_dims=target_encoder_hidden_dims,
            activation=activation,
            learning_rate=estimator_learning_rate,
            max_grad_norm=estimator_max_grad_norm,
            num_prototypes=num_prototypes,
            temperature=temperature,
            optimizer=estimator_optimizer,
        )
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(
                distribution_cfg.pop("class_name")
            )
            self.distribution: Distribution | None = dist_class(
                output_dim, **distribution_cfg
            )
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.mlp = MLP(
            spec.actor_input_dim,
            mlp_output_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        history = obs[self.history_group]
        if history.ndim != 3:
            raise ValueError(
                f"HIM history must be [B, H, F], got {tuple(history.shape)}."
            )
        current_frame = history[..., -1, :]
        with torch.no_grad():
            velocity, latent = self.estimator.encode(canonical_history(history))
        return torch.cat((current_frame, velocity, latent), dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        mlp_output = self.mlp(self.get_latent(obs, masks, hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def policy_parameters(self):
        yield from self.mlp.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def reset(
        self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    def as_jit(self) -> nn.Module:
        return _HimActorJit(self)

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)
