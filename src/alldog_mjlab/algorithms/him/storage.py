"""Robot-independent HIM storage interfaces and computation."""

from __future__ import annotations

from collections.abc import Generator

import torch
from rsl_rl.storage import RolloutStorage


class HIMTransition(RolloutStorage.Transition):
    def __init__(self) -> None:
        super().__init__()
        self.next_estimator_input: torch.Tensor | None = None
        self.next_estimator_velocity: torch.Tensor | None = None


class HIMBatch(RolloutStorage.Batch):
    def __init__(
        self,
        *args,
        next_estimator_input: torch.Tensor | None = None,
        next_estimator_velocity: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.next_estimator_input = next_estimator_input
        self.next_estimator_velocity = next_estimator_velocity


class HIMRolloutStorage(RolloutStorage):
    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        *,
        next_estimator_input_dim: int,
        next_estimator_velocity_dim: int,
    ) -> None:
        super().__init__(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
        )
        self.next_estimator_input = torch.zeros(
            num_transitions_per_env, num_envs, next_estimator_input_dim, device=device
        )
        self.next_estimator_velocity = torch.zeros(
            num_transitions_per_env,
            num_envs,
            next_estimator_velocity_dim,
            device=device,
        )

    def add_transition(self, transition: HIMTransition) -> None:
        index = self.step
        super().add_transition(transition)
        self.next_estimator_input[index].copy_(transition.next_estimator_input)
        self.next_estimator_velocity[index].copy_(transition.next_estimator_velocity)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[HIMBatch, None, None]:
        if self.training_type != "rl":
            raise ValueError(
                "This function is only available for reinforcement learning training."
            )
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(
            (p.flatten(0, 1) for p in self.distribution_params)
        )
        next_estimator_input = self.next_estimator_input.flatten(0, 1)
        next_estimator_velocity = self.next_estimator_velocity.flatten(0, 1)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                yield HIMBatch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(
                        (p[batch_idx] for p in old_distribution_params)
                    ),
                    next_estimator_input=next_estimator_input[batch_idx],
                    next_estimator_velocity=next_estimator_velocity[batch_idx],
                )
