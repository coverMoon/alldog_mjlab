"""HIM rollout storage：RSL-RL v5.4.2 ``RolloutStorage`` 的最小扩展。

只额外保存 HIM estimator 真正缺失的 successor target：

```text
next_estimator_input     [T, B, target_encoder_input_dim]
next_estimator_velocity  [T, B, velocity_dim]
```

current canonical history 不需要额外保存：它由 transition observation（``actor`` group）
在 update 时通过 ``canonical_history`` 得到。不复制 legacy ``HIMRolloutStorage`` 全文件，
也不重复保存 actor / critic tensor。
"""

from __future__ import annotations

from collections.abc import Generator

import torch

from rsl_rl.storage import RolloutStorage


class HIMTransition(RolloutStorage.Transition):
    """transition + HIM successor estimator target。"""

    def __init__(self) -> None:
        super().__init__()
        self.next_estimator_input: torch.Tensor | None = None
        self.next_estimator_velocity: torch.Tensor | None = None


class HIMBatch(RolloutStorage.Batch):
    """mini-batch + HIM successor estimator target。"""

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
    """``RolloutStorage`` + successor estimator target buffers。"""

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
            num_transitions_per_env, num_envs, next_estimator_velocity_dim, device=device
        )

    def add_transition(self, transition: HIMTransition) -> None:
        index = self.step
        super().add_transition(transition)
        self.next_estimator_input[index].copy_(transition.next_estimator_input)
        self.next_estimator_velocity[index].copy_(transition.next_estimator_velocity)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[HIMBatch, None, None]:
        """与父类相同顺序的 mini-batch 生成器，额外携带 successor estimator target。

        复制父类实现的原因：父类按内部 ``randperm`` 索引打乱后再 yield，子类无法从
        yield 出来的 ``Batch`` 反推同一 ``batch_idx``；要保证 estimator target 与
        PPO mini-batch 逐样本对齐，只能在同一处索引下构造 batch。
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
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
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
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
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    next_estimator_input=next_estimator_input[batch_idx],
                    next_estimator_velocity=next_estimator_velocity[batch_idx],
                )
