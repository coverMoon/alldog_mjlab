"""Robot-independent HIM ppo interfaces and computation."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from .policy import HIMPolicy
from .spec import (
    HIMInterface,
    HIMSpec,
    canonical_history,
    estimator_target_input,
    him_spec_from_obs,
)
from .storage import HIMBatch, HIMRolloutStorage, HIMTransition


class HIMPPO(PPO):
    actor: HIMPolicy
    storage: HIMRolloutStorage

    def __init__(
        self,
        actor: HIMPolicy,
        critic: MLPModel,
        storage: HIMRolloutStorage,
        *,
        spec: HIMSpec,
        interface: HIMInterface,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        optimizer_name = kwargs.get("optimizer", "adam")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise NotImplementedError(
                "HIMPPO does not support RND / symmetry extensions."
            )
        super().__init__(actor, critic, storage, device=device, **kwargs)
        self.spec = spec
        self.interface = interface
        self.transition = HIMTransition()
        self.optimizer = resolve_optimizer(optimizer_name)(
            chain(self.actor.policy_parameters(), self.critic.parameters()),
            lr=self.learning_rate,
        )

    def process_env_step(
        self,
        obs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        next_input, next_velocity = self._successor_estimator_targets(obs, extras)
        self.transition.next_estimator_input = next_input
        self.transition.next_estimator_velocity = next_velocity
        super().process_env_step(obs, rewards, dones, extras)

    def _successor_estimator_targets(
        self, obs, extras: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = obs[self.interface.history_group]
        frame = history[..., -1, :]
        velocity = obs[self.interface.velocity_group]
        next_input = estimator_target_input(frame, velocity, self.spec)
        next_velocity = velocity
        terminal_ids = extras.get(self.interface.terminal_ids_key)
        if terminal_ids is not None:
            terminal_frame = extras[self.interface.terminal_frame_key]
            terminal_velocity = extras[self.interface.terminal_velocity_key]
            next_input = next_input.clone()
            next_velocity = next_velocity.clone()
            next_input[terminal_ids] = estimator_target_input(
                terminal_frame, terminal_velocity, self.spec
            )
            next_velocity[terminal_ids] = terminal_velocity
        return (next_input, next_velocity)

    def _update_estimator(self, batch: HIMBatch, lr: float) -> tuple[float, float]:
        history = canonical_history(batch.observations[self.interface.history_group])
        return self.actor.estimator.update(
            history, batch.next_estimator_input, batch.next_estimator_velocity, lr=lr
        )

    def _zero_estimator_grads(self) -> None:
        for param in self.actor.estimator.parameters():
            param.grad = None

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(self.actor.output_distribution_params)
            entropy = self.actor.output_entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-05, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(0.01, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            estimation_loss, swap_loss = self._update_estimator(
                batch, lr=self.learning_rate
            )
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            ratio = torch.exp(
                actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            )
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            self.optimizer.zero_grad()
            self._zero_estimator_grads()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.policy_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.storage.clear()
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimation": mean_estimation_loss,
            "swap": mean_swap_loss,
        }

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> "HIMPPO":
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["algorithm"].pop("share_cnn_encoders", None)
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        him_cfg = cfg["him"]
        interface = HIMInterface(
            history_group=him_cfg["history_group"],
            velocity_group=him_cfg["velocity_group"],
            terminal_ids_key=him_cfg["terminal_ids_key"],
            terminal_frame_key=him_cfg["terminal_frame_key"],
            terminal_velocity_key=him_cfg["terminal_velocity_key"],
        )
        spec = him_spec_from_obs(
            obs,
            history_group=interface.history_group,
            velocity_group=interface.velocity_group,
            command_dim=him_cfg["command_dim"],
            latent_dim=him_cfg["latent_dim"],
            action_dim=env.num_actions,
        )
        cfg["obs_groups"] = resolve_obs_groups(
            obs, cfg["obs_groups"], ["actor", "critic"]
        )
        actor: HIMPolicy = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            spec=spec,
            history_group=interface.history_group,
            **cfg["actor"],
        ).to(device)
        print(f"HIM Actor Model: {actor}")
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")
        storage = HIMRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            next_estimator_input_dim=spec.target_encoder_input_dim,
            next_estimator_velocity_dim=spec.velocity_dim,
        )
        alg: HIMPPO = alg_class(
            actor,
            critic,
            storage,
            spec=spec,
            interface=interface,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["estimator_optimizer_state_dict"] = (
            self.actor.estimator.optimizer.state_dict()
        )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_estimator_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        if load_estimator_optimizer and "estimator_optimizer_state_dict" in loaded_dict:
            self.actor.estimator.optimizer.load_state_dict(
                loaded_dict["estimator_optimizer_state_dict"]
            )
        return load_iteration
