"""RL configuration for Black velocity tasks.

`black_ppo_runner_cfg` 与 `him_rl_cfg.black_him_runner_cfg` 共用同一份
`BLACK_CONFIG` 数值来源；这里提供两个共享 helper，供 PPO / HIMPPO 两条路径复用，
避免 HIM 复制一套 PPO 超参数。
"""

from typing import Literal

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from .black_config import BLACK_CONFIG


def black_ppo_algorithm_kwargs() -> dict:
    """PPO 算法超参数映射（普通 PPO 与 HIMPPO 唯一来源：BLACK_CONFIG.algorithm）。"""
    algorithm = BLACK_CONFIG.algorithm
    return {
        "value_loss_coef": algorithm.value_loss_coef,
        "use_clipped_value_loss": algorithm.use_clipped_value_loss,
        "clip_param": algorithm.clip_param,
        "entropy_coef": algorithm.entropy_coef,
        "num_learning_epochs": algorithm.num_learning_epochs,
        "num_mini_batches": algorithm.num_mini_batches,
        "learning_rate": algorithm.learning_rate,
        "schedule": algorithm.schedule,
        "gamma": algorithm.gamma,
        "lam": algorithm.lam,
        "desired_kl": algorithm.desired_kl,
        "max_grad_norm": algorithm.max_grad_norm,
    }


def black_critic_model_cfg() -> RslRlModelCfg:
    """critic model cfg（普通 PPO 与 HIMPPO 共用）。"""
    policy = BLACK_CONFIG.policy
    return RslRlModelCfg(
        hidden_dims=policy.critic_hidden_dims,
        activation=policy.activation,
        obs_normalization=policy.critic_obs_normalization,
    )


def black_actor_distribution_cfg() -> dict:
    """actor Gaussian distribution cfg（普通 PPO 与 HIMPPO 共用）。"""
    policy = BLACK_CONFIG.policy
    return {
        "class_name": "GaussianDistribution",
        "init_std": policy.initial_std,
        "std_type": policy.std_type,
    }


def black_ppo_runner_cfg(stage: Literal["flat", "rough"]) -> RslRlOnPolicyRunnerCfg:
    policy = BLACK_CONFIG.policy
    runner = BLACK_CONFIG.runner
    if stage not in ("flat", "rough"):
        raise ValueError(f"unknown Black training stage: {stage}")
    stage_runner = getattr(runner, stage)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # Black policy 直接使用已固定 scale 的 45 维 observation，
            # 不做 running normalization（critic 保持 True）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=black_actor_distribution_cfg(),
        ),
        critic=black_critic_model_cfg(),
        algorithm=RslRlPpoAlgorithmCfg(**black_ppo_algorithm_kwargs()),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=stage_runner.run_name,
        load_run=stage_runner.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
    )
