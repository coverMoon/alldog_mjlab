"""RL configuration for Black velocity tasks."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from .black_config import BLACK_CONFIG


def black_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    policy = BLACK_CONFIG.policy
    algorithm = BLACK_CONFIG.algorithm
    runner = BLACK_CONFIG.runner
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # Black policy 直接使用已固定 scale 的 45 维 observation，
            # 不做 running normalization（critic 保持 True）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": policy.initial_std,
                "std_type": policy.std_type,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=policy.critic_hidden_dims,
            activation=policy.activation,
            obs_normalization=policy.critic_obs_normalization,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=algorithm.value_loss_coef,
            use_clipped_value_loss=algorithm.use_clipped_value_loss,
            clip_param=algorithm.clip_param,
            entropy_coef=algorithm.entropy_coef,
            num_learning_epochs=algorithm.num_learning_epochs,
            num_mini_batches=algorithm.num_mini_batches,
            learning_rate=algorithm.learning_rate,
            schedule=algorithm.schedule,
            gamma=algorithm.gamma,
            lam=algorithm.lam,
            desired_kl=algorithm.desired_kl,
            max_grad_norm=algorithm.max_grad_norm,
        ),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
    )
