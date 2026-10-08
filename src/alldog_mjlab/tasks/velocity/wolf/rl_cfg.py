"""Wolf 任务的 RL / runner 配置（与 black/rl_cfg.py、black/him_rl_cfg.py 同构）。

数值来源全部是 WOLF_CONFIG；模型 / 算法 dataclass 复用 Black 已验证的通用类
（`CommandCurriculumRunnerCfg`、`HimRslRlModelCfg`、`HimRslRlPpoAlgorithmCfg`、
`HimRunnerParams`），不复制 PPO / HIM 超参数结构。
"""

from __future__ import annotations
from typing import Literal

from mjlab.rl import (
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)

from alldog_mjlab.tasks.velocity.black.curriculum_checkpoint import (
    VelocityCommandCurriculumRunner,  # noqa: F401  （runner_cls 注册复用）
)
from alldog_mjlab.tasks.velocity.black.him_rl_cfg import (
    HimRslRlModelCfg,
    HimRslRlOnPolicyRunnerCfg,
    HimRslRlPpoAlgorithmCfg,
    HimRunnerParams,
    _default_him_params,
)
from alldog_mjlab.tasks.velocity.black.rl_cfg import (
    BlackRslRlOnPolicyRunnerCfg,
    black_actor_distribution_cfg,
    black_critic_model_cfg,
    black_ppo_algorithm_kwargs,
)
from alldog_mjlab.tasks.velocity.wolf.wolf_config import WOLF_CONFIG


def wolf_ppo_runner_cfg(stage: Literal["flat"] = "flat") -> BlackRslRlOnPolicyRunnerCfg:
    """构造 Wolf PPO 的 runner cfg。

    网络分布 / critic / PPO 超参数沿用 Black 已验证配置结构，数值取自
    WOLF_CONFIG（当前与 Black 一致取值，但语义上是 Wolf 的独立入口）。
    """
    from alldog_mjlab.tasks.velocity.wolf.wolf_config import WOLF_CONFIG as _W
    del _W
    if stage != "flat":
        raise ValueError(f"unknown Wolf training stage: {stage!r}（本轮仅 flat）")
    policy = WOLF_CONFIG.policy
    runner = WOLF_CONFIG.runner
    return BlackRslRlOnPolicyRunnerCfg(
        command_curriculum_restore=runner.command_curriculum_restore,
        actor=RslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # Wolf actor 直接使用已固定 scale 的 53 维 observation，
            # 不做 running normalization（critic 保持 True）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=black_actor_distribution_cfg(WOLF_CONFIG),
        ),
        critic=black_critic_model_cfg(WOLF_CONFIG),
        algorithm=RslRlPpoAlgorithmCfg(**black_ppo_algorithm_kwargs(WOLF_CONFIG)),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=runner.flat.run_name,
        load_run=runner.flat.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
    )


def wolf_him_runner_cfg(stage: Literal["flat"] = "flat") -> HimRslRlOnPolicyRunnerCfg:
    """构造 Wolf HIM 的 runner cfg（与 ``wolf_ppo_runner_cfg`` 同 stage 语义）。"""
    if stage != "flat":
        raise ValueError(f"unknown Wolf HIM training stage: {stage!r}（本轮仅 flat）")
    policy = WOLF_CONFIG.policy
    runner = WOLF_CONFIG.runner
    him = WOLF_CONFIG.him
    return HimRslRlOnPolicyRunnerCfg(
        command_curriculum_restore=runner.command_curriculum_restore,
        actor=HimRslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # HIM actor 不做 running normalization（HIMPolicy 显式不支持）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=black_actor_distribution_cfg(WOLF_CONFIG),
            encoder_hidden_dims=him.encoder_hidden_dims,
            target_encoder_hidden_dims=him.target_encoder_hidden_dims,
            num_prototypes=him.num_prototypes,
            temperature=him.temperature,
            estimator_learning_rate=him.estimator_learning_rate,
            estimator_max_grad_norm=him.estimator_max_grad_norm,
        ),
        critic=black_critic_model_cfg(WOLF_CONFIG),
        algorithm=HimRslRlPpoAlgorithmCfg(**black_ppo_algorithm_kwargs(WOLF_CONFIG)),
        him=_default_him_params(),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=runner.flat_him.run_name,
        load_run=runner.flat_him.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
        # Wolf 训练路线：wolf-flat PPO → wolf-flat HIM（warm start）。
        # warm start 由 WolfHimOnPolicyRunner 处理（source = wolf-flat PPO）。
        warm_start=False,
        warm_start_supported=True,
    )
