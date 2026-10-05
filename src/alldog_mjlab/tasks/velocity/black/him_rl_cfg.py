"""RL / runner configuration for Black HIM tasks。

结构（与普通 Black PPO 共用同一份 ``BLACK_CONFIG``）：

```text
BLACK_CONFIG
   ├── policy / algorithm / runner        ← black_ppo_runner_cfg 与 black_him_runner_cfg 共用
   └── him                                ← 仅 HIM estimator / latent 参数
```

本文件只声明 HIM 独有的 class / 结构与 estimator 超参数，不复制 PPO 数值，
也不注册 task。``asdict(HimRslRlOnPolicyRunnerCfg)`` 可直接传给
``MjlabOnPolicyRunner``（与 MjLab train CLI 的路径一致）。
"""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from alldog_mjlab.tasks.velocity.black import him as black_him
from alldog_mjlab.tasks.velocity.black.black_config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.black.rl_cfg import (
    black_actor_distribution_cfg,
    black_critic_model_cfg,
    black_ppo_algorithm_kwargs,
)


@dataclass
class HimRslRlModelCfg(RslRlModelCfg):
    """HIM actor model cfg（``RslRlModelCfg`` + HIM estimator 结构参数）。"""

    class_name: str = "alldog_mjlab.algorithms.him.policy:HIMPolicy"
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    target_encoder_hidden_dims: tuple[int, ...] = (128, 64)
    num_prototypes: int = 32
    temperature: float = 3.0
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0
    estimator_optimizer: str = "adam"


@dataclass
class HimRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """HIMPPO algorithm cfg（PPO 超参数与普通 PPO 同源）。"""

    class_name: str = "alldog_mjlab.algorithms.him.ppo:HIMPPO"


@dataclass
class HimRunnerParams:
    """HIM 算法与 task-side interface 的显式配置（Unit 1 / Unit 2 contract）。"""

    history_group: str
    velocity_group: str
    command_dim: int
    latent_dim: int
    terminal_ids_key: str
    terminal_frame_key: str
    terminal_velocity_key: str


def _default_him_params() -> HimRunnerParams:
    """默认 HIM interface（group 名与 terminal extras key 来自 Unit 1 冻结 contract）。"""
    return HimRunnerParams(
        history_group=black_him.BLACK_HIM_ACTOR_GROUP,
        velocity_group=black_him.BLACK_HIM_VELOCITY_GROUP,
        command_dim=black_him.BLACK_HIM_COMMAND_DIM,
        latent_dim=BLACK_CONFIG.him.latent_dim,
        terminal_ids_key=black_him.BLACK_HIM_TERMINAL_IDS_KEY,
        terminal_frame_key=black_him.BLACK_HIM_TERMINAL_FRAME_KEY,
        terminal_velocity_key=black_him.BLACK_HIM_TERMINAL_VELOCITY_KEY,
    )


@dataclass
class HimRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM runner cfg（actor/algorithm class 与 HIM interface 为 HIM 独有）。"""

    actor: HimRslRlModelCfg = field(default_factory=HimRslRlModelCfg)
    algorithm: HimRslRlPpoAlgorithmCfg = field(default_factory=HimRslRlPpoAlgorithmCfg)
    him: HimRunnerParams = field(default_factory=_default_him_params)


def black_him_runner_cfg(stage: Literal["flat", "rough"]) -> HimRslRlOnPolicyRunnerCfg:
    """构造 Black HIM 的 runner cfg（与 ``black_ppo_runner_cfg`` 同 stage 语义）。"""
    policy = BLACK_CONFIG.policy
    runner = BLACK_CONFIG.runner
    him = BLACK_CONFIG.him
    if stage not in ("flat", "rough"):
        raise ValueError(f"unknown Black HIM training stage: {stage}")
    stage_runner = getattr(runner, f"{stage}_him")
    return HimRslRlOnPolicyRunnerCfg(
        actor=HimRslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # HIM actor 不做 running normalization（HIMPolicy 显式不支持）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=black_actor_distribution_cfg(),
            encoder_hidden_dims=him.encoder_hidden_dims,
            target_encoder_hidden_dims=him.target_encoder_hidden_dims,
            num_prototypes=him.num_prototypes,
            temperature=him.temperature,
            estimator_learning_rate=him.estimator_learning_rate,
            estimator_max_grad_norm=him.estimator_max_grad_norm,
        ),
        critic=black_critic_model_cfg(),
        algorithm=HimRslRlPpoAlgorithmCfg(**black_ppo_algorithm_kwargs()),
        him=_default_him_params(),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=stage_runner.run_name,
        load_run=stage_runner.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
    )
