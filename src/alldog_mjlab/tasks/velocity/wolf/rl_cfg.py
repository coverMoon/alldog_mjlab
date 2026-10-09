"""Wolf 任务的 RL / runner 配置（task local，不 import Black 的 rl_cfg）。

结构与 black/rl_cfg.py、black/him_rl_cfg.py 独立同构；数值来源全部是
WOLF_CONFIG。模型 / 算法 dataclass（``RslRlModelCfg`` 等）与算法层的公共实现
（`alldog_mjlab.algorithms.him.*` via class_name 字符串）属机器人无关公共基础，
继续正常复用。

保持与旧 Wolf 配置完全相同的字段值（class_name、超参数、restore mode 字段），
保证现有 checkpoint / CLI / export 路径兼容；只是声明归属到 Wolf。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from alldog_mjlab.tasks.velocity.wolf import him as wolf_him
from alldog_mjlab.tasks.velocity.wolf.wolf_config import WOLF_CONFIG


# ---------------------------------------------------------------------------
# 共用 helper（PPO 与 HIM 数值同源）
# ---------------------------------------------------------------------------


def wolf_ppo_algorithm_kwargs() -> dict:
    """PPO 算法超参数映射（普通 PPO 与 HIMPPO；数值来自 WOLF_CONFIG）。"""
    algorithm = WOLF_CONFIG.algorithm
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


def wolf_critic_model_cfg() -> RslRlModelCfg:
    """critic model cfg（普通 PPO 与 HIMPPO 共用）。"""
    policy = WOLF_CONFIG.policy
    return RslRlModelCfg(
        hidden_dims=policy.critic_hidden_dims,
        activation=policy.activation,
        obs_normalization=policy.critic_obs_normalization,
    )


def wolf_actor_distribution_cfg() -> dict:
    """actor Gaussian distribution cfg（普通 PPO 与 HIMPPO 共用）。"""
    policy = WOLF_CONFIG.policy
    return {
        "class_name": "GaussianDistribution",
        "init_std": policy.initial_std,
        "std_type": policy.std_type,
    }


@dataclass
class WolfRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Wolf PPO runner cfg：command curriculum checkpoint 恢复模式。

    ``command_curriculum_restore``：

    ```text
    auto  （默认）：同 stage resume -> full；跨 stage resume -> range；
                  旧 checkpoint（无 curriculum state） -> none + warning
    full  ：逐值恢复 range / EMA / streak / buffer（同 stage 精确续训）
    range ：只恢复 vx range，统计 fresh（跨 stage / 跨训练条件 continuation）
    none  ：不恢复，保持 config 初始范围
    ```
    """

    command_curriculum_restore: str = "auto"


# ---------------------------------------------------------------------------
# HIM 配置（Wolf 独立声明；字段值与旧 Wolf HIM cfg 一致）
# ---------------------------------------------------------------------------


@dataclass
class WolfHimRslRlModelCfg(RslRlModelCfg):
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
class WolfHimRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """HIMPPO algorithm cfg（PPO 超参数与普通 PPO 同源）。"""

    class_name: str = "alldog_mjlab.algorithms.him.ppo:HIMPPO"


@dataclass
class WolfHimRunnerParams:
    """HIM 算法与 task-side interface 的显式配置。

    group 名与 terminal extras key 来自 wolf/him.py 的冻结 contract
    （HIM 算法 interface；与 Black 同名但各 task env 各自持有）。
    """

    history_group: str
    velocity_group: str
    command_dim: int
    latent_dim: int
    terminal_ids_key: str
    terminal_frame_key: str
    terminal_velocity_key: str


def _default_him_params() -> WolfHimRunnerParams:
    """默认 HIM interface（group 名与 terminal extras key 来自 wolf/him.py）。"""
    return WolfHimRunnerParams(
        history_group=wolf_him.WOLF_HIM_ACTOR_GROUP,
        velocity_group=wolf_him.WOLF_HIM_VELOCITY_GROUP,
        command_dim=wolf_him.WOLF_HIM_COMMAND_DIM,
        latent_dim=WOLF_CONFIG.him.latent_dim,
        terminal_ids_key=wolf_him.WOLF_HIM_TERMINAL_IDS_KEY,
        terminal_frame_key=wolf_him.WOLF_HIM_TERMINAL_FRAME_KEY,
        terminal_velocity_key=wolf_him.WOLF_HIM_TERMINAL_VELOCITY_KEY,
    )


@dataclass
class WolfHimRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM runner cfg（actor/algorithm class 与 HIM interface 为 HIM 独有）。

    ``warm_start``：用 wolf-flat PPO checkpoint 初始化新 HIM run（初始化，不是
    resume）；与 ``resume`` 互斥。source checkpoint 由 ``load_run`` /
    ``load_checkpoint`` 定位。
    """

    actor: WolfHimRslRlModelCfg = field(default_factory=WolfHimRslRlModelCfg)
    algorithm: WolfHimRslRlPpoAlgorithmCfg = field(default_factory=WolfHimRslRlPpoAlgorithmCfg)
    him: WolfHimRunnerParams = field(default_factory=_default_him_params)
    warm_start: bool = False
    warm_start_supported: bool = True
    # 与 PPO 相同语义：auto 同 stage resume -> full，跨 stage resume（flat HIM→
    # rough HIM）-> range。PPO→HIM warm start 不走该字段：固定 mode='range'。
    command_curriculum_restore: str = "auto"


# ---------------------------------------------------------------------------
# stage builders
# ---------------------------------------------------------------------------


_STAGES = ("flat", "rough")


def wolf_ppo_runner_cfg(stage: Literal["flat", "rough"] = "flat") -> WolfRslRlOnPolicyRunnerCfg:
    """构造 Wolf PPO 的 runner cfg（stage = flat | rough；数值取自 WOLF_CONFIG）。"""
    if stage not in _STAGES:
        raise ValueError(f"unknown Wolf training stage: {stage!r}")
    policy = WOLF_CONFIG.policy
    runner = WOLF_CONFIG.runner
    stage_runner = getattr(runner, stage)
    return WolfRslRlOnPolicyRunnerCfg(
        command_curriculum_restore=runner.command_curriculum_restore,
        actor=RslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # Wolf actor 直接使用已固定 scale 的 53 维 observation，
            # 不做 running normalization（critic 保持 True）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=wolf_actor_distribution_cfg(),
        ),
        critic=wolf_critic_model_cfg(),
        algorithm=RslRlPpoAlgorithmCfg(**wolf_ppo_algorithm_kwargs()),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=stage_runner.run_name,
        load_run=stage_runner.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
    )


def wolf_him_runner_cfg(stage: Literal["flat", "rough"] = "flat") -> WolfHimRslRlOnPolicyRunnerCfg:
    """构造 Wolf HIM 的 runner cfg（与 ``wolf_ppo_runner_cfg`` 同 stage 语义）。

    PPO→HIM warm start 只支持 flat（正式训练路线：flat PPO → flat HIM →
    rough HIM full resume）；rough HIM 显式不支持 warm start，请求时 fail-loud。
    """
    if stage not in _STAGES:
        raise ValueError(f"unknown Wolf HIM training stage: {stage!r}")
    policy = WOLF_CONFIG.policy
    runner = WOLF_CONFIG.runner
    him = WOLF_CONFIG.him
    stage_runner = getattr(runner, f"{stage}_him")
    return WolfHimRslRlOnPolicyRunnerCfg(
        command_curriculum_restore=runner.command_curriculum_restore,
        actor=WolfHimRslRlModelCfg(
            hidden_dims=policy.actor_hidden_dims,
            activation=policy.activation,
            # HIM actor 不做 running normalization（HIMPolicy 显式不支持）。
            obs_normalization=policy.actor_obs_normalization,
            distribution_cfg=wolf_actor_distribution_cfg(),
            encoder_hidden_dims=him.encoder_hidden_dims,
            target_encoder_hidden_dims=him.target_encoder_hidden_dims,
            num_prototypes=him.num_prototypes,
            temperature=him.temperature,
            estimator_learning_rate=him.estimator_learning_rate,
            estimator_max_grad_norm=him.estimator_max_grad_norm,
        ),
        critic=wolf_critic_model_cfg(),
        algorithm=WolfHimRslRlPpoAlgorithmCfg(**wolf_ppo_algorithm_kwargs()),
        him=_default_him_params(),
        seed=runner.seed,
        experiment_name=runner.experiment_name,
        run_name=stage_runner.run_name,
        load_run=stage_runner.load_run,
        save_interval=runner.save_interval,
        num_steps_per_env=runner.num_steps_per_env,
        max_iterations=runner.max_iterations,
        # 正式训练路线 wolf-flat PPO → wolf-flat HIM（warm start）→ wolf-rough HIM
        # （full resume）；PPO→HIM warm start 只支持 flat。
        warm_start_supported=(stage == "flat"),
    )
