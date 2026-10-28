"""Black PPO/HIM network, optimizer and runner configuration."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .mdp import him

_STAGE_RUNS = {
    "flat": ("flat", ".*_flat$"),
    "rough": ("rough", ".*_rough$"),
    "flat_him": ("flat_him", ".*_flat_him$"),
    "rough_him": ("rough_him", ".*_rough_him$"),
}


@dataclass
class BlackRunnerCfg(RslRlOnPolicyRunnerCfg):
    checkpoint_path: str | None = None
    command_curriculum_restore: str = "range"


@dataclass
class BlackHimModelCfg(RslRlModelCfg):
    class_name: str = "alldog_mjlab.algorithms.him.policy:HIMPolicy"
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    target_encoder_hidden_dims: tuple[int, ...] = (128, 64)
    num_prototypes: int = 32
    temperature: float = 3.0
    estimator_learning_rate: float = 0.001
    estimator_max_grad_norm: float = 10.0
    estimator_optimizer: str = "adam"


@dataclass
class BlackHimAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "alldog_mjlab.algorithms.him.ppo:HIMPPO"


@dataclass
class HimRunnerParams:
    history_group: str = him.BLACK_HIM_ACTOR_GROUP
    velocity_group: str = him.BLACK_HIM_VELOCITY_GROUP
    command_dim: int = him.BLACK_HIM_COMMAND_DIM
    latent_dim: int = 16
    terminal_ids_key: str = him.BLACK_HIM_TERMINAL_IDS_KEY
    terminal_frame_key: str = him.BLACK_HIM_TERMINAL_FRAME_KEY
    terminal_velocity_key: str = him.BLACK_HIM_TERMINAL_VELOCITY_KEY


@dataclass
class BlackHimRunnerCfg(BlackRunnerCfg):
    actor: BlackHimModelCfg = field(default_factory=BlackHimModelCfg)
    algorithm: BlackHimAlgorithmCfg = field(default_factory=BlackHimAlgorithmCfg)
    him: HimRunnerParams = field(default_factory=HimRunnerParams)
    warm_start: bool = False
    warm_start_supported: bool = True


def make_rl_cfg(stage: Literal["flat", "rough"] = "flat", him_enabled: bool = False):
    if stage not in ("flat", "rough"):
        raise ValueError(f"unknown stage: {stage!r}")
    run_name, load_run = _STAGE_RUNS[stage + ("_him" if him_enabled else "")]
    actor_cls = BlackHimModelCfg if him_enabled else RslRlModelCfg
    algorithm_cls = BlackHimAlgorithmCfg if him_enabled else RslRlPpoAlgorithmCfg
    runner_cls = BlackHimRunnerCfg if him_enabled else BlackRunnerCfg
    return runner_cls(
        actor=actor_cls(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=algorithm_cls(
            **{
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 0.001,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            }
        ),
        experiment_name="black_locomotion",
        seed=42,
        run_name=run_name,
        load_run=load_run,
        save_interval=50,
        num_steps_per_env=64,
        max_iterations=10000,
        **({"warm_start_supported": stage == "flat"} if him_enabled else {}),
    )


def black_ppo_runner_cfg(stage: Literal["flat", "rough"] = "flat"):
    return make_rl_cfg(stage, him_enabled=False)


def black_him_runner_cfg(stage: Literal["flat", "rough"] = "flat"):
    return make_rl_cfg(stage, him_enabled=True)
