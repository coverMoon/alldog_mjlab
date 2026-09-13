from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg
from .source_config import BlackCfgPPO


def values(cls):
    return {key:value for key,value in vars(cls).items() if not key.startswith('_')}


@dataclass
class HIMRunnerCfg(RslRlBaseRunnerCfg):
    policy: dict = field(default_factory=lambda: values(BlackCfgPPO.policy))
    him_algorithm: dict = field(default_factory=lambda: values(BlackCfgPPO.algorithm))


def runner_cfg(rough=False):
    return HIMRunnerCfg(num_steps_per_env=100, max_iterations=1000, save_interval=50, experiment_name="black_him_rough" if rough else "black_him_flat", clip_actions=100., logger="tensorboard", upload_model=False)
