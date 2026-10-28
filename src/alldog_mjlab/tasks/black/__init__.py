"""Register four Black locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    black_flat_env_cfg,
    black_flat_him_env_cfg,
    black_rough_env_cfg,
    black_rough_him_env_cfg,
)
from .rl_cfg import black_him_runner_cfg, black_ppo_runner_cfg
from .runner import BlackHimOnPolicyRunner, BlackVelocityOnPolicyRunner

register_mjlab_task(
    task_id="black-flat",
    env_cfg=black_flat_env_cfg(),
    play_env_cfg=black_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="black-rough",
    env_cfg=black_rough_env_cfg(),
    play_env_cfg=black_rough_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="rough"),
    runner_cls=BlackVelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="black-flat-him",
    env_cfg=black_flat_him_env_cfg(),
    play_env_cfg=black_flat_him_env_cfg(play=True),
    rl_cfg=black_him_runner_cfg(stage="flat"),
    runner_cls=BlackHimOnPolicyRunner,
)
register_mjlab_task(
    task_id="black-rough-him",
    env_cfg=black_rough_him_env_cfg(),
    play_env_cfg=black_rough_him_env_cfg(play=True),
    rl_cfg=black_him_runner_cfg(stage="rough"),
    runner_cls=BlackHimOnPolicyRunner,
)
