"""Register four Wolf locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    wolf_flat_env_cfg,
    wolf_flat_him_env_cfg,
    wolf_rough_env_cfg,
    wolf_rough_him_env_cfg,
)
from .rl_cfg import wolf_him_runner_cfg, wolf_ppo_runner_cfg
from .runner import WolfHimOnPolicyRunner, WolfVelocityCommandCurriculumRunner

register_mjlab_task(
    task_id="wolf-flat",
    env_cfg=wolf_flat_env_cfg(),
    play_env_cfg=wolf_flat_env_cfg(play=True),
    rl_cfg=wolf_ppo_runner_cfg(stage="flat"),
    runner_cls=WolfVelocityCommandCurriculumRunner,
)
register_mjlab_task(
    task_id="wolf-rough",
    env_cfg=wolf_rough_env_cfg(),
    play_env_cfg=wolf_rough_env_cfg(play=True),
    rl_cfg=wolf_ppo_runner_cfg(stage="rough"),
    runner_cls=WolfVelocityCommandCurriculumRunner,
)
register_mjlab_task(
    task_id="wolf-flat-him",
    env_cfg=wolf_flat_him_env_cfg(),
    play_env_cfg=wolf_flat_him_env_cfg(play=True),
    rl_cfg=wolf_him_runner_cfg(stage="flat"),
    runner_cls=WolfHimOnPolicyRunner,
)
register_mjlab_task(
    task_id="wolf-rough-him",
    env_cfg=wolf_rough_him_env_cfg(),
    play_env_cfg=wolf_rough_him_env_cfg(play=True),
    rl_cfg=wolf_him_runner_cfg(stage="rough"),
    runner_cls=WolfHimOnPolicyRunner,
)
