"""Register the validated Black velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import black_flat_env_cfg, black_rough_env_cfg
from .rl_cfg import black_ppo_runner_cfg


register_mjlab_task(
    task_id="black-flat",
    env_cfg=black_flat_env_cfg(),
    play_env_cfg=black_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# black-rough 当前只覆盖 terrain generator + terrain curriculum：
# action / observation / reward / reset / termination / DR / command 与 flat 相同，
# 尚未做过 rough PPO 训练（见 .ai/MIGRATION.md）。
register_mjlab_task(
    task_id="black-rough",
    env_cfg=black_rough_env_cfg(),
    play_env_cfg=black_rough_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
