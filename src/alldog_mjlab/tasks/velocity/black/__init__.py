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

# black-rough 已完成功能迁移和约 500 iteration 的 PPO baseline；
# 长训练收敛与定量评估仍待后续进行（见 .ai/MIGRATION.md §17.2）。
register_mjlab_task(
    task_id="black-rough",
    env_cfg=black_rough_env_cfg(),
    play_env_cfg=black_rough_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
