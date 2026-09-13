"""Register the validated Black flat-ground PPO task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import black_flat_env_cfg
from .rl_cfg import black_ppo_runner_cfg


register_mjlab_task(
    task_id="black-flat",
    env_cfg=black_flat_env_cfg(),
    play_env_cfg=black_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
