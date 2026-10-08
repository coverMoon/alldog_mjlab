"""Register the Wolf flat velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task

from alldog_mjlab.tasks.velocity.black.curriculum_checkpoint import (
    VelocityCommandCurriculumRunner,
)
from .env_cfgs import wolf_flat_env_cfg, wolf_flat_him_env_cfg
from .him_runner import WolfHimOnPolicyRunner
from .rl_cfg import wolf_him_runner_cfg, wolf_ppo_runner_cfg

# wolf-flat：普通 PPO，与 Black flat 同一 training 链路形态（curriculum +
# checkpoint env state 的 runner 由公共 VelocityCommandCurriculumRunner 提供）。
register_mjlab_task(
    task_id="wolf-flat",
    env_cfg=wolf_flat_env_cfg(),
    play_env_cfg=wolf_flat_env_cfg(play=True),
    rl_cfg=wolf_ppo_runner_cfg(stage="flat"),
    runner_cls=VelocityCommandCurriculumRunner,
)

# wolf-flat-him：wolf-flat + HIM observation/history/terminal contract + HIMPPO。
# 随机初始化可直接训练；`--agent.warm-start True` 从 wolf-flat PPO checkpoint
# 初始化（初始化，不是 resume；Wolf 与 Black 双向都不兼容，fail-loud）。
register_mjlab_task(
    task_id="wolf-flat-him",
    env_cfg=wolf_flat_him_env_cfg(),
    play_env_cfg=wolf_flat_him_env_cfg(play=True),
    rl_cfg=wolf_him_runner_cfg(stage="flat"),
    runner_cls=WolfHimOnPolicyRunner,
)
