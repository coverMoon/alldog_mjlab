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
from .him.env_cfg import black_him_env_cfg
from .him.rl_cfg import runner_cfg
from nwwolf_mjlab.algorithms.him.runner import HIMRunner

for _rough in (False, True):
    register_mjlab_task(
        task_id="black-him-rough" if _rough else "black-him-flat",
        env_cfg=black_him_env_cfg(rough=_rough),
        play_env_cfg=black_him_env_cfg(play=True, rough=_rough),
        rl_cfg=runner_cfg(rough=_rough),
        runner_cls=HIMRunner,
    )
