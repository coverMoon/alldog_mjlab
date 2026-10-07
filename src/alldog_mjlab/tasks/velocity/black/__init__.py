"""Register the validated Black velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .curriculum_checkpoint import BlackVelocityOnPolicyRunner
from .env_cfgs import (
    black_flat_env_cfg,
    black_flat_him_env_cfg,
    black_rough_env_cfg,
    black_rough_him_env_cfg,
)
from .him_rl_cfg import black_him_runner_cfg
from .him_runner import BlackHimOnPolicyRunner
from .rl_cfg import black_ppo_runner_cfg


register_mjlab_task(
    task_id="black-flat",
    env_cfg=black_flat_env_cfg(),
    play_env_cfg=black_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

# black-rough 已完成功能迁移和约 500 iteration 的 PPO baseline；
# 长训练收敛与定量评估仍待后续进行（见 .ai/MIGRATION.md §17.2）。
register_mjlab_task(
    task_id="black-rough",
    env_cfg=black_rough_env_cfg(),
    play_env_cfg=black_rough_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="rough"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

# black-flat-him = black-flat + HIM observation/history/terminal contract + HIMPPO。
# 随机初始化可直接训练；可选 `--agent.warm-start True` 从 black-flat PPO checkpoint
# 初始化 actor/critic（初始化，不是 resume，见 §22 / him_runner.py）。
register_mjlab_task(
    task_id="black-flat-him",
    env_cfg=black_flat_him_env_cfg(),
    play_env_cfg=black_flat_him_env_cfg(play=True),
    rl_cfg=black_him_runner_cfg(stage="flat"),
    runner_cls=BlackHimOnPolicyRunner,
)

# black-rough-him = black-rough + HIM observation/history/terminal contract + HIMPPO。
# terrain / reward / command / termination / DR / action contract 全部复用 rough，不因 HIM 修改。
# 训练路线：flat PPO → flat HIM（warm start）→ rough HIM（--agent.resume True full resume，
# 完整恢复 HIM 训练状态；不继承 flat env runtime state）。rough HIM 显式不支持 PPO→HIM
# warm start（见 him_runner.py / him_rl_cfg.py 的 warm_start_supported）。
register_mjlab_task(
    task_id="black-rough-him",
    env_cfg=black_rough_him_env_cfg(),
    play_env_cfg=black_rough_him_env_cfg(play=True),
    rl_cfg=black_him_runner_cfg(stage="rough"),
    runner_cls=BlackHimOnPolicyRunner,
)
