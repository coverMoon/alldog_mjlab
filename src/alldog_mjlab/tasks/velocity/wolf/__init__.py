"""Register the Wolf velocity tasks（flat / rough × PPO / HIM，全部 task local）。

四个任务共用 Wolf 的基础控制契约（robot asset / IMU / 16-D action / 53-D actor /
56-D critic / PD / default pose / 50 Hz policy dt / DR 事件表）；flat 与 rough 只差
terrain 与其派生语义（terrain generator、base_height 测量方式、out_of_terrain_bounds、
terrain curriculum）。

训练路线：

```text
wolf-flat PPO
    → wolf-rough PPO               （resume；command curriculum 跨 stage range 恢复）
    → wolf-flat-him                （PPO→HIM warm start；拷贝 checkpoint 网络列）
wolf-flat-him
    → wolf-rough-him               （HIM full resume；command curriculum range 恢复）
```

rough HIM 显式不支持 PPO→HIM warm start（`warm_start_supported=False`，请求时
fail-loud）。
"""

from mjlab.tasks.registry import register_mjlab_task

from .curriculum_checkpoint import WolfVelocityCommandCurriculumRunner
from .env_cfgs import (
    wolf_flat_env_cfg,
    wolf_flat_him_env_cfg,
    wolf_rough_env_cfg,
    wolf_rough_him_env_cfg,
)
from .him_runner import WolfHimOnPolicyRunner
from .rl_cfg import wolf_him_runner_cfg, wolf_ppo_runner_cfg

# wolf-flat：普通 PPO（curriculum + checkpoint env state 的 runner 由 Wolf 本地
# WolfVelocityCommandCurriculumRunner 提供，ROBOT="wolf"）。
register_mjlab_task(
    task_id="wolf-flat",
    env_cfg=wolf_flat_env_cfg(),
    play_env_cfg=wolf_flat_env_cfg(play=True),
    rl_cfg=wolf_ppo_runner_cfg(stage="flat"),
    runner_cls=WolfVelocityCommandCurriculumRunner,
)

# wolf-rough：普通 PPO + rough terrain（flat contract + generator 地形 / terrain
# level 课程 / 越界截断 / terrain-relative base_height）。actor/critic shape 与
# flat 完全一致，可从 wolf-flat checkpoint 直接 resume。
register_mjlab_task(
    task_id="wolf-rough",
    env_cfg=wolf_rough_env_cfg(),
    play_env_cfg=wolf_rough_env_cfg(play=True),
    rl_cfg=wolf_ppo_runner_cfg(stage="rough"),
    runner_cls=WolfVelocityCommandCurriculumRunner,
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

# wolf-rough-him = wolf-rough + HIM contract。训练路线：flat HIM → rough HIM
# （--agent.resume True full resume；不继承 flat env runtime state；command
# curriculum 跨 stage range 恢复）。rough HIM 显式不支持 PPO→HIM warm start。
register_mjlab_task(
    task_id="wolf-rough-him",
    env_cfg=wolf_rough_him_env_cfg(),
    play_env_cfg=wolf_rough_him_env_cfg(play=True),
    rl_cfg=wolf_him_runner_cfg(stage="rough"),
    runner_cls=WolfHimOnPolicyRunner,
)
