"""Wolf HIM 的 runner：复用 Black HIM runner 的公共机制，仅 override task 标识。

warm start / checkpoint / curriculum 逻辑全部继承；Wolf 与 Black 的唯一差异是：

```text
WARM_START_SOURCE_TASK = "wolf-flat"   （仅接受 wolf-flat PPO checkpoint；
                                        与 Black checkpoint 的 actor 维度也不兼容
                                        —— 45 != 53，warm_start_from_ppo_actor 会
                                        fail-loud，双重保险）
ROBOT = "wolf"                          （curriculum state provenance，
                                        跨机器人恢复 fail-loud）
```
"""

from alldog_mjlab.tasks.velocity.black.him_runner import BlackHimOnPolicyRunner


class WolfHimOnPolicyRunner(BlackHimOnPolicyRunner):
    """Wolf PPO→HIM warm start / HIM resume runner（无第二份实现）。"""

    WARM_START_SOURCE_TASK = "wolf-flat"
    ROBOT = "wolf"
