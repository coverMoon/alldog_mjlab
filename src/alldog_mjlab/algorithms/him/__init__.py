"""HIM algorithm 层（RSL-RL v5.4.2 + MjLab v1.6.0）。

模块职责：

```text
spec.py       HIMSpec / HIMInterface / canonical history / target 构造
estimator.py  HIMEstimator（source encoder + target encoder + prototypes + loss/update）
policy.py     HIMPolicy（RSL-RL actor 模型接口：estimator + actor MLP + distribution）
storage.py    HIMRolloutStorage（successor estimator target buffers）
ppo.py        HIMPPO（PPO + 每 mini-batch estimator update）
```

不包含 runner、task registration、PPO→HIM warm start 与 exporter。
"""

from .estimator import HIMEstimator, sinkhorn as sinkhorn
from .policy import HIMPolicy
from .ppo import HIMPPO
from .spec import (
    HIMInterface,
    HIMSpec,
    canonical_history,
    estimator_target_input,
    him_spec_from_obs,
)
from .storage import HIMBatch, HIMRolloutStorage, HIMTransition
from .warm_start import WarmStartReport, warm_start_from_ppo_actor

__all__ = [
    "HIMBatch",
    "HIMEstimator",
    "HIMInterface",
    "HIMPolicy",
    "HIMPPO",
    "HIMRolloutStorage",
    "HIMSpec",
    "HIMTransition",
    "WarmStartReport",
    "canonical_history",
    "estimator_target_input",
    "him_spec_from_obs",
    "sinkhorn",
    "warm_start_from_ppo_actor",
]
