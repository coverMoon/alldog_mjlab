"""PPO → HIM actor warm start（initialization，不是 resume）。

语义边界：

```text
warm start = 用 PPO checkpoint 初始化新 HIM 模型（actor / distribution / critic / critic normalizer）
resume     = 同一算法的完整训练状态恢复（HIMPPO.load）
```

本模块只做前者，并且只支持同类结构迁移：

```text
PPO actor  : current_frame_dim → hidden → action_dim
HIM actor  : current_frame_dim + velocity_dim + latent_dim → hidden → action_dim

first Linear : W_him[:, :current_frame_dim] = W_ppo
               W_him[:, current_frame_dim:] = 0
               b_him = b_ppo
后续 Linear  : 逐 shape 严格复制
```

不迁移 PPO optimizer / iteration / env state / adaptive LR；estimator（source encoder /
target encoder / prototypes）保持 HIM 新模型的初始化，不从 PPO checkpoint 伪造。

这是 algorithm 层通用逻辑：不做 Black-specific 解析，也不在 ``HIMPPO.load()`` 中自动
识别 PPO checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .policy import HIMPolicy
from .spec import HIMSpec


@dataclass(frozen=True)
class WarmStartReport:
    """warm-start 迁移结果（用于日志 / 测试断言）。"""

    source_actor_input_dim: int
    target_actor_input_dim: int
    first_layer_copied_columns: int
    first_layer_zero_columns: int
    actor_policy_keys_copied: int
    critic_keys_copied: int


def _first_linear_param_names(module: nn.Module) -> tuple[str, str]:
    """在 module（这里的 actor MLP）中解析第一个 ``nn.Linear`` 的 weight / bias 参数名。"""
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            prefix = f"{name}." if name else ""
            return f"{prefix}weight", f"{prefix}bias"
    raise ValueError("warm start 失败：actor MLP 中找不到 nn.Linear layer")


def warm_start_from_ppo_actor(
    policy: HIMPolicy,
    critic: nn.Module,
    checkpoint: dict,
    spec: HIMSpec,
) -> WarmStartReport:
    """把 PPO checkpoint 的 actor / distribution / critic / normalizer state 迁移到 HIM 模型。

    任何结构 / shape / key 不一致都直接抛 ``ValueError``（不做 partial load，不用
    ``strict=False``）。
    """
    if "actor_state_dict" not in checkpoint:
        raise ValueError("warm start 失败：checkpoint 缺少 actor_state_dict")
    if "critic_state_dict" not in checkpoint:
        raise ValueError("warm start 失败：checkpoint 缺少 critic_state_dict")
    source_actor: dict = checkpoint["actor_state_dict"]
    source_critic: dict = checkpoint["critic_state_dict"]

    target_actor = dict(policy.state_dict())
    target_critic = dict(critic.state_dict())

    estimator_keys = {key for key in target_actor if key.startswith("estimator.")}
    policy_keys = set(target_actor) - estimator_keys

    source_estimator_keys = {key for key in source_actor if key.startswith("estimator.")}
    if source_estimator_keys:
        raise ValueError(
            "warm start 失败：source checkpoint 含 estimator state（看起来是 HIM checkpoint，"
            "不是 PPO actor）。warm start 只接受 PPO checkpoint；HIM checkpoint 请用 --resume。"
        )
    missing = sorted(policy_keys - set(source_actor))
    if missing:
        raise ValueError(f"warm start 失败：source actor 缺少 target 需要的 state keys: {missing}")
    unexpected = sorted(set(source_actor) - policy_keys)
    if unexpected:
        raise ValueError(
            f"warm start 失败：source actor 含 target 不支持的 state keys（结构不兼容）: {unexpected}"
        )

    first_weight_suffix, first_bias_suffix = _first_linear_param_names(policy.mlp)
    first_weight_key = f"mlp.{first_weight_suffix}"
    if first_weight_key not in policy_keys:
        raise ValueError(f"warm start 失败：actor state 中找不到 first layer weight {first_weight_key}")

    source_weight = source_actor[first_weight_key]
    target_weight = target_actor[first_weight_key]
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError("warm start 失败：actor first layer weight 不是 2D")
    if source_weight.shape[1] != spec.single_frame_dim:
        raise ValueError(
            f"warm start 失败：source actor input dim {source_weight.shape[1]} != "
            f"HIM current frame dim {spec.single_frame_dim}"
        )
    if target_weight.shape[1] != spec.actor_input_dim:
        raise ValueError(
            f"warm start 失败：target actor input dim {target_weight.shape[1]} != "
            f"HIM actor input dim {spec.actor_input_dim}"
        )
    if source_weight.shape[0] != target_weight.shape[0]:
        raise ValueError(
            f"warm start 失败：first layer hidden dim 不一致 "
            f"({source_weight.shape[0]} vs {target_weight.shape[0]})"
        )

    new_actor = dict(target_actor)
    new_first_weight = torch.zeros_like(target_weight)
    new_first_weight[:, : source_weight.shape[1]] = source_weight.to(new_first_weight.dtype)
    new_actor[first_weight_key] = new_first_weight

    copied = 0
    for key in sorted(policy_keys):
        if key == first_weight_key:
            continue
        if source_actor[key].shape != target_actor[key].shape:
            raise ValueError(
                f"warm start 失败：actor state shape mismatch at '{key}': "
                f"{tuple(source_actor[key].shape)} vs {tuple(target_actor[key].shape)}"
            )
        new_actor[key] = source_actor[key].to(target_actor[key].dtype)
        copied += 1
    # estimator keys 保持 target（新初始化）原值；strict load 验证整体 key 集合。
    policy.load_state_dict(new_actor, strict=True)

    if set(source_critic) != set(target_critic):
        raise ValueError(
            "warm start 失败：critic state keys 不一致: "
            f"missing={sorted(set(target_critic) - set(source_critic))}, "
            f"unexpected={sorted(set(source_critic) - set(target_critic))}"
        )
    for key in target_critic:
        if source_critic[key].shape != target_critic[key].shape:
            raise ValueError(
                f"warm start 失败：critic state shape mismatch at '{key}': "
                f"{tuple(source_critic[key].shape)} vs {tuple(target_critic[key].shape)}"
            )
    critic.load_state_dict(source_critic, strict=True)

    return WarmStartReport(
        source_actor_input_dim=int(source_weight.shape[1]),
        target_actor_input_dim=int(target_weight.shape[1]),
        first_layer_copied_columns=int(source_weight.shape[1]),
        first_layer_zero_columns=int(target_weight.shape[1] - source_weight.shape[1]),
        actor_policy_keys_copied=copied,
        critic_keys_copied=len(target_critic),
    )
