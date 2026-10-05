"""Black HIM 的 MjlabOnPolicyRunner 扩展。

只增加一件事：可选的 PPO → HIM warm-start 初始化。训练循环 / checkpoint / resume
语义完全继承 ``MjlabOnPolicyRunner``，没有 custom HIM runner，也没有 HIM 特判分支。

warm start 与 resume 严格区分：

```text
--agent.warm-start True   ：用 black-flat PPO checkpoint 初始化新 HIM run（新 optimizer /
                            新 iteration / 新 env state）
--agent.resume True       ：同一 HIM 算法的完整训练状态恢复（HIMPPO.load）
两者同时开启 → 直接报错
```

source checkpoint 定位复用 runner cfg 的 ``load_run`` / ``load_checkpoint``，
路径解析与 MjLab ``run_train`` 的 resume 路径完全一致
（``<log_root>/<experiment_name>/<run>/<checkpoint>``）。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from alldog_mjlab.algorithms.him.warm_start import warm_start_from_ppo_actor

WARM_START_SOURCE_TASK = "black-flat"


class BlackHimOnPolicyRunner(MjlabOnPolicyRunner):
    """``MjlabOnPolicyRunner`` + 可选 PPO → HIM warm start。"""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if train_cfg.get("warm_start") and train_cfg.get("resume"):
            raise ValueError(
                "PPO→HIM warm start 与同算法 resume 互斥："
                "`--agent.warm-start` 与 `--agent.resume` 不能同时使用。"
                "warm start 是初始化（新 optimizer / iteration），resume 是 HIM 训练状态恢复。"
            )
        self._warm_start_info: dict | None = None
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if train_cfg.get("warm_start"):
            self._warm_start_info = self._apply_warm_start(train_cfg, log_dir)

    def _resolve_source_checkpoint(self, train_cfg: dict, log_dir) -> Path:
        if log_dir is None:
            raise ValueError(
                "warm start 需要 `--log-root`（用于定位 source run）："
                "请提供 `--agent.load-run` / `--agent.load-checkpoint`。"
            )
        log_root_path = Path(log_dir).parent
        return get_checkpoint_path(
            log_root_path,
            train_cfg["load_run"],
            train_cfg["load_checkpoint"],
        )

    def _apply_warm_start(self, train_cfg: dict, log_dir) -> dict:
        source_path = self._resolve_source_checkpoint(train_cfg, log_dir)
        checkpoint = torch.load(str(source_path), map_location=self.device, weights_only=False)
        report = warm_start_from_ppo_actor(
            self.alg.actor,
            self.alg.critic,
            checkpoint,
            self.alg.spec,
        )
        print(
            f"[INFO] PPO→HIM warm start: source={source_path} "
            f"actor_input {report.source_actor_input_dim}→{report.target_actor_input_dim} "
            f"(copied {report.first_layer_copied_columns}, zero {report.first_layer_zero_columns}), "
            f"critic keys {report.critic_keys_copied}. "
            "optimizer / iteration / env state 未继承。"
        )
        return {
            "source_task": WARM_START_SOURCE_TASK,
            "source_checkpoint": str(source_path),
            **asdict(report),
        }

    @property
    def warm_start_info(self) -> dict | None:
        return self._warm_start_info

    def save(self, path: str, infos: dict | None = None) -> None:
        """与父类相同，只额外记录 warm-start provenance。"""
        if self._warm_start_info is not None:
            infos = {**(infos or {}), "warm_start": dict(self._warm_start_info)}
        super().save(path, infos)
