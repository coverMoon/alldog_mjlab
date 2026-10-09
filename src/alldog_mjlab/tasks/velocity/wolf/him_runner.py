"""Wolf HIM 的 MjlabOnPolicyRunner 扩展（task local，不 import Black task）。

实现与 black/him_runner.py 独立同构：只增加可选的 PPO → HIM warm-start 初始化
+ command curriculum checkpoint 状态（经 Wolf 的 checkpoint mixin）。训练循环 /
checkpoint / resume 语义完全继承 ``MjlabOnPolicyRunner``，没有 custom HIM runner。

warm start 与 resume 严格区分：

```text
--agent.warm-start True   ：用 wolf-flat PPO checkpoint 初始化新 HIM run
                            （新 optimizer / 新 iteration / 新 env state）；
                            仅 wolf-flat-him 支持（rough HIM 从 flat HIM
                            checkpoint full resume）
--agent.resume True       ：同一 HIM 算法的完整训练状态恢复（HIMPPO.load）
两者同时开启 → 直接报错
```

正式训练路线为 flat PPO → flat HIM → rough HIM，因此 rough HIM 显式不支持
PPO→HIM warm start（`warm_start_supported = False`，见 rl_cfg.py），请求时直接报错。

source checkpoint 定位复用 runner cfg 的 ``load_run`` / ``load_checkpoint``，
路径解析与 MjLab ``run_train`` 的 resume 路径完全一致
（``<log_root>/<experiment_name>/<run>/<checkpoint>``）。

checkpoint 兼容性：现有 wolf-flat-him checkpoint（由 Black 目录的
``WolfHimOnPolicyRunner`` 旧实例保存）的内容结构与本类读写的结构一致
（``infos.env_state.command_curriculum`` / ``infos.warm_start``），可直接 resume。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from alldog_mjlab.algorithms.him.warm_start import warm_start_from_ppo_actor
from alldog_mjlab.tasks.velocity.wolf.curriculum_checkpoint import (
    CommandCurriculumCheckpointMixin,
    apply_command_curriculum_state,
)

WARM_START_SOURCE_TASK = "wolf-flat"


class WolfHimOnPolicyRunner(CommandCurriculumCheckpointMixin, MjlabOnPolicyRunner):
    """``MjlabOnPolicyRunner`` + 可选 PPO → HIM warm start + curriculum
    checkpoint 状态（经 Wolf 的 checkpoint mixin，与 Wolf PPO 路径同语义）。
    """

    WARM_START_SOURCE_TASK: str = WARM_START_SOURCE_TASK
    ROBOT: str = "wolf"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if train_cfg.get("warm_start") and train_cfg.get("resume"):
            raise ValueError(
                "PPO→HIM warm start 与同算法 resume 互斥："
                "`--agent.warm-start` 与 `--agent.resume` 不能同时使用。"
                "warm start 是初始化（新 optimizer / iteration），resume 是 HIM 训练状态恢复。"
            )
        if train_cfg.get("warm_start") and not train_cfg.get("warm_start_supported", True):
            raise ValueError(
                "PPO→HIM warm start currently supported only for wolf-flat-him. "
                "Train/warm-start flat HIM first, then full-resume into wolf-rough-him."
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
        # PPO→HIM warm start 的 command curriculum 语义：只拷贝 vx range（EMA /
        # streak / buffer fresh）；旧 PPO checkpoint 没有 curriculum state 时
        # 打印 warning 并保持 config 初始范围，不让 warm start 失败。
        # ``robot`` provenance 检查（ROBOT="wolf"）拒绝跨机器人 warm start。
        curriculum_state = (checkpoint.get("infos") or {}).get("env_state", {}).get(
            "command_curriculum"
        )
        curriculum_restored: str | None
        if curriculum_state is not None:
            curriculum_restored = apply_command_curriculum_state(
                self.env.unwrapped, curriculum_state, "range", robot=self.ROBOT
            )
        else:
            curriculum_restored = None
            print(
                "[WARN] PPO→HIM warm start: source checkpoint 没有 command "
                "curriculum state；command range 保持 config 初始 [-1, 1]。"
            )
        print(
            f"[INFO] PPO→HIM warm start: source={source_path} "
            f"actor_input {report.source_actor_input_dim}→{report.target_actor_input_dim} "
            f"(copied {report.first_layer_copied_columns}, zero {report.first_layer_zero_columns}), "
            f"critic keys {report.critic_keys_copied}. "
            "optimizer / iteration / env state 未继承。"
        )
        return {
            "source_task": self.WARM_START_SOURCE_TASK,
            "source_checkpoint": str(source_path),
            "command_curriculum_restored": curriculum_restored,
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
