"""Wolf PPO/HIM runners with curriculum checkpoint and warm-start support."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from alldog_mjlab.algorithms.him.warm_start import warm_start_from_ppo_actor
from alldog_mjlab.tasks.wolf.mdp.curriculums import (
    RestoreMode,
    get_command_curriculum_term,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
_RUN_NAME_STAGES = {
    "wolf_flat": "flat",
    "wolf_rough": "rough",
    "wolf_flat_him": "flat",
    "wolf_rough_him": "rough",
}


def stage_from_run_name(run_name: str | None) -> str:
    if run_name not in _RUN_NAME_STAGES:
        raise ValueError(
            f"cannot determine command curriculum stage from run_name {run_name!r}; expected one of {sorted(_RUN_NAME_STAGES)}"
        )
    return _RUN_NAME_STAGES[run_name]


def resolve_restore_mode(
    explicit: str, source_stage: str | None, target_stage: str
) -> RestoreMode:
    if explicit in ("none", "range", "full"):
        return explicit
    if explicit != "auto":
        raise ValueError(
            f"unknown command_curriculum_restore mode: {explicit!r} (expected auto | none | range | full)"
        )
    if source_stage is None:
        return "none"
    if source_stage == target_stage:
        return "full"
    return "range"


def collect_command_curriculum_state(env: ManagerBasedRlEnv) -> dict[str, Any] | None:
    term = get_command_curriculum_term(env)
    if term is None:
        return None
    return term.state_dict()


def apply_command_curriculum_state(
    env: ManagerBasedRlEnv,
    state: dict[str, Any] | None,
    restore_mode: RestoreMode,
    robot: str | None = None,
) -> RestoreMode | None:
    term = get_command_curriculum_term(env)
    if term is None:
        return None
    if robot is not None and state is not None:
        state_robot = state.get("robot")
        if state_robot is not None and state_robot != robot:
            raise ValueError(
                f"command curriculum state robot mismatch: checkpoint {state_robot!r} != current task {robot!r}；禁止跨机器人恢复（curriculum only restores the same robot's vx range semantics）。"
            )
        if state_robot is None:
            print(
                f"[WARN] checkpoint 中无 curriculum robot 字段（旧 checkpoint，版本 1）；无法验证来源是否为 {robot!r}，按同 robot 继续处理。"
            )
    if state is None:
        if restore_mode == "none":
            print(
                "[WARN] checkpoint 中没有 command curriculum state（旧 checkpoint）：command curriculum 从 config 初始范围 [-1, 1] 开始，统计 fresh。"
            )
            return "none"
        raise ValueError(
            f"restore mode {restore_mode!r} requires command curriculum state in checkpoint infos.env_state, but it is missing (old checkpoint?). Use command_curriculum_restore='none' or train from scratch."
        )
    restored = resolve_restore_mode(restore_mode, state.get("stage"), term.stage)
    if (
        state.get("version") in (1, 2)
        and restored == "full"
        and (restore_mode == "auto")
    ):
        print(
            "[WARN] curriculum statistics version changed; restoring speed range and clearing statistics"
        )
        restored = "range"
    term.load_state_dict(state, mode=restored)
    if restored == "none":
        detail = "range and statistics unchanged"
    elif restored == "range":
        detail = "EMA / streak / buffer fresh"
    else:
        detail = f"EMA {state['ema_low']:.3f}/{state['ema_high']:.3f}, streak {state['pass_streak']}, buffer {len(state['buffer_cmd_x'])}"
    print(
        f"[INFO] command curriculum restore ({restored}): vx_min {term.command_range[0]:.3f} / vx_max {term.command_range[1]:.3f}, {detail}"
    )
    return restored


class CommandCurriculumCheckpointMixin:
    ROBOT: str = "wolf"
    env: RslRlVecEnvWrapper
    cfg: dict

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        checkpoint_path = train_cfg.get("checkpoint_path")
        if checkpoint_path and train_cfg.get("resume"):
            raise ValueError(
                "use checkpoint_path for an explicit file or resume for run lookup, not both"
            )
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if checkpoint_path and (not train_cfg.get("warm_start")):
            self.load(
                str(Path(checkpoint_path).expanduser()),
                strict=True,
                map_location=device,
            )

    def save(self, path: str, infos: dict | None = None) -> None:
        env_state: dict[str, Any] = {
            "common_step_counter": self.env.unwrapped.common_step_counter
        }
        curriculum = collect_command_curriculum_state(self.env.unwrapped)
        if curriculum is not None:
            env_state["command_curriculum"] = curriculum
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(
            path, map_location=map_location or self.device, weights_only=False
        )
        actor_state = checkpoint.get("actor_state_dict", {})
        if actor_state:
            source_him = any((key.startswith("estimator.") for key in actor_state))
            target_him = hasattr(self.alg.get_policy(), "estimator")
            if source_him != target_him:
                raise ValueError(
                    "checkpoint algorithm mismatch: PPO/HIM resume requires the same algorithm; use explicit warm start for PPO to HIM"
                )
            if (
                target_him
                and (load_cfg is None or load_cfg.get("optimizer", False))
                and ("estimator_optimizer_state_dict" not in checkpoint)
            ):
                raise ValueError(
                    "HIM resume checkpoint is missing estimator optimizer state"
                )
        source_state = ((checkpoint.get("infos") or {}).get("env_state") or {}).get(
            "command_curriculum"
        )
        if source_state and source_state.get("robot") not in (None, self.ROBOT):
            raise ValueError("checkpoint robot mismatch")
        infos = super().load(path, load_cfg, strict, map_location)
        self._restore_command_curriculum(infos)
        return infos

    def _restore_command_curriculum(self, infos: dict | None) -> None:
        env_state = (infos or {}).get("env_state") or {}
        state = env_state.get("command_curriculum")
        explicit = self.cfg.get("command_curriculum_restore", "auto")
        if state is None and explicit == "auto":
            apply_command_curriculum_state(
                self.env.unwrapped, None, "none", robot=self.ROBOT
            )
            return
        apply_command_curriculum_state(
            self.env.unwrapped, state, explicit, robot=self.ROBOT
        )


class WolfVelocityCommandCurriculumRunner(
    CommandCurriculumCheckpointMixin, VelocityOnPolicyRunner
):
    ROBOT = "wolf"


WARM_START_SOURCE_TASK = "wolf-flat"


class WolfHimOnPolicyRunner(CommandCurriculumCheckpointMixin, MjlabOnPolicyRunner):
    WARM_START_SOURCE_TASK: str = WARM_START_SOURCE_TASK
    ROBOT: str = "wolf"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if train_cfg.get("warm_start") and train_cfg.get("resume"):
            raise ValueError(
                "PPO→HIM warm start 与同算法 resume 互斥：`--agent.warm-start` 与 `--agent.resume` 不能同时使用。warm start 是初始化（新 optimizer / iteration），resume 是 HIM 训练状态恢复。"
            )
        if train_cfg.get("warm_start") and (
            not train_cfg.get("warm_start_supported", True)
        ):
            raise ValueError(
                "PPO→HIM warm start currently supported only for wolf-flat-him. Train/warm-start flat HIM first, then full-resume into wolf-rough-him."
            )
        self._warm_start_info: dict | None = None
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if train_cfg.get("warm_start"):
            self._warm_start_info = self._apply_warm_start(train_cfg, log_dir)

    def _resolve_source_checkpoint(self, train_cfg: dict, log_dir) -> Path:
        explicit = Path(
            train_cfg.get("checkpoint_path") or train_cfg["load_checkpoint"]
        ).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        if train_cfg.get("checkpoint_path") or explicit.is_absolute():
            raise FileNotFoundError(f"checkpoint not found: {explicit}")
        if log_dir is None:
            raise ValueError(
                "warm start 需要 `--log-root`（用于定位 source run）：请提供 `--agent.load-run` / `--agent.load-checkpoint`。"
            )
        log_root_path = Path(log_dir).parent
        return get_checkpoint_path(
            log_root_path, train_cfg["load_run"], train_cfg["load_checkpoint"]
        )

    def _apply_warm_start(self, train_cfg: dict, log_dir) -> dict:
        source_path = self._resolve_source_checkpoint(train_cfg, log_dir)
        checkpoint = torch.load(
            str(source_path), map_location=self.device, weights_only=False
        )
        report = warm_start_from_ppo_actor(
            self.alg.actor, self.alg.critic, checkpoint, self.alg.spec
        )
        curriculum_state = (
            (checkpoint.get("infos") or {})
            .get("env_state", {})
            .get("command_curriculum")
        )
        curriculum_restored: str | None
        if curriculum_state is not None:
            curriculum_restored = apply_command_curriculum_state(
                self.env.unwrapped, curriculum_state, "range", robot=self.ROBOT
            )
        else:
            curriculum_restored = None
            print(
                "[WARN] PPO→HIM warm start: source checkpoint 没有 command curriculum state；command range 保持 config 初始 [-1, 1]。"
            )
        print(
            f"[INFO] PPO→HIM warm start: source={source_path} actor_input {report.source_actor_input_dim}→{report.target_actor_input_dim} (copied {report.first_layer_copied_columns}, zero {report.first_layer_zero_columns}), critic keys {report.critic_keys_copied}. optimizer / iteration / env state 未继承。"
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
        if self._warm_start_info is not None:
            infos = {**(infos or {}), "warm_start": dict(self._warm_start_info)}
        super().save(path, infos)
