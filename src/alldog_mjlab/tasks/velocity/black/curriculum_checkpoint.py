"""Command curriculum 状态在 runner checkpoint 中的持久化（算法无关公共层）。

服务 PPO（VelocityOnPolicyRunner 路径的 BlackVelocityOnPolicyRunner）与
HIM（BlackHimOnPolicyRunner）两条 runner 路径：

- save：把 task 层 ``ForwardSpeedCommandCurriculum`` 的 authoritative state 写入
  checkpoint ``infos.env_state.command_curriculum``（与 common_step_counter 同处，
  模型 checkpoint 与 curriculum state 永远不错位；不再使用旧 super-dog 的独立 JSON）；
- load：按 restore mode 恢复（none / range / full），见 ``resolve_restore_mode()``。

计算 / buffer / 评估逻辑全部留在 task CurriculumTerm，本层只做序列化与模式判定。
save 主干与 MjLab v1.6.0 ``MjlabOnPolicyRunner.save`` 一致（pinned 复制；
其父类会整体重建 ``infos["env_state"]``，不经复制无法扩展该字段），升级 RSL-RL /
MjLab 时需重新核对。"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from alldog_mjlab.tasks.velocity.black.curriculums import (
    RestoreMode,
    get_command_curriculum_term,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

# 项目冻结的 run-name contract（black_config.RunnerParams）：run_name 后缀
# "_him" 只区分算法，terrain stage 由前缀决定。集中在这里，不散落字符串判断。
_RUN_NAME_STAGES = {
    "flat": "flat",
    "rough": "rough",
    "flat_him": "flat",
    "rough_him": "rough",
}


def stage_from_run_name(run_name: str | None) -> str:
    """从冻结的 stage run 名映射 terrain stage（flat | rough）；未知名 fail-loud。"""
    if run_name not in _RUN_NAME_STAGES:
        raise ValueError(
            "cannot determine command curriculum stage from run_name "
            f"{run_name!r}; expected one of {sorted(_RUN_NAME_STAGES)}"
        )
    return _RUN_NAME_STAGES[run_name]


def resolve_restore_mode(
    explicit: str,
    source_stage: str | None,
    target_stage: str,
) -> RestoreMode:
    """由显式配置（``command_curriculum_restore``）与 stage 元数据确定恢复模式。

    默认工作流（``auto``）：

    ```text
    new training             fresh（checkpoint 无 curriculum state，也无需恢复）
    same-stage resume        full（同 stage 精确续训：range + EMA + streak + buffer）
    cross-stage resume       range（black-flat→black-rough / flat HIM→rough HIM）
    旧 checkpoint（无 state）  none（打印 warning，保持 config 初始范围）
    source stage 未知         range（保守：范围保留，统计重新开始）
    ```

    PPO→HIM warm start 不走 resume load 路径，由 BlackHimOnPolicyRunner 显式
    传 ``mode="range"``。
    """
    if explicit in ("none", "range", "full"):
        return explicit  # type: ignore[return-value]
    if explicit != "auto":
        raise ValueError(
            f"unknown command_curriculum_restore mode: {explicit!r} "
            "(expected auto | none | range | full)"
        )
    if source_stage is None:
        return "none"
    if source_stage == target_stage:
        return "full"
    return "range"


def collect_command_curriculum_state(env: ManagerBasedRlEnv) -> dict[str, Any] | None:
    """收集 env 的 command curriculum state；无 term（play / disabled）时 None。"""
    term = get_command_curriculum_term(env)
    if term is None:
        return None
    return term.state_dict()


def apply_command_curriculum_state(
    env: ManagerBasedRlEnv,
    state: dict[str, Any] | None,
    restore_mode: RestoreMode,
) -> RestoreMode | None:
    """把 checkpoint 中的 curriculum state 按模式恢复到 env 的 curriculum term。

    - env 无 curriculum term（play / export）：静默忽略；
    - state 为 None（旧 checkpoint）：mode 为 ``none`` 时保持初始范围并打印一次
      明确 warning；显式 range/full 模式下因缺少数据无法恢复，fail-loud。
    返回实际使用的 mode（state None 或 term 缺失时为 None）。
    """
    term = get_command_curriculum_term(env)
    if term is None:
        return None
    if state is None:
        if restore_mode == "none":
            print(
                "[WARN] checkpoint 中没有 command curriculum state（旧 checkpoint）："
                "command curriculum 从 config 初始范围 [-1, 1] 开始，统计 fresh。"
            )
            return "none"
        raise ValueError(
            f"restore mode {restore_mode!r} requires command curriculum state in "
            "checkpoint infos.env_state, but it is missing (old checkpoint?). "
            "Use command_curriculum_restore='none' or train from scratch."
        )
    restored = resolve_restore_mode(restore_mode, state.get("stage"), term.stage)
    term.load_state_dict(state, mode=restored)
    if restored == "range":
        detail = "EMA / streak / buffer fresh"
    else:
        detail = (
            f"EMA {state['ema_low']:.3f}/{state['ema_high']:.3f}, "
            f"streak {state['pass_streak']}, "
            f"buffer {len(state['buffer_cmd_x'])}"
        )
    print(
        f"[INFO] command curriculum restore ({restored}): "
        f"vx_min {term.command_range[0]:.3f} / vx_max {term.command_range[1]:.3f}, {detail}"
    )
    return restored


class CommandCurriculumCheckpointMixin:
    """在 MjlabOnPolicyRunner 之上持久化 command curriculum env state 的 mixin。

    MRO 约定：必须排在 ``MjlabOnPolicyRunner``（或 ``VelocityOnPolicyRunner``）
    之前，ensure save() 先于父类 / 兄弟类执行、load() 后于父类执行。
    """

    env: RslRlVecEnvWrapper
    cfg: dict

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """写 checkpoint：curriculum state 追加进 ``infos.env_state``。

        save 主干与 MjLab v1.6.0 ``MjlabOnPolicyRunner.save`` 逐行一致
        （pinned dependency；唯一差异是 env_state 多一个 command_curriculum 字段）。
        """
        env_state: dict[str, Any] = {
            "common_step_counter": self.env.unwrapped.common_step_counter
        }
        curriculum = collect_command_curriculum_state(self.env.unwrapped)
        if curriculum is not None:
            env_state["command_curriculum"] = curriculum
        infos = {**(infos or {}), "env_state": env_state}
        # Inline base MjlabOnPolicyRunner.save() to extend infos.env_state.
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
        """先走 MjLab 原生 load（模型 / optimizer / counter），再恢复 curriculum。"""
        infos = super().load(path, load_cfg, strict, map_location)  # type: ignore[safe-super]
        self._restore_command_curriculum(infos)
        return infos

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _restore_command_curriculum(self, infos: dict | None) -> None:
        """按 auto / 显式模式恢复 curriculum state（export/play 无 term 时自动跳过）。"""
        env_state = (infos or {}).get("env_state") or {}
        state = env_state.get("command_curriculum")
        explicit = self.cfg.get("command_curriculum_restore", "auto")
        if state is None and explicit == "auto":
            # PPO / HIM 均可能来自旧 checkpoint；none 模式 + warning（一次 / per load）。
            apply_command_curriculum_state(self.env.unwrapped, None, "none")
            return
        apply_command_curriculum_state(self.env.unwrapped, state, explicit)


class BlackVelocityOnPolicyRunner(CommandCurriculumCheckpointMixin, VelocityOnPolicyRunner):
    """Black 普通 PPO 的 runner：VelocityOnPolicyRunner + checkpoint env state。"""
