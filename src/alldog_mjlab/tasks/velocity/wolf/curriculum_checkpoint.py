"""Wolf 的 command curriculum 状态在 runner checkpoint 中的持久化（task local）。

实现与 black/curriculum_checkpoint.py 独立同构（不 import Black），服务于 Wolf
PPO（`WolfVelocityCommandCurriculumRunner`）与 HIM（`WolfHimOnPolicyRunner`）
两条 runner 路径：

- save：把 task 层 ``WolfForwardSpeedCommandCurriculum`` 的 authoritative state
  写入 checkpoint ``infos.env_state.command_curriculum``（与 common_step_counter
  同处，模型 checkpoint 与 curriculum state 永远不错位）；
- load：按 restore mode 恢复（auto / none / range / full）。

序列化字段与 Black 版本一致（``version`` / ``stage`` / ``robot`` / vx range / EMA /
pass streak / buffer），保证现有 wolf-flat PPO/HIM checkpoint 直接可用。

save 主干与 MjLab v1.6.0 ``MjlabOnPolicyRunner.save`` 一致（pinned 复制；
其父类会整体重建 ``infos["env_state"]``，不经复制无法扩展该字段），升级 RSL-RL /
MjLab 时需重新核对。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from alldog_mjlab.tasks.velocity.wolf.curriculums import (
    RestoreMode,
    get_command_curriculum_term,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

# 项目冻结的 Wolf run-name contract（RunnerParams 的 stage run 名）。run_name
# 后缀 "_him" 只区分算法，terrain stage 由名字映射；集中在这里，不散落字符串判断。
_RUN_NAME_STAGES = {
    "wolf_flat": "flat",
    "wolf_rough": "rough",
    "wolf_flat_him": "flat",
    "wolf_rough_him": "rough",
}


def stage_from_run_name(run_name: str | None) -> str:
    """从冻结的 Wolf stage run 名映射 terrain stage（flat | rough）；未知名 fail-loud。"""
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
    cross-stage resume       range（wolf-flat→wolf-rough / flat HIM→rough HIM）
    旧 checkpoint（无 state）  none（打印 warning，保持 config 初始范围）
    source stage 未知         range（保守：范围保留，统计重新开始）
    ```

    PPO→HIM warm start 不走 resume load 路径，由 WolfHimOnPolicyRunner 显式
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
    robot: str | None = None,
) -> RestoreMode | None:
    """把 checkpoint 中的 curriculum state 按模式恢复到 env 的 curriculum term。

    - env 无 curriculum term（play / export）：静默忽略；
    - state 为 None（旧 checkpoint）：mode 为 ``none`` 时保持初始范围并打印一次
      明确 warning；显式 range/full 模式下因缺少数据无法恢复，fail-loud；
    - ``robot`` provenance 检查（可选）：checkpoint state 携带的 robot 与当前
      task 的 robot 不同 → fail-loud（跨机器人恢复禁止）；旧 checkpoint 无该
      字段时仅打印 warning（版本 1 迁移期）。
    返回实际使用的 mode（state None 或 term 缺失时为 None）。
    """
    term = get_command_curriculum_term(env)
    if term is None:
        return None
    if robot is not None and state is not None:
        state_robot = state.get("robot")
        if state_robot is not None and state_robot != robot:
            raise ValueError(
                "command curriculum state robot mismatch: checkpoint "
                f"{state_robot!r} != current task {robot!r}；禁止跨机器人恢复"
                "（curriculum only restores the same robot's vx range semantics）。"
            )
        if state_robot is None:
            print(
                f"[WARN] checkpoint 中无 curriculum robot 字段（旧 checkpoint，版本 1）；"
                f"无法验证来源是否为 {robot!r}，按同 robot 继续处理。"
            )
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

    ``ROBOT``：checkpoint provenance（跨机器人恢复 fail-loud）；Wolf task
    子类显式设置（Wolf = ``"wolf"``）。
    """

    ROBOT: str = "wolf"

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
            apply_command_curriculum_state(
                self.env.unwrapped, None, "none", robot=self.ROBOT
            )
            return
        apply_command_curriculum_state(
            self.env.unwrapped, state, explicit, robot=self.ROBOT
        )


class WolfVelocityCommandCurriculumRunner(CommandCurriculumCheckpointMixin, VelocityOnPolicyRunner):
    """Wolf 普通 PPO 的 runner：VelocityOnPolicyRunner + checkpoint env state。

    ``ROBOT = "wolf"``：与 Wolf curriculum state 的 ``robot="wolf"`` 对齐。
    修复旧注册的问题：wolf-flat PPO 曾直接复用 Black 目录的 runner
    （类属性 ROBOT="black"），与 checkpoint state 中 robot="wolf" 的 provenance
    判定不一致。
    """

    ROBOT = "wolf"
