"""Black velocity task 的 stateful termination terms。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers import ManagerTermBase
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

# command 方向归一化的分母下限，避免零指令时除零。
_COMMAND_NORM_FLOOR = 1e-6


class StuckTermination(ManagerTermBase):
    """停滞检测：沿 commanded planar direction 的 progress speed 持续过低即终止。

    与无状态 termination 不同，本 term 需要跨 step 保存每个 env 的停滞时长，
    因此是 class-based term：``__init__`` 分配计时 buffer，``__call__`` 逐步累加，
    ``TerminationManager.reset()`` 在 episode reset 时按 env 清零。

    判定语义：
    - 只使用 command 的 planar 分量（vx / vy），不含 yaw；
    - progress speed 取 body-frame root 线速度在 command 方向上的投影，
      因此速度模长很大但方向不符（正交）或沿反方向运动都算 stalled；
    - 只要 move command 失效、progress 恢复或处于 grace 期，计时立即归零（连续计时）。
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._command_name: str = cfg.params["command_name"]
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._command_threshold: float = cfg.params["command_threshold"]
        self._velocity_threshold: float = cfg.params["velocity_threshold"]
        self._grace_s: float = cfg.params["grace_s"]
        self._timeout_s: float = cfg.params["timeout_s"]
        # 计时单位是秒（不是 step 数），每个 env 独立。
        self._stuck_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self._stuck_time[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
        del kwargs  # 参数已在 __init__ 中读取。
        # raw physical command（m/s），不是乘过 obs scale 的 actor observation。
        command = env.command_manager.get_command(self._command_name)

        move_cmd_norm = torch.norm(command[:, :2], dim=1)
        move_cmd = move_cmd_norm > self._command_threshold
        cmd_dir = command[:, :2] / torch.clamp(
            move_cmd_norm.unsqueeze(1), min=_COMMAND_NORM_FLOOR
        )
        progress_speed = torch.sum(
            self._asset.data.root_link_lin_vel_b[:, :2] * cmd_dir, dim=1
        )
        stalled = progress_speed < self._velocity_threshold

        grace_done = env.episode_length_buf.float() * env.step_dt > self._grace_s
        stuck_mask = move_cmd & stalled & grace_done

        self._stuck_time[:] = torch.where(
            stuck_mask,
            self._stuck_time + env.step_dt,
            torch.zeros_like(self._stuck_time),
        )
        return self._stuck_time > self._timeout_s
