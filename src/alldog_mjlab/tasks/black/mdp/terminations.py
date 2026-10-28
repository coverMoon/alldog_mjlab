"""Black terminations interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers import ManagerTermBase
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
_COMMAND_NORM_FLOOR = 1e-06


class StuckTermination(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._command_name: str = cfg.params["command_name"]
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._command_threshold: float = cfg.params["command_threshold"]
        self._velocity_threshold: float = cfg.params["velocity_threshold"]
        self._grace_s: float = cfg.params["grace_s"]
        self._timeout_s: float = cfg.params["timeout_s"]
        self._stuck_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self._stuck_time[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
        del kwargs
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
