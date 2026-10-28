"""Black curriculums interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from mjlab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
CURRICULUM_STATE_VERSION = 2
CURRICULUM_TERM_NAME = "command"
RestoreMode = Literal["none", "range", "full"]
RunnerRestoreMode = Literal["auto", "none", "range", "full"]


@dataclass(frozen=True)
class CommandCurriculumParams:
    enabled: bool = True
    initial_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    max_abs_vx: float = 2.0
    step: float = 0.1
    threshold_low: float = 0.7
    threshold_offset: float = 0.1
    ema_alpha: float = 0.2
    required_passes: int = 2
    buffer_min: int = 256
    min_low_count: int = 8
    min_high_count: int = 4
    low_speed_min: float = 0.2
    low_high_split_ratio: float = 0.6

    def threshold_high(self) -> float:
        return self.threshold_low - self.threshold_offset

    def validate(self) -> None:
        if not 0.0 < self.ema_alpha < 1.0:
            raise ValueError("command curriculum ema_alpha must be in (0, 1)")
        if self.required_passes < 1:
            raise ValueError("command curriculum required_passes must be >= 1")
        if self.buffer_min < 1:
            raise ValueError("command curriculum buffer_min must be >= 1")
        if self.step <= 0.0 or self.max_abs_vx <= 0.0:
            raise ValueError("command curriculum step / max_abs_vx must be positive")
        if self.threshold_high() <= 0.0:
            raise ValueError(
                f"command curriculum high threshold must be positive: {self.threshold_low} - {self.threshold_offset}"
            )
        if self.min_low_count < 1 or self.min_high_count < 1:
            raise ValueError("command curriculum group counts must be >= 1")


class ForwardSpeedCommandCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._params: CommandCurriculumParams = cfg.params["curriculum_params"]
        self._params.validate()
        self._command_name: str = cfg.params["command_name"]
        self._reward_term_name: str = cfg.params["reward_term_name"]
        self._stage: str = cfg.params["stage"]
        self._robot: str = cfg.params.get("robot", "black")
        self._vx_min: float = self._params.initial_lin_vel_x[0]
        self._vx_max: float = self._params.initial_lin_vel_x[1]
        self._ema_low: float = 0.0
        self._ema_high: float = 0.0
        self._pass_streak: int = 0
        self._buffer_cmd_x: list[torch.Tensor] = []
        self._buffer_tracking_ratio: list[torch.Tensor] = []
        self._last_low_count: int = 0
        self._last_high_count: int = 0
        self._last_low_ratio: float = 0.0
        self._last_high_ratio: float = 0.0
        self._last_evaluated: float = 0.0
        self._last_progressed: float = 0.0

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        del env_ids

    def __call__(
        self, env: ManagerBasedRlEnv, env_ids: Any, **kwargs
    ) -> dict[str, float]:
        del kwargs
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        telemetry = self._telemetry()
        if int(env.common_step_counter) == 0 or len(env_ids) == 0:
            return telemetry
        telemetry = self._collect_samples(env, env_ids, telemetry)
        if self._sample_count() < self._params.buffer_min:
            return telemetry
        return self._evaluate(env, telemetry)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": CURRICULUM_STATE_VERSION,
            "stage": self._stage,
            "robot": self._robot,
            "tracking_reward_term": self._reward_term_name,
            "vx_min": self._vx_min,
            "vx_max": self._vx_max,
            "ema_low": self._ema_low,
            "ema_high": self._ema_high,
            "pass_streak": self._pass_streak,
            "buffer_cmd_x": self._buffer(),
            "buffer_tracking_ratio": self._buffer(self._buffer_tracking_ratio),
        }

    def load_state_dict(self, state: dict[str, Any], mode: RestoreMode) -> None:
        if mode == "none":
            return
        if not isinstance(state, dict):
            raise ValueError(
                f"command curriculum state must be a dict, got {type(state)}"
            )
        state_version = state.get("version")
        if state_version not in (1, 2):
            raise ValueError(
                f"unsupported command curriculum state version: {state_version!r}"
            )
        if mode == "full":
            if state_version != CURRICULUM_STATE_VERSION:
                raise ValueError(
                    "incompatible curriculum statistics version; use range or none restore"
                )
            if state.get("tracking_reward_term") != self._reward_term_name:
                raise ValueError(
                    "command curriculum tracking reward mismatch; use range or none restore"
                )
        vx_min, vx_max = (float(state["vx_min"]), float(state["vx_max"]))
        self._validate_range(vx_min, vx_max)
        self._vx_min, self._vx_max = (vx_min, vx_max)
        self._apply_range()
        if mode == "range":
            self._reset_statistics()
            return
        self._ema_low = float(state["ema_low"])
        self._ema_high = float(state["ema_high"])
        self._pass_streak = int(state["pass_streak"])
        cmd_x = torch.as_tensor(state["buffer_cmd_x"], dtype=torch.float32).flatten()
        ratio = torch.as_tensor(
            state["buffer_tracking_ratio"], dtype=torch.float32
        ).flatten()
        if cmd_x.numel() != ratio.numel():
            raise ValueError(
                f"command curriculum buffer length mismatch: cmd_x {cmd_x.numel()} vs ratio {ratio.numel()}"
            )
        self._buffer_cmd_x = [cmd_x.cpu()]
        self._buffer_tracking_ratio = [ratio.cpu()]
        if not (torch.isfinite(cmd_x).all() and torch.isfinite(ratio).all()):
            raise ValueError("command curriculum buffer contains non-finite values")

    def reset_statistics(self) -> None:
        self._reset_statistics()

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def command_range(self) -> tuple[float, float]:
        return (self._vx_min, self._vx_max)

    def _reset_statistics(self) -> None:
        self._ema_low = 0.0
        self._ema_high = 0.0
        self._pass_streak = 0
        self._buffer_cmd_x = []
        self._buffer_tracking_ratio = []

    def _telemetry(self) -> dict[str, float]:
        return {
            "vx_min": self._vx_min,
            "vx_max": self._vx_max,
            "ema_low": self._ema_low,
            "ema_high": self._ema_high,
            "pass_streak": float(self._pass_streak),
            "buffer_count": float(self._sample_count()),
            "low_count": float(self._last_low_count),
            "high_count": float(self._last_high_count),
            "low_ratio": self._last_low_ratio,
            "high_ratio": self._last_high_ratio,
            "evaluated": self._last_evaluated,
            "progressed": self._last_progressed,
        }

    def _sample_count(self) -> int:
        return sum((int(chunk.numel()) for chunk in self._buffer_cmd_x))

    def _buffer(self, chunks: list[torch.Tensor] | None = None) -> torch.Tensor:
        chunks = self._buffer_cmd_x if chunks is None else chunks
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks).cpu()

    def _collect_samples(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, telemetry: dict[str, float]
    ) -> dict[str, float]:
        reward_manager = env.reward_manager
        weight = abs(float(reward_manager.get_term_cfg(self._reward_term_name).weight))
        if weight <= 0.0:
            self._pass_streak = 0
            return telemetry
        episode_steps = env.episode_length_buf[env_ids]
        valid_len = episode_steps > 0
        env_ids = env_ids[valid_len]
        if len(env_ids) == 0:
            return telemetry
        episode_steps = episode_steps[valid_len]
        episode_sum = reward_manager._episode_sums[self._reward_term_name][env_ids]
        ratios = (
            (episode_sum / (episode_steps.float() * env.step_dt * weight))
            .detach()
            .float()
            .cpu()
        )
        commands = env.command_manager.get_command(self._command_name)
        vx = commands[env_ids, 0].abs().detach().float().cpu()
        finite = torch.isfinite(ratios) & torch.isfinite(vx)
        if bool(finite.any()):
            self._buffer_cmd_x.append(vx[finite].clone())
            self._buffer_tracking_ratio.append(ratios[finite].clone())
        telemetry["buffer_count"] = float(self._sample_count())
        return telemetry

    def _evaluate(
        self, env: ManagerBasedRlEnv, telemetry: dict[str, float]
    ) -> dict[str, float]:
        p = self._params
        cmd_x = self._buffer()
        ratios = self._buffer(self._buffer_tracking_ratio)
        max_vx = max(abs(self._vx_min), abs(self._vx_max))
        split = p.low_high_split_ratio * max_vx
        low_mask = (cmd_x > p.low_speed_min) & (cmd_x <= split)
        high_mask = cmd_x > split
        low_count = int(low_mask.sum().item())
        high_count = int(high_mask.sum().item())
        telemetry["low_count"] = float(low_count)
        telemetry["high_count"] = float(high_count)
        telemetry["evaluated"] = 0.0
        telemetry["progressed"] = 0.0
        self._last_low_count = low_count
        self._last_high_count = high_count
        self._last_evaluated = 0.0
        self._last_progressed = 0.0
        if low_count < p.min_low_count or high_count < p.min_high_count:
            self._pass_streak = 0
            return telemetry
        low_ratio = float(ratios[low_mask].mean().item())
        high_ratio = float(ratios[high_mask].mean().item())
        telemetry["low_ratio"] = low_ratio
        telemetry["high_ratio"] = high_ratio
        telemetry["evaluated"] = 1.0
        self._last_low_ratio = low_ratio
        self._last_high_ratio = high_ratio
        self._last_evaluated = 1.0
        if not (math.isfinite(low_ratio) and math.isfinite(high_ratio)):
            self._pass_streak = 0
            return telemetry
        self._ema_low = (1.0 - p.ema_alpha) * self._ema_low + p.ema_alpha * low_ratio
        self._ema_high = (1.0 - p.ema_alpha) * self._ema_high + p.ema_alpha * high_ratio
        telemetry["ema_low"] = self._ema_low
        telemetry["ema_high"] = self._ema_high
        if self._ema_low > p.threshold_low and self._ema_high > p.threshold_high():
            self._pass_streak += 1
        else:
            self._pass_streak = 0
        telemetry["pass_streak"] = float(self._pass_streak)
        if self._pass_streak >= p.required_passes:
            new_min = round(max(self._vx_min - p.step, -p.max_abs_vx), 6)
            new_max = round(min(self._vx_max + p.step, p.max_abs_vx), 6)
            if (new_min, new_max) != (self._vx_min, self._vx_max):
                self._vx_min, self._vx_max = (new_min, new_max)
                self._apply_range()
                telemetry["vx_min"] = new_min
                telemetry["vx_max"] = new_max
                telemetry["progressed"] = 1.0
                self._last_progressed = 1.0
            self._pass_streak = 0
            telemetry["pass_streak"] = 0.0
        self._buffer_cmd_x = []
        self._buffer_tracking_ratio = []
        telemetry["buffer_count"] = 0.0
        return telemetry

    def _apply_range(self) -> None:
        term_cfg = self._env.command_manager.get_term(self._command_name).cfg
        term_cfg.ranges.lin_vel_x = (self._vx_min, self._vx_max)

    def _validate_range(self, vx_min: float, vx_max: float) -> None:
        p = self._params
        if vx_min > vx_max:
            raise ValueError(f"invalid vx range: ({vx_min}, {vx_max})")
        if vx_min < -p.max_abs_vx or vx_max > p.max_abs_vx:
            raise ValueError(
                f"vx range ({vx_min}, {vx_max}) exceeds curriculum max ±{p.max_abs_vx}"
            )


def get_command_curriculum_term(
    env: ManagerBasedRlEnv,
) -> ForwardSpeedCommandCurriculum | None:
    manager = getattr(env, "curriculum_manager", None)
    if manager is None or CURRICULUM_TERM_NAME not in getattr(
        manager, "active_terms", []
    ):
        return None
    term_cfg = manager.get_term_cfg(CURRICULUM_TERM_NAME)
    term = term_cfg.func
    if not isinstance(term, ForwardSpeedCommandCurriculum):
        raise TypeError(
            f"curriculum term 'command' is {type(term)}, expected ForwardSpeedCommandCurriculum"
        )
    return term
