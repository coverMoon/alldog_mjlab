"""Black 速度任务的 curriculum terms（task 层，算法无关）。

ForwardSpeedCommandCurriculum 是 super-dog Black 后期 performance-based
forward-speed command curriculum（``update_command_curriculum()``）在 MjLab v1.6.0
上的移植：

- 评价样本 = 每个完成 episode 的线速度 tracking ratio（§performance metric 见
  MIGRATION.md command curriculum 契约）；
- buffer 累计 >= ``buffer_min`` 且 low / high 组样本数都足够时才做一次 evaluation；
- EMA(0.2) 双双过线且连续 ``required_passes`` 次成功 evaluation 才把
  ``UniformVelocityCommandCfg.ranges.lin_vel_x`` 双边各扩 ``step``；
- 终态 range 为 [-max_abs_vx, +max_abs_vx]，达到后不再修改 range，只继续统计。

边界与不变量：

- 本 term 只改 ``ranges.lin_vel_x``；``lin_vel_y`` / ``ang_vel_z``、native sampler
  比例、resampling 间隔永不触碰；
- ``__call__`` 由 CurriculumManager 在 ``_reset_idx()`` 开头触发，此时
  episode sums / episode length / 当前 command 仍属于刚结束的 episode；
- play 模式下 curriculum 不注册（``_configure_play`` 清空 curriculum），
  因此 term 不存在，天然不更新。

私有 API 依赖（MjLab v1.6.0 pinned）：RewardManager 在 v1.6.0 中没有公开的
episode-sum accessor（仅有 ``get_term_cfg`` / ``active_terms`` 等接口），因此
本 term 最小访问 ``reward_manager._episode_sums``；升级 MjLab 时需核对该字段。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import math

import torch

from mjlab.managers import CurriculumTermCfg, ManagerTermBase

from alldog_mjlab.tasks.velocity.black.black_config import (
    BLACK_CONFIG,
    CommandCurriculumParams,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# state_dict 的版本号；load_state_dict 拒绝不支持的版本（fail-loud）。
CURRICULUM_STATE_VERSION = 1

# CurriculumManager 注册名：决定 TensorBoard 前缀 Curriculum/command/*。
CURRICULUM_TERM_NAME = "command"

# 恢复模式语义（见 runner 侧 resolve logic）：
#   none  —— 不恢复（新训练 / 旧 checkpoint fallback，保持初始范围与 fresh 统计）
#   range —— 只恢复 vx 范围（跨训练条件 / 跨 stage / PPO→HIM warm start）
#   full  —— 逐值恢复 range + EMA + streak + buffer（同 stage 精确续训）
# "auto" 仅用于 runner cfg 默认策略：按 source/target stage 元数据选择 full / range / none。
RestoreMode = Literal["none", "range", "full"]
RunnerRestoreMode = Literal["auto", "none", "range", "full"]


class ForwardSpeedCommandCurriculum(ManagerTermBase):
    """性能驱动的 forward-speed command 课程（stateful class-based term）。

    authoritative state：vx range、EMA、pass streak、pending buffer。
    telemetry cache（上次 evaluation 的计数 / ratio / evaluated / progressed）
    只是 TensorBoard 观测量，不是课程状态。

    buffer 保存在 CPU 上（float），避免 GPU 训练时的 device mismatch。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._params: CommandCurriculumParams = BLACK_CONFIG.command.command_curriculum
        self._params.validate()
        self._command_name: str = cfg.params["command_name"]
        # reward term 名固定为 Black tracking reward；通过 params 传入避免散落。
        self._reward_term_name: str = cfg.params["reward_term_name"]
        # 阶段元数据（flat | rough），来自 env cfg builder，用于 checkpoint
        # restore mode 判定；不由 run name / tensor shape 推测。
        self._stage: str = cfg.params["stage"]
        # authoritative runtime state。
        self._vx_min: float = self._params.initial_lin_vel_x[0]
        self._vx_max: float = self._params.initial_lin_vel_x[1]
        self._ema_low: float = 0.0
        self._ema_high: float = 0.0
        self._pass_streak: int = 0
        self._buffer_cmd_x: list[torch.Tensor] = []
        self._buffer_tracking_ratio: list[torch.Tensor] = []
        # telemetry cache（非 authoritative）。
        self._last_low_count: int = 0
        self._last_high_count: int = 0
        self._last_low_ratio: float = 0.0
        self._last_high_ratio: float = 0.0
        self._last_evaluated: float = 0.0
        self._last_progressed: float = 0.0

    # ------------------------------------------------------------------
    # Term lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        """Episode reset 不清课程状态：buffer / EMA / streak 跨 episode 累计。"""
        del env_ids

    def __call__(self, env: ManagerBasedRlEnv, env_ids: Any, **kwargs) -> dict[str, float]:
        """CurriculumManager.compute 入口（``_reset_idx`` 开头触发）。

        此时 episode sums / episode length / command 仍属于刚结束的 episode；
        读取后 RewardManager 才清零。返回 telemetry dict（float，无 NaN），
        由 CurriculumManager 记录为 ``Curriculum/command/*``。
        """
        del kwargs
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        telemetry = self._telemetry()
        # 首次 reset（common_step_counter == 0）：无 episode 样本，不收集、
        # 不更新 EMA、不增加 streak、不扩 range（初始 range 保持 [-1, 1]）。
        if int(env.common_step_counter) == 0 or len(env_ids) == 0:
            return telemetry
        telemetry = self._collect_samples(env, env_ids, telemetry)
        if self._sample_count() < self._params.buffer_min:
            return telemetry
        return self._evaluate(env, telemetry)

    # ------------------------------------------------------------------
    # State API（runner 只通过这里读写，不触碰内部字段）
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """authoritative state 序列化（CPU / 普通 Python 类型，存 checkpoint）。"""
        return {
            "version": CURRICULUM_STATE_VERSION,
            "stage": self._stage,
            "vx_min": self._vx_min,
            "vx_max": self._vx_max,
            "ema_low": self._ema_low,
            "ema_high": self._ema_high,
            "pass_streak": self._pass_streak,
            "buffer_cmd_x": self._buffer(),
            "buffer_tracking_ratio": self._buffer(self._buffer_tracking_ratio),
        }

    def load_state_dict(self, state: dict[str, Any], mode: RestoreMode) -> None:
        """按恢复模式装载课程状态。

        - full：range / EMA / streak / buffer 逐值恢复（同 stage 精确续训）；
        - range：只恢复 vx range，EMA / streak / buffer 清空
          （跨 stage / 跨训练条件 / PPO→HIM warm start）；
        - none：什么都不做（新训练保持初始 state）。

        buffer 至多保留 buffer_min 个样本（超过部分按先进先出截断，避免
        异常 checkpoint 膨胀）。
        """
        if mode == "none":
            return
        if not isinstance(state, dict):
            raise ValueError(f"command curriculum state must be a dict, got {type(state)}")
        if state.get("version") != CURRICULUM_STATE_VERSION:
            raise ValueError(
                "unsupported command curriculum state version: "
                f"{state.get('version')!r} (expected {CURRICULUM_STATE_VERSION})"
            )
        vx_min, vx_max = float(state["vx_min"]), float(state["vx_max"])
        self._validate_range(vx_min, vx_max)
        self._vx_min, self._vx_max = vx_min, vx_max
        self._apply_range()
        if mode == "range":
            # 跨训练条件：范围保留，统计重新开始。
            self._reset_statistics()
            return
        self._ema_low = float(state["ema_low"])
        self._ema_high = float(state["ema_high"])
        self._pass_streak = int(state["pass_streak"])
        cmd_x = torch.as_tensor(state["buffer_cmd_x"], dtype=torch.float32).flatten()
        ratio = torch.as_tensor(state["buffer_tracking_ratio"], dtype=torch.float32).flatten()
        if cmd_x.numel() != ratio.numel():
            raise ValueError(
                "command curriculum buffer length mismatch: "
                f"cmd_x {cmd_x.numel()} vs ratio {ratio.numel()}"
            )
        self._buffer_cmd_x = [cmd_x.cpu()]
        self._buffer_tracking_ratio = [ratio.cpu()]
        if not (torch.isfinite(cmd_x).all() and torch.isfinite(ratio).all()):
            raise ValueError("command curriculum buffer contains non-finite values")

    def reset_statistics(self) -> None:
        """统计 fresh（EMA / streak / buffer 清零），range 不变。"""
        self._reset_statistics()

    # ------------------------------------------------------------------
    # Properties（只读 telemetry / 元数据）
    # ------------------------------------------------------------------

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def command_range(self) -> tuple[float, float]:
        return (self._vx_min, self._vx_max)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _reset_statistics(self) -> None:
        self._ema_low = 0.0
        self._ema_high = 0.0
        self._pass_streak = 0
        self._buffer_cmd_x = []
        self._buffer_tracking_ratio = []

    def _telemetry(self) -> dict[str, float]:
        """telemetry dict：evaluated / progressed 属于本次调用，其余为缓存值。"""
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
        """buffer 内有效 episode 样本数（跨 chunk 的 numel 总和）。"""
        return sum(int(chunk.numel()) for chunk in self._buffer_cmd_x)

    def _buffer(self, chunks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """buffer 拼接为单个 CPU 1-D tensor（buffer 为空时返回空 tensor）。"""
        chunks = self._buffer_cmd_x if chunks is None else chunks
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks).cpu()

    def _collect_samples(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        telemetry: dict[str, float],
    ) -> dict[str, float]:
        """收集本批完成 episode 的 (|vx command|, tracking ratio) 样本。

        tracking_ratio = episode_tracking_reward_sum
            / (episode_length_steps × env.step_dt × |tracking_reward_weight|)

        理论范围 [0, 1]（浮点误差除外）；MjLab v1.6.0 的 RewardManager 保存的是
        raw × weight × step_dt 的 episode sum，因此按该公式还原 raw 均值。
        权重经 ``get_term_cfg`` 读取（不硬编码）；``_episode_sums`` 是 v1.6.0
        pinned private API（无公开 accessor，见文件头注释）。
        """
        reward_manager = env.reward_manager
        weight = abs(float(reward_manager.get_term_cfg(self._reward_term_name).weight))
        if weight <= 0.0:
            # 与 super-dog 一致：tracking 无效时课程判定失败（streak 清零）。
            self._pass_streak = 0
            return telemetry
        episode_steps = env.episode_length_buf[env_ids]
        # 无效 / 零长度 episode 不进入 buffer。
        valid_len = episode_steps > 0
        env_ids = env_ids[valid_len]
        if len(env_ids) == 0:
            return telemetry
        episode_steps = episode_steps[valid_len]
        episode_sum = reward_manager._episode_sums[self._reward_term_name][env_ids]
        ratios = (
            episode_sum
            / (episode_steps.float() * env.step_dt * weight)
        ).detach().float().cpu()
        # abs(current vx command)：episode 结束时刻的 body-frame 指令。
        commands = env.command_manager.get_command(self._command_name)
        vx = commands[env_ids, 0].abs().detach().float().cpu()
        finite = torch.isfinite(ratios) & torch.isfinite(vx)
        if bool(finite.any()):
            self._buffer_cmd_x.append(vx[finite].clone())
            self._buffer_tracking_ratio.append(ratios[finite].clone())
        telemetry["buffer_count"] = float(self._sample_count())
        return telemetry

    def _evaluate(self, env: ManagerBasedRlEnv, telemetry: dict[str, float]) -> dict[str, float]:
        """一次 low / high evaluation（buffer >= buffer_min 时触发）。

        super-dog Black 语义：
        - 组样本数不足 → 不更新 EMA、streak 清零、**保留 buffer**（等待更多样本，
          否则 high 组永远收集不满）；
        - 完成一次真正的 low/high evaluation（含 pass rule 与可能的 range 扩展）
          后清空本轮 buffer。
        """
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
            # 组不足：streak 清零、EMA 不动、buffer 保留（不清空）。
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
            # round 到 1e-6 避免 0.1 步长的浮点累积误差（[-1.2] vs -1.2000000000000002）。
            new_min = round(max(self._vx_min - p.step, -p.max_abs_vx), 6)
            new_max = round(min(self._vx_max + p.step, p.max_abs_vx), 6)
            # 达到最大范围后不能再扩展：数值不变，progressed 也不再置 1。
            if (new_min, new_max) != (self._vx_min, self._vx_max):
                self._vx_min, self._vx_max = new_min, new_max
                self._apply_range()
                telemetry["vx_min"] = new_min
                telemetry["vx_max"] = new_max
                telemetry["progressed"] = 1.0
                self._last_progressed = 1.0
            self._pass_streak = 0
            telemetry["pass_streak"] = 0.0
        # 完成一次真正的 evaluation：清空本轮 buffer。
        self._buffer_cmd_x = []
        self._buffer_tracking_ratio = []
        telemetry["buffer_count"] = 0.0
        return telemetry

    def _apply_range(self) -> None:
        """把 authoritative range 写回运行中的 UniformVelocityCommand cfg。

        只改 ``cfg.ranges.lin_vel_x``；下一次 command resample 生效，
        已采样、正在执行的 command 不变。
        """
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
    """从 env 找到已注册的 command curriculum term（play / 无 curriculum 时 None）。

    通过公开的 ``CurriculumManager.get_term_cfg`` 访问 class term 实例
    （``_resolve_common_term_cfg`` 会把 class term 的 func 替换为实例）。
    """
    manager = getattr(env, "curriculum_manager", None)
    if manager is None or CURRICULUM_TERM_NAME not in getattr(manager, "active_terms", []):
        return None
    term_cfg = manager.get_term_cfg(CURRICULUM_TERM_NAME)
    term = term_cfg.func
    if not isinstance(term, ForwardSpeedCommandCurriculum):
        raise TypeError(
            f"curriculum term 'command' is {type(term)}, expected "
            "ForwardSpeedCommandCurriculum"
        )
    return term
