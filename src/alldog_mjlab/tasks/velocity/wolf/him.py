"""Wolf HIM task-side observation / history / estimator target contract（task local）。

与 black/him.py 独立同构（interface contract 数值与语义完全一致，代码归属于
Wolf，不 import Black）：HIM algorithm integration 所需的 task-side 数据契约——

- actor observation 的 MjLab v1.6.0 原生 history（``[num_envs, 6, 53]``，
  MjLab 内部时间顺序为 oldest → newest）；
- estimator velocity 训练 target（scaled true base linear velocity，``[B, 3]``）；
- terminal successor target 捕获（done env 在 reset 覆盖前保存，含 terminal 前
  ``sim.forward()`` / ``sim.sense()`` 刷新）。

本模块**不**包含 HIM estimator / Sinkhorn / prototype loss / HIMPPO /
PPO→HIM warm start / TorchScript exporter（见 `alldog_mjlab.algorithms.him`）。

HIM group 名（``actor`` / ``estimator_velocity``）、extras key 与历史布局是
HIM 算法接口 contract，保持不变。维度不硬编码：``wolf_him_observation_spec()``
从已构造 env 的 runtime resolved dims 读取。
"""

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTermCfg,
)
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.noise import NoiseCfg

# ---------------------------------------------------------------------------
# HIM observation contract（interface contract，不属于训练调参）
# ---------------------------------------------------------------------------

# HIM actor history 长度（official HIM：6 帧）。
WOLF_HIM_HISTORY_LENGTH: int = 6

# 提供 history 的 observation group 名（复用普通 Wolf PPO actor terms）。
WOLF_HIM_ACTOR_GROUP: str = "actor"

# estimator velocity target group 名与维度。
WOLF_HIM_VELOCITY_GROUP: str = "estimator_velocity"
WOLF_HIM_VELOCITY_DIM: int = 3

# actor frame 前缀的 command 维度（Wolf command = vx / vy / wz）。
WOLF_HIM_COMMAND_DIM: int = 3

# estimator velocity target 数值尺度：official HIM ``obs_scales.lin_vel = 2.0``。
# 语义为 body-frame true base linear velocity（root_link_lin_vel_b）。
WOLF_HIM_VELOCITY_SCALE: float = 2.0

# terminal successor target recorder 名与 ``env.extras`` key（与 Black 一致；
# 仅供 HIM 算法 interface 使用，不同 task 的 env 实例互不共享 extras）。
WOLF_HIM_TERMINAL_RECORDER: str = "him_terminal"
WOLF_HIM_TERMINAL_IDS_KEY: str = "him_terminal_env_ids"
WOLF_HIM_TERMINAL_FRAME_KEY: str = "him_terminal_actor_frame"
WOLF_HIM_TERMINAL_VELOCITY_KEY: str = "him_terminal_estimator_velocity"


@dataclass(frozen=True)
class WolfHimObservationSpec:
    """HIM task-side 维度 contract（供 algorithm 层查询，避免散落 magic number）。

    ``history_dim`` / ``target_encoder_input_dim`` 一律由这里派生，不在算法内部
    出现 318 / 53 等字面量。
    """

    single_frame_dim: int
    history_length: int
    command_dim: int
    velocity_dim: int
    action_dim: int

    @property
    def history_dim(self) -> int:
        """MjLab task-side actor history 展平后的维度（history_length × single_frame_dim）。"""
        return self.history_length * self.single_frame_dim

    @property
    def target_encoder_input_dim(self) -> int:
        """HIM target encoder 逻辑输入维度（frame 去掉 command + velocity）。"""
        return (self.single_frame_dim - self.command_dim) + self.velocity_dim


def wolf_him_observation_spec(env: ManagerBasedRlEnv) -> WolfHimObservationSpec:
    """从已构造 env 读取 HIM 维度 contract（唯一来源是 runtime resolved dims）。"""
    actor_dim = env.observation_manager.group_obs_dim[WOLF_HIM_ACTOR_GROUP]
    assert isinstance(actor_dim, tuple) and len(actor_dim) == 2, actor_dim
    assert actor_dim[0] == WOLF_HIM_HISTORY_LENGTH, actor_dim
    velocity_dim = env.observation_manager.group_obs_dim[WOLF_HIM_VELOCITY_GROUP]
    assert isinstance(velocity_dim, tuple) and len(velocity_dim) == 1, velocity_dim
    assert velocity_dim[0] == WOLF_HIM_VELOCITY_DIM, velocity_dim
    return WolfHimObservationSpec(
        single_frame_dim=actor_dim[1],
        history_length=actor_dim[0],
        command_dim=WOLF_HIM_COMMAND_DIM,
        velocity_dim=velocity_dim[0],
        action_dim=env.action_manager.total_action_dim,
    )


def configure_him_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """在普通 Wolf env cfg 上加 HIM observation / history contract。

    actor group 保持当前 Wolf PPO 的全部 term（顺序、noise、scale、clip 不变），
    只在 group 级打开 MjLab 原生 history：

    - ``history_length = 6``，``flatten_history_dim = False``；
    - pipeline 仍为 compute → noise → clip → scale → history；
    - 输出为 frame-major 的 ``[B, 6, 53]``（MjLab 内部 oldest → newest），
      而不是 term-major。

    estimator velocity 是一个独立 group（``[B, 3]``，无 history / 无 noise），
    数值为 official HIM 的 scaled true base linear velocity。
    """
    actor_group = cfg.observations[WOLF_HIM_ACTOR_GROUP]
    assert actor_group.concatenate_terms
    assert actor_group.history_length is None, actor_group.history_length
    actor_group.history_length = WOLF_HIM_HISTORY_LENGTH
    actor_group.flatten_history_dim = False

    cfg.observations[WOLF_HIM_VELOCITY_GROUP] = ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(
                func=envs_mdp.base_lin_vel,
                scale=WOLF_HIM_VELOCITY_SCALE,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )


def configure_him_terminal_targets(cfg: ManagerBasedRlEnvCfg) -> None:
    """注册 terminal successor target recorder（done env 在 reset 覆盖前捕获）。"""
    cfg.recorders = {
        **cfg.recorders,
        WOLF_HIM_TERMINAL_RECORDER: RecorderTermCfg(func=HimTerminalTargets),
    }


def him_current_frame(actor_history: torch.Tensor) -> torch.Tensor:
    """task-side actor history 的当前帧（MjLab 内部 newest = index -1）。"""
    return actor_history[..., -1, :]


def canonical_him_history(actor_history: torch.Tensor) -> torch.Tensor:
    """MjLab task-side history → official HIM canonical frame-major history。

    MjLab 原生顺序为 oldest → newest（``[B, H, D]``）；official HIM / legacy
    deployment 期望 frame-major 且 newest → oldest（``[B, H × D]``）。
    """
    return actor_history.flip(1).flatten(1)


def him_target_encoder_input(
    actor_frame: torch.Tensor,
    estimator_velocity: torch.Tensor,
    spec: WolfHimObservationSpec,
) -> torch.Tensor:
    """HIM target encoder 逻辑输入（Wolf contract）。

    ``successor actor frame 去掉 command`` + ``successor scaled true base linear
    velocity``（Wolf 为 50 + 3 = 53 维）。显式由 spec 的 ``command_dim`` 切分，
    不做 magic slice。
    """
    return torch.cat((actor_frame[..., spec.command_dim :], estimator_velocity), dim=-1)


def compute_observation_frame(
    env: ManagerBasedRlEnv, group_name: str
) -> torch.Tensor:
    """按 ``ObservationManager`` 同一 pipeline 计算某 group 的当前单帧。

    pipeline 与 MjLab v1.6.0 ``ObservationManager.compute_group`` 一致
    （compute → noise → clip → scale），但不触碰 history / delay buffer，因此可以在
    reset 覆盖 done env 之前单独取得 terminal state 的当前帧。

    只用于 terminal target 捕获：正常 transition 的当前帧唯一来源仍是
    ``him_current_frame(obs_buf[actor])``。
    """
    manager = env.observation_manager
    frames = []
    for term_name in manager.active_terms[group_name]:
        term_cfg = manager.get_term_cfg(group_name, term_name)
        obs = term_cfg.func(env, **term_cfg.params).clone()
        if term_cfg.noise is not None:
            assert isinstance(term_cfg.noise, NoiseCfg), (
                f"{group_name}/{term_name}: NoiseModelCfg 需要 manager 内部实例，"
                "compute_observation_frame 尚未支持"
            )
            obs = term_cfg.noise.apply(obs)
        if term_cfg.clip:
            obs = obs.clip(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            obs = obs * term_cfg.scale
        frames.append(obs)
    return torch.cat(frames, dim=-1)


class HimTerminalTargets(RecorderTerm):
    """在 reset 覆盖 terminal state 之前，为 done env 捕获 estimator successor target。

    MjLab ``step()`` 顺序为 ``record_pre_reset`` → ``_reset_idx`` → observation
    compute，因此 done env 从 ``step()`` 返回的是**新 episode** 的第一帧，不能作为
    HIM 的 terminal successor target。这里在覆盖前从 terminal state 重新计算：

    - terminal successor current actor frame（``[n, 53]``，含与 actor group 一致的
      noise / scale / clip）；
    - terminal successor estimator velocity（``[n, 3]``）。

    noise 为独立重采样，与正常 transition 同分布，而不是同一个 draw（与 official
    HIM ``compute_termination_observations`` 一致）。

    ``step()`` 中 derived quantities 落后一个 physics substep，因此先
    ``sim.forward()`` + ``sim.sense()`` 使 terminal target 基于 terminal state 本身。

    捕获结果写入 ``env.extras``；没有 done env 的 step 结束时清除，避免 stale 数据
    被 algorithm 误用。
    """

    _TERMINAL_KEYS = (
        WOLF_HIM_TERMINAL_IDS_KEY,
        WOLF_HIM_TERMINAL_FRAME_KEY,
        WOLF_HIM_TERMINAL_VELOCITY_KEY,
    )

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        env = self._env
        env.sim.forward()
        env.sim.sense()
        frame = compute_observation_frame(env, WOLF_HIM_ACTOR_GROUP)
        velocity = compute_observation_frame(env, WOLF_HIM_VELOCITY_GROUP)
        env.extras[WOLF_HIM_TERMINAL_IDS_KEY] = env_ids.clone()
        env.extras[WOLF_HIM_TERMINAL_FRAME_KEY] = frame[env_ids].clone()
        env.extras[WOLF_HIM_TERMINAL_VELOCITY_KEY] = velocity[env_ids].clone()

    def record_post_step(self) -> None:
        if self._env.reset_buf.any():
            return
        for key in self._TERMINAL_KEYS:
            self._env.extras.pop(key, None)
