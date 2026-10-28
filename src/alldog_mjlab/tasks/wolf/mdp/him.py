"""Wolf him interfaces."""

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RecorderTermCfg
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.noise import NoiseCfg

WOLF_HIM_HISTORY_LENGTH: int = 6
WOLF_HIM_ACTOR_GROUP: str = "actor"
WOLF_HIM_VELOCITY_GROUP: str = "estimator_velocity"
WOLF_HIM_VELOCITY_DIM: int = 3
WOLF_HIM_COMMAND_DIM: int = 3
WOLF_HIM_VELOCITY_SCALE: float = 2.0
WOLF_HIM_TERMINAL_RECORDER: str = "him_terminal"
WOLF_HIM_TERMINAL_IDS_KEY: str = "him_terminal_env_ids"
WOLF_HIM_TERMINAL_FRAME_KEY: str = "him_terminal_actor_frame"
WOLF_HIM_TERMINAL_VELOCITY_KEY: str = "him_terminal_estimator_velocity"


@dataclass(frozen=True)
class WolfHimObservationSpec:
    single_frame_dim: int
    history_length: int
    command_dim: int
    velocity_dim: int
    action_dim: int

    @property
    def history_dim(self) -> int:
        return self.history_length * self.single_frame_dim

    @property
    def target_encoder_input_dim(self) -> int:
        return self.single_frame_dim - self.command_dim + self.velocity_dim


def wolf_him_observation_spec(env: ManagerBasedRlEnv) -> WolfHimObservationSpec:
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
    actor_group = cfg.observations[WOLF_HIM_ACTOR_GROUP]
    assert actor_group.concatenate_terms
    assert actor_group.history_length is None, actor_group.history_length
    actor_group.history_length = WOLF_HIM_HISTORY_LENGTH
    actor_group.flatten_history_dim = False
    cfg.observations[WOLF_HIM_VELOCITY_GROUP] = ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(
                func=envs_mdp.base_lin_vel, scale=WOLF_HIM_VELOCITY_SCALE
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )


def configure_him_terminal_targets(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.recorders = {
        **cfg.recorders,
        WOLF_HIM_TERMINAL_RECORDER: RecorderTermCfg(func=HimTerminalTargets),
    }


def him_current_frame(actor_history: torch.Tensor) -> torch.Tensor:
    return actor_history[..., -1, :]


def canonical_him_history(actor_history: torch.Tensor) -> torch.Tensor:
    return actor_history.flip(1).flatten(1)


def him_target_encoder_input(
    actor_frame: torch.Tensor,
    estimator_velocity: torch.Tensor,
    spec: WolfHimObservationSpec,
) -> torch.Tensor:
    return torch.cat((actor_frame[..., spec.command_dim :], estimator_velocity), dim=-1)


def compute_observation_frame(env: ManagerBasedRlEnv, group_name: str) -> torch.Tensor:
    manager = env.observation_manager
    frames = []
    for term_name in manager.active_terms[group_name]:
        term_cfg = manager.get_term_cfg(group_name, term_name)
        obs = term_cfg.func(env, **term_cfg.params).clone()
        if term_cfg.noise is not None:
            assert isinstance(term_cfg.noise, NoiseCfg), (
                f"{group_name}/{term_name}: NoiseModelCfg 需要 manager 内部实例，compute_observation_frame 尚未支持"
            )
            obs = term_cfg.noise.apply(obs)
        if term_cfg.clip:
            obs = obs.clip(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            obs = obs * term_cfg.scale
        frames.append(obs)
    return torch.cat(frames, dim=-1)


class HimTerminalTargets(RecorderTerm):
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
