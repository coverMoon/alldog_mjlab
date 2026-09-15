"""Black's explicit PD/friction law and episode-constant control-step delay."""
from dataclasses import dataclass

import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.pd_actuator import IdealPdActuator, pd_torque
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from alldog_mjlab.robots.black.black_constants import get_spec
from .source_config import BlackCfg

LEGS = ("FL", "FR", "RR", "RL")
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))


@dataclass(kw_only=True)
class BlackPdCfg(IdealPdActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return BlackPd(self, entity, target_ids, target_names)


class BlackPd(IdealPdActuator):
    @staticmethod
    def control_law(params, cmd):
        torque = pd_torque(params["stiffness"], params["damping"], cmd)
        torque -= 0.35 * torch.tanh(3.0 * cmd.vel) + 0.1 * cmd.vel
        return torque.clamp(-params["force_limit"], params["force_limit"])


def him_spec():
    spec = get_spec()
    for key in list(spec.keys):
        spec.delete(key)
    # Source disables robot self-collisions. Preserve visual-only geoms.
    for index, geom in enumerate(spec.geoms):
        if geom.contype or geom.conaffinity:
            geom.name = f"collision_{index}"
            geom.contype = 1
            geom.conaffinity = 0
    return spec


def robot_cfg():
    return EntityCfg(
        spec_fn=him_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(BlackCfg.init_state.pos),
            joint_pos=dict(BlackCfg.init_state.default_joint_angles),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(BlackPdCfg(
                target_names_expr=(name,), stiffness=40.0,
                damping=BlackCfg.control.damping[name], effort_limit=20.0,
                # Friction is in the explicit law; do not apply XML friction twice.
                frictionloss=0.0, viscous_damping=0.0, armature=0.0,
            ) for name in JOINTS),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


@dataclass(kw_only=True)
class DelayedPositionActionCfg(JointPositionActionCfg):
    max_lag: int = 2

    def build(self, env):
        return DelayedPositionAction(self, env)


class DelayedPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        return self._entity.find_joints(JOINTS, preserve_order=True)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.substep = 0
        self.queue = torch.zeros(self.num_envs, cfg.max_lag + 1, self.action_dim, device=self.device)
        self.lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous = torch.zeros_like(self._raw_actions)
        self.previous_previous = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions):
        self.substep = 0
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(self._raw_actions)
        self._raw_actions.copy_(actions.clamp(-100.0, 100.0))
        self.queue[:, 1:] = self.queue[:, :-1].clone()
        self.queue[:, 0] = self._raw_actions
        delayed = self.queue[torch.arange(self.num_envs, device=self.device), self.lag]
        self._processed_actions = self._offset + 0.25 * delayed

    def apply_actions(self):
        if self.substep == 1:
            self._entity.data.write_external_wrench(torch.zeros(self.num_envs, len(self._entity.body_names), 3, device=self.device), None)
        self.substep += 1
        super().apply_actions()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self.queue[ids] = 0
        self.previous[ids] = 0
        self.previous_previous[ids] = 0
        self.lag[ids] = torch.randint(self.cfg.max_lag + 1, self.lag[ids].shape, device=self.device)
        self._processed_actions[ids] = self._offset[ids]
