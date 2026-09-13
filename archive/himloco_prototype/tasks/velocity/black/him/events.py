"""Reset, perturbations and performance-based command curriculum."""
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from dataclasses import dataclass
from mjlab.utils.lab_api.math import quat_apply
from .mdp import state


class BlackCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        xy = self.vel_command_b[env_ids, :2]
        self.vel_command_b[env_ids, :2] *= (xy.norm(dim=-1) > .2).unsqueeze(-1)


@dataclass(kw_only=True)
class BlackCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return BlackCommand(self, env)


def reset_joints(env, env_ids):
    robot = env.scene["robot"]
    pos = robot.data.default_joint_pos[env_ids] * torch.empty_like(robot.data.default_joint_pos[env_ids]).uniform_(.5, 1.5)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)


def perturbation(env, env_ids, **kwargs):
    del env_ids, kwargs
    s = state(env)
    # Legacy interval is in control steps, not seconds. One physics-step pulse.
    s.disturbance.zero_()
    if env.common_step_counter % 8 == 0:
        s.disturbance.uniform_(-30., 30.)
    # The action term clears the applied wrench after the first substep.
    world_force = quat_apply(s.robot.data.root_link_quat_w, s.disturbance)
    s.robot.data.write_external_wrench(world_force[:, None], None, body_ids=s.robot.find_bodies("trunk")[0])


class CommandCurriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.low = self.high = 0.
        self.streak = 0
        self.samples = []

    def __call__(self, env, env_ids):
        command = env.command_manager.get_term("twist")
        valid = env_ids[env.episode_length_buf[env_ids] > 0]
        if len(valid):
            sums = env.reward_manager._episode_sums["tracking_lin_vel"][valid]
            ratios = sums / (env.episode_length_buf[valid] * env.step_dt * 2.)
            self.samples.append(torch.stack((command.command[valid, 0].abs(), ratios), -1).detach())
        if sum(len(x) for x in self.samples) >= 256:
            values = torch.cat(self.samples)
            boundary = max(.2, .6*max(abs(x) for x in command.cfg.ranges.lin_vel_x))
            low = values[(values[:,0] > .2) & (values[:,0] <= boundary), 1]
            high = values[values[:,0] > boundary, 1]
            if len(low) >= 8 and len(high) >= 4:
                self.low = .8*self.low + .2*low.mean().item()
                self.high = .8*self.high + .2*high.mean().item()
                self.streak = self.streak+1 if self.low > .7 and self.high > .6 else 0
                if self.streak >= 2:
                    lo, hi = command.cfg.ranges.lin_vel_x
                    command.cfg.ranges.lin_vel_x = (max(-2., lo-.1), min(2., hi+.1))
                    self.streak = 0
                self.samples.clear()
            else:
                self.streak = 0
        return {"max_x": command.cfg.ranges.lin_vel_x[1], "low_ema": self.low, "high_ema": self.high}

from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.envs.mdp.dr._core import _randomize_model_field


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const_0)
def randomize_inertia(env, env_ids, asset_cfg):
    # MuJoCo stores principal inertias, unlike Isaac Gym's body-frame tensor.
    _randomize_model_field(env, env_ids, "body_inertia", entity_type="body", ranges=(.5,1.5), operation="scale", asset_cfg=asset_cfg)


def randomize_gains(env, env_ids):
    # Source draws one Kp factor and one Kd factor per robot, shared by joints.
    kp = torch.empty(len(env_ids), 1, device=env.device).uniform_(.8,1.2)
    kd = torch.empty_like(kp).uniform_(.8,1.2)
    for actuator in env.scene["robot"].actuators:
        actuator.set_gains(env_ids, kp=actuator.default_stiffness[env_ids]*kp, kd=actuator.default_damping[env_ids]*kd)
