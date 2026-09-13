"""MuJoCo state adapter for the source Black reward and observation contract."""
import torch
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.lab_api.math import quat_apply
from .control import JOINTS, LEGS
from .source_config import BlackCfg
from .source_rewards import SourceRewards


class BlackState(SourceRewards):
    def __init__(self, env):
        self.env = env
        self.cfg = BlackCfg
        self.robot = env.scene["robot"]
        self.num_envs, self.device, self.dt = env.num_envs, env.device, env.step_dt
        self.joint_ids = self.robot.find_joints(JOINTS, preserve_order=True)[0]
        self.feet_names = [f"{leg}_foot" for leg in LEGS]
        self.feet_indices = self.robot.find_bodies(self.feet_names, preserve_order=True)[0]
        self.penalised_contact_indices = self.robot.find_bodies(("trunk", ".*_thigh", ".*_calf"))[0]
        self.termination_ids = self.robot.find_bodies(("trunk", ".*_thigh"))[0]
        self._init_raibert_buffers()
        self.history = torch.zeros(self.num_envs, 6, 45, device=self.device)
        self.current = torch.zeros(self.num_envs, 45, device=self.device)
        self.obs_tick = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.stuck_tick = -1
        self.refresh_tick = -1
        self.stuck_time = torch.zeros(self.num_envs, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        self.last_impact_contacts = self.last_contacts.clone()
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.disturbance = torch.zeros(self.num_envs, 3, device=self.device)
        self.noise = torch.tensor([0.] * 3 + [0.075] * 3 + [0.05] * 3 + [0.08] * 12 + [0.1] * 12 + [0.] * 12, device=self.device)
        sensor = env.scene["body_contact"]
        self.contact_body_ids = self.robot.find_bodies(tuple(sensor.primary_names), preserve_order=True)[0]
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.torque_limits = torch.full((12,), 20., device=self.device)
        self.dof_vel_limits = torch.full((12,), 25., device=self.device)

    def refresh(self, force=False):
        if not force and self.refresh_tick == self.env._sim_step_counter:
            return
        self.refresh_tick = self.env._sim_step_counter
        data = self.robot.data
        self.episode_length_buf = self.env.episode_length_buf
        self.base_quat = data.root_link_quat_w
        self.root_states = torch.cat((data.root_link_pose_w, data.root_link_vel_w), -1)
        self.base_lin_vel = data.root_link_lin_vel_b
        self.base_ang_vel = data.root_link_ang_vel_b
        self.projected_gravity = data.projected_gravity_b
        self.dof_pos = data.joint_pos[:, self.joint_ids]
        self.dof_vel = data.joint_vel[:, self.joint_ids]
        self.default_dof_pos = data.default_joint_pos[:, self.joint_ids]
        self.torques = data.qfrc_actuator[:, self.joint_ids]
        action = self.env.action_manager.get_term("joint_pos")
        self.actions = action.raw_action
        self.last_actions = action.previous
        self.last_last_actions = action.previous_previous
        self.feet_pos = data.body_link_pos_w[:, self.feet_indices]
        self.feet_vel = data.body_link_lin_vel_w[:, self.feet_indices]
        self.rigid_state = torch.cat((data.body_link_pose_w, data.body_link_vel_w), -1)
        self.commands = self.env.command_manager.get_command("twist")
        ranges = self.env.command_manager.get_term("twist").cfg.ranges
        self.command_ranges = {"lin_vel_x": ranges.lin_vel_x, "lin_vel_y": ranges.lin_vel_y, "ang_vel_yaw": ranges.ang_vel_z}
        self.contact_forces = torch.zeros(self.num_envs, len(self.robot.body_names), 3, device=self.device)
        self.contact_forces[:, self.contact_body_ids] = self.env.scene["body_contact"].data.force
        terrain = self.env.scene.terrain
        self.terrain_levels = terrain.terrain_levels if hasattr(terrain, "terrain_levels") else torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = getattr(self.env, "reset_buf", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.time_out_buf = getattr(self.env, "reset_time_outs", torch.zeros_like(self.reset_buf))

    def frame(self):
        self.refresh(force=True)
        changed = self.obs_tick != self.env.common_step_counter
        if changed.any():
            current = torch.cat((self.commands * self.commands.new_tensor([2., 2., .25]), self.base_ang_vel * .25, self.projected_gravity, self.dof_pos-self.default_dof_pos, self.dof_vel*.05, self.actions), -1)
            if self.env.cfg.observations["actor"].enable_corruption:
                current += (2 * torch.rand_like(current) - 1) * self.noise
            self.current[changed] = current[changed].clamp(-100, 100)
            self.history[changed, 1:] = self.history[changed, :-1].clone()
            self.history[changed, 0] = self.current[changed]
            self.obs_tick[changed] = self.env.common_step_counter
        return self.current

    def critic(self):
        frame = self.frame()
        heights = ((self.root_states[:, 2:3] - .5 - self._terrain_heights("terrain_scan")).clamp(-1, 1) * 5)
        if self.env.cfg.observations["actor"].enable_corruption:
            heights += torch.empty_like(heights).uniform_(-.5, .5)
        return torch.cat((frame, self.base_lin_vel*2, self.disturbance, heights), -1).clamp(-100, 100)

    def _terrain_heights(self, name):
        data = self.env.scene[name].data
        z = data.hit_pos_w[..., 2]
        return torch.where(data.distances >= 0, z, torch.zeros_like(z))

    def _get_under_body_height_samples(self):
        return self._terrain_heights("body_height_scan")

    def _get_base_heights(self):
        return self.root_states[:, 2] - self._get_under_body_height_samples().mean(-1)

    def _get_feet_heights(self):
        return self.env.scene["foot_height_scan"].data.heights

    def reset(self, ids):
        self.refresh_tick = -1
        for x in (self.history, self.current, self.stuck_time, self.last_contacts, self.last_impact_contacts, self.feet_air_time, self.disturbance):
            x[ids] = 0
        self.obs_tick[ids] = -1


def state(env):
    if not hasattr(env, "_him_state"):
        env._him_state = BlackState(env)
    return env._him_state


def actor_observation(env):
    s = state(env)
    s.frame()
    return s.history.flatten(1).clone()


def critic_observation(env):
    return state(env).critic()


def reward(env, name):
    if name == "termination":
        return env.reset_terminated.float()
    s = state(env)
    s.refresh()
    return getattr(s, "_reward_" + name)()


def illegal_contact(env):
    s = state(env)
    s.refresh()
    return (s.contact_forces[:, s.termination_ids].norm(dim=-1) > 1.).any(-1)


def stuck(env):
    s = state(env)
    s.refresh()
    if s.stuck_tick != env.common_step_counter:
        norm = s.commands[:, :2].norm(dim=-1)
        speed = (s.base_lin_vel[:, :2] * s.commands[:, :2] / norm[:, None].clamp(min=1e-6)).sum(-1)
        mask = (norm > .2) & (speed < .05) & (env.episode_length_buf * env.step_dt > 1.)
        s.stuck_time = torch.where(mask, s.stuck_time + env.step_dt, 0.)
        s.stuck_tick = env.common_step_counter
    return s.stuck_time > 4.


def reset_state(env, env_ids):
    if hasattr(env, "_him_state"):
        state(env).reset(env_ids)


class TerminalObservations(RecorderTerm):
    def record_pre_reset(self, env_ids):
        # The built-in obs_buf still refers to the preceding step here.
        self._env.sim.forward()
        self._env.sim.sense()
        self._env.extras["him_terminal_ids"] = env_ids.clone()
        self._env.extras["him_terminal_critic"] = state(self._env).critic()[env_ids].clone()

    def record_post_step(self):
        if not self._env.reset_buf.any():
            self._env.extras.pop("him_terminal_ids", None)
            self._env.extras.pop("him_terminal_critic", None)
