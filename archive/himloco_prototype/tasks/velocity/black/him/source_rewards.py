# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Active reward formulas ported from HIMLoco Black; state supplied by adapter."""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_rotate_inverse

class SourceRewards:

    def _init_raibert_buffers(self):
        """为 trot/clearance/Raibert 奖励建立稳定的脚名映射与名义落脚点。"""
        cfg = self.cfg.rewards.raibert
        num_feet = len(self.feet_indices)
        self.foot_name_to_index = {}
        self.foot_phase_offsets = torch.zeros(num_feet, device=self.device, requires_grad=False)
        self.nominal_foothold_xy = torch.zeros(num_feet, 2, device=self.device, requires_grad=False)
        for (i, foot_name) in enumerate(self.feet_names):
            leg_prefix = foot_name.split('_')[0]
            self.foot_name_to_index[leg_prefix] = i
            is_front = leg_prefix.startswith('F')
            is_left = leg_prefix.endswith('L')
            self.foot_phase_offsets[i] = 0.0 if leg_prefix in ('FL', 'RR') else 0.5
            self.nominal_foothold_xy[i, 0] = cfg.nominal_front_x if is_front else cfg.nominal_rear_x
            self.nominal_foothold_xy[i, 1] = cfg.nominal_y if is_left else -cfg.nominal_y

    def _get_feet_state_in_body_frame(self):
        """返回脚端相对机身的局部位置和速度。"""
        feet_rel_pos = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        feet_rel_vel = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        flat_base_quat = self.base_quat.unsqueeze(1).repeat(1, len(self.feet_indices), 1).view(-1, 4)
        feet_pos_body = quat_rotate_inverse(flat_base_quat, feet_rel_pos.reshape(-1, 3)).view(self.num_envs, len(self.feet_indices), 3)
        feet_vel_body = quat_rotate_inverse(flat_base_quat, feet_rel_vel.reshape(-1, 3)).view(self.num_envs, len(self.feet_indices), 3)
        return (feet_pos_body, feet_vel_body)

    def _get_phase(self):
        """ 
        内部辅助函数，计算相位
        仅用于计算奖励函数，不作为观测输入给网络
        """
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt % cycle_time / cycle_time
        return phase

    def _get_gait_phase(self):
        """
        根据相位生成理想的触地掩码 (Stance Mask)
        1 表示支撑相 (应触地)，0 表示摆动相 (应抬脚)
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        return stance_mask

    def _get_terrain_variability(self):
        cfg = self.cfg.rewards.terrain_adaptive
        if not cfg.enabled:
            return torch.zeros(self.num_envs, device=self.device)
        under_body_heights = self._get_under_body_height_samples()
        terrain_variability = torch.std(under_body_heights, dim=1)
        return torch.clamp(terrain_variability, min=0.0, max=cfg.terrain_variability_clip)

    def _get_adaptive_decay_scale(self, cfg_node, terrain_variability):
        if not (self.cfg.rewards.terrain_adaptive.enabled and cfg_node.enabled):
            return torch.ones_like(terrain_variability)
        scale = torch.exp(-torch.square(terrain_variability) / cfg_node.sigma)
        return torch.clamp(scale, min=cfg_node.min_scale, max=cfg_node.max_scale)

    def _get_clearance_margin(self, terrain_variability):
        cfg = self.cfg.rewards.terrain_adaptive.foot_clearance
        if not (self.cfg.rewards.terrain_adaptive.enabled and cfg.enabled):
            return torch.zeros(self.num_envs, 1, device=self.device)
        extra_clearance = cfg.std_gain * terrain_variability.unsqueeze(1)
        return torch.clamp(extra_clearance, min=0.0, max=cfg.max_extra_clearance)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        penalty = torch.square(self.base_lin_vel[:, 2])
        level_scale = torch.where(self.terrain_levels > 0, 0.1, 1.0)
        return penalty * level_scale

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        姿态惩罚
        利用特权观测的高度信息，在地形起伏大（如爬楼梯）时，
        降低对 Pitch (前后俯仰) 的惩罚，但保持 Roll (左右侧倾) 的严格惩罚。
        """
        pitch_proj = self.projected_gravity[:, 0]
        roll_proj = self.projected_gravity[:, 1]
        terrain_variability = self._get_terrain_variability()
        pitch_scale = self._get_adaptive_decay_scale(self.cfg.rewards.terrain_adaptive.orientation, terrain_variability)
        penalty = torch.abs(roll_proj) + torch.abs(pitch_proj) * pitch_scale
        return penalty

    def _reward_joint_power(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        base_height = self._get_base_heights()
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_foot_clearance(self):
        """
        [地形自适应的相位抬腿惩罚]
        平地时约束脚高贴近期望摆动轨迹；地形起伏增大时，放宽高抬脚惩罚，
        允许机器人为了跨台阶/障碍而抬得更高。
        """
        feet_height = self._get_feet_heights()
        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0
        sin_val = torch.sin(2 * torch.pi * feet_phases)
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        terrain_variability = self._get_terrain_variability()
        extra_clearance = self._get_clearance_margin(terrain_variability)
        clearance_cfg = self.cfg.rewards.terrain_adaptive.foot_clearance
        stance_tolerance = 0.02 + clearance_cfg.stance_gain * extra_clearance
        stance_penalty = torch.relu(feet_height - stance_tolerance)
        swing_target = -sin_val * self.cfg.rewards.clearance_height_target
        swing_low_penalty = torch.relu(swing_target - feet_height)
        swing_high_penalty = torch.relu(feet_height - (swing_target + extra_clearance))
        swing_penalty = swing_low_penalty + clearance_cfg.swing_high_penalty_weight * swing_high_penalty
        error = torch.where(sin_val > 0, stance_penalty, swing_penalty)
        return torch.sum(error, dim=1) * move_cmd.float()

    def _reward_action_rate(self):
        """地形复杂时适度放松一阶动作变化惩罚，给越障爆发留出空间。"""
        penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(self.cfg.rewards.terrain_adaptive.action_rate, terrain_variability)
        return penalty * scale

    def _reward_smoothness(self):
        """地形复杂时保留结构性平滑约束，但允许二阶动作变化更灵活。"""
        penalty = torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(self.cfg.rewards.terrain_adaptive.smoothness, terrain_variability)
        return penalty * scale

    def _reward_feet_air_time(self):
        """
        以目标步态周期为参考的腾空时间奖励。
        只在首次落地时结算，鼓励摆动腿完成完整的一步，但不鼓励无限制延长滞空时间。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.feet_air_time += self.dt
        target_air_time = self.cfg.rewards.cycle_time * 0.5
        min_air_time = target_air_time * 0.5
        first_contact = (self.feet_air_time > min_air_time) * contact_filt
        air_time_error = self.feet_air_time - target_air_time
        rew_air_time = torch.exp(-torch.square(air_time_error) / 0.01) * first_contact
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        rew_air_time = torch.sum(rew_air_time, dim=1) * move_cmd.float()
        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_collision(self):
        return torch.sum(1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        is_still = torch.norm(self.commands[:, :2], dim=1) < 0.1
        pos_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        vel_error = torch.sum(torch.abs(self.dof_vel), dim=1)
        error = pos_error + 0.05 * vel_error
        return error * is_still

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _reward_trot(self):
        """
        [Trot 步态引导奖励]
        鼓励对角线脚同时接触地面，且符合目标相位
        """
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        fl = contact_prob[:, self.foot_name_to_index['FL']]
        fr = contact_prob[:, self.foot_name_to_index['FR']]
        rl = contact_prob[:, self.foot_name_to_index['RL']]
        rr = contact_prob[:, self.foot_name_to_index['RR']]
        diag1_sync = 1.0 - torch.abs(fl - rr)
        diag2_sync = 1.0 - torch.abs(fr - rl)
        diag_sync = 0.5 * (diag1_sync + diag2_sync)
        s1 = 0.5 * (fl + rr)
        s2 = 0.5 * (fr + rl)
        stance_mask = self._get_gait_phase().float()
        (target_s1, target_s2) = (stance_mask[:, 0], stance_mask[:, 1])
        match_s1 = 1.0 - torch.abs(s1 - target_s1)
        match_s2 = 1.0 - torch.abs(s2 - target_s2)
        phase_match = 0.5 * (match_s1 + match_s2)
        rew = 0.4 * diag_sync + 0.6 * phase_match
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        rew = rew * move_cmd.float()
        return rew

    def _reward_hip_pos(self):
        """ 
        [髋关节限位惩罚]
        惩罚髋关节 (Hip/Abduction) 偏离默认角度的程度。
        防止机器人两腿张得太开 (劈叉) 或向内收得太多。
        """
        hip_indices = [0, 3, 6, 9]
        penalty = torch.sum(torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]), dim=1)
        vy = self.commands[:, 1]
        vw = self.commands[:, 2]
        is_straight_command = (torch.abs(vy) < 0.1) & (torch.abs(vw) < 0.1)
        scale = torch.where(is_straight_command, 1.0, 0.2)
        return scale * penalty

    def _reward_all_joint_pos(self):
        """
        [所有关节限位惩罚]
        惩罚所有关节偏离默认角度的程度
        防止动作变形
        """
        return torch.sum(torch.square(self.dof_pos[:, :] - self.default_dof_pos[:, :]), dim=1)

    def _reward_foot_slip(self):
        """
        [脚底打滑惩罚]
        触地时如果脚有水平速度则惩罚
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(rew, dim=1)

    def _reward_foot_impact_vel(self):
        """
        只在首次触地时惩罚过大的向下落地速度。
        小的正常落地速度通过安全阈值过滤，避免把接触后的微小振动也算作冲击。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = contact & ~self.last_impact_contacts
        self.last_impact_contacts = contact
        impact_vel = torch.clamp(-self.feet_vel[:, :, 2] - 0.2, min=0.0)
        return torch.sum(torch.square(impact_vel) * first_contact.float(), dim=1)

    def _reward_progress(self):
        """
        轻量的命令方向进展奖励。
        只鼓励沿当前平移指令方向的正向速度，避免台阶前“停住保平衡”。
        """
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        move_cmd = cmd_norm > 0.1
        cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-06)
        progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
        bounded_progress = torch.minimum(torch.relu(progress_speed), cmd_norm)
        return bounded_progress * move_cmd.float()

    def _reward_raibert(self):
        """
        [Raibert 落脚点奖励]
        将当前命令按可用指令范围归一化，再映射到有界的目标落脚点偏移。
        这样即使速度课程继续放大，奖励给出的目标点也不会无限前冲。
        """
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        if not torch.any(move_cmd).item():
            return torch.zeros(self.num_envs, device=self.device)
        cfg = self.cfg.rewards.raibert
        (foot_pos_body, foot_vel_body) = self._get_feet_state_in_body_frame()
        cmd_limits = self.commands.new_tensor([max(abs(self.command_ranges['lin_vel_x'][0]), abs(self.command_ranges['lin_vel_x'][1]), 1e-06), max(abs(self.command_ranges['lin_vel_y'][0]), abs(self.command_ranges['lin_vel_y'][1]), 1e-06)]).view(1, 1, 2)
        cmd_xy_norm = torch.clamp(self.commands[:, :2].unsqueeze(1) / cmd_limits, min=-1.0, max=1.0)
        vel_error_norm = torch.clamp((self.commands[:, :2] - self.base_lin_vel[:, :2]).unsqueeze(1) / cmd_limits, min=-1.0, max=1.0)
        linear_drive = torch.clamp(cmd_xy_norm + cfg.vel_error_gain * vel_error_norm, min=-1.0, max=1.0)
        max_linear_offset = self.commands.new_tensor([cfg.max_linear_offset_x, cfg.max_linear_offset_y]).view(1, 1, 2)
        target_xy = self.nominal_foothold_xy.unsqueeze(0) + linear_drive * max_linear_offset
        yaw_limit = max(abs(self.command_ranges['ang_vel_yaw'][0]), abs(self.command_ranges['ang_vel_yaw'][1]), 1e-06)
        yaw_norm = torch.clamp(self.commands[:, 2].view(self.num_envs, 1, 1) / yaw_limit, min=-1.0, max=1.0)
        yaw_basis = torch.stack((-self.nominal_foothold_xy[:, 1], self.nominal_foothold_xy[:, 0]), dim=1)
        yaw_basis = yaw_basis / torch.clamp(torch.norm(yaw_basis, dim=1, keepdim=True), min=1e-06)
        target_xy = target_xy + cfg.max_yaw_offset * cfg.yaw_gain * yaw_norm * yaw_basis.unsqueeze(0)
        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0
        swing_progress = torch.clamp((feet_phases - 0.5) * 2.0, min=0.0, max=1.0)
        late_swing_start_x = getattr(cfg, 'late_swing_start_x', getattr(cfg, 'late_swing_start', 0.35))
        late_swing_start_latyaw = getattr(cfg, 'late_swing_start_latyaw', getattr(cfg, 'late_swing_start', 0.15))
        late_swing_x = torch.clamp((swing_progress - late_swing_start_x) / max(1e-06, 1.0 - late_swing_start_x), min=0.0, max=1.0)
        late_swing_latyaw = torch.clamp((swing_progress - late_swing_start_latyaw) / max(1e-06, 1.0 - late_swing_start_latyaw), min=0.0, max=1.0)
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        touchdown_weight = 1.0 + cfg.touchdown_gain * contact_prob
        planning_weight_x = late_swing_x * touchdown_weight
        planning_weight_latyaw = late_swing_latyaw * touchdown_weight
        nominal_xy = self.nominal_foothold_xy.unsqueeze(0)
        x_offset = linear_drive[:, :, 0:1] * cfg.max_linear_offset_x
        lateral_offset = torch.cat((torch.zeros_like(x_offset), linear_drive[:, :, 1:2] * cfg.max_linear_offset_y), dim=2)
        yaw_offset = cfg.max_yaw_offset * cfg.yaw_gain * yaw_norm * yaw_basis.unsqueeze(0)
        x_target = nominal_xy[:, :, 0:1] + x_offset
        x_error = foot_pos_body[:, :, 0:1] - x_target
        tracking_reward_x = torch.exp(-torch.square(x_error.squeeze(2)) / max(cfg.tracking_sigma ** 2, 1e-06))
        latyaw_target = lateral_offset + yaw_offset
        latyaw_actual = torch.cat((foot_pos_body[:, :, 0:1] - x_target, foot_pos_body[:, :, 1:2] - nominal_xy[:, :, 1:2]), dim=2)
        latyaw_error = latyaw_actual - latyaw_target
        tracking_reward_latyaw = torch.exp(-torch.sum(torch.square(latyaw_error), dim=2) / max(cfg.tracking_sigma ** 2, 1e-06))
        target_dir_x = torch.sign(x_target - foot_pos_body[:, :, 0:1])
        approach_speed_x = (foot_vel_body[:, :, 0:1] * target_dir_x).squeeze(2)
        approach_bonus_x = torch.clamp(approach_speed_x, min=0.0, max=cfg.max_approach_speed) / max(cfg.max_approach_speed, 1e-06)
        target_dir_latyaw = latyaw_target - latyaw_actual
        target_dir_latyaw = target_dir_latyaw / torch.clamp(torch.norm(target_dir_latyaw, dim=2, keepdim=True), min=1e-06)
        approach_speed_latyaw = torch.sum(foot_vel_body[:, :, :2] * target_dir_latyaw, dim=2)
        approach_bonus_latyaw = torch.clamp(approach_speed_latyaw, min=0.0, max=cfg.max_approach_speed) / max(cfg.max_approach_speed, 1e-06)
        reward_x = planning_weight_x * (tracking_reward_x + cfg.approach_bonus * approach_bonus_x)
        reward_latyaw = planning_weight_latyaw * (tracking_reward_latyaw + cfg.approach_bonus * approach_bonus_latyaw)
        reward = 0.5 * (reward_x + reward_latyaw)
        return torch.sum(reward, dim=1) * move_cmd.float()
