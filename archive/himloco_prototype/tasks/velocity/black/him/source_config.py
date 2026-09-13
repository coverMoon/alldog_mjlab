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

"""Snapshot of current HIMLoco Black config. No Isaac Gym dependency."""

class LeggedRobotCfg:

    class env:
        num_envs = 3400
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 12
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20

    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        slope_treshold = 0.75

    class commands:
        curriculum = True
        max_curriculum = 3.0
        curriculum_threshold = 0.8
        curriculum_ema_alpha = 0.1
        curriculum_required_passes = 1
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {'joint_a': 0.0, 'joint_b': 0.0}

    class control:
        control_type = 'P'
        stiffness = {'joint_a': 10.0, 'joint_b': 15.0}
        damping = {'joint_a': 1.0, 'joint_b': 1.5}
        action_scale = 0.5
        decimation = 4
        hip_reduction = 1.0

    class asset:
        file = ''
        name = 'legged_robot'
        foot_name = 'None'
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_kp = True
        kp_range = [0.9, 1.1]
        randomize_kd = True
        kd_range = [0.9, 1.1]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.0
        delay = True

    class rewards:

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -1e-05
            dof_vel = -0.0
            dof_acc = -2.5e-07
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 1.0
        max_contact_force = 100.0
        clearance_height_target = 0.09

    class normalization:

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5, 3.0]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class LeggedRobotCfgPPO:
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 0.001
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        learning_rate_min = 1e-05
        learning_rate_max = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100
        max_iterations = 200000
        save_interval = 20
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        resume_command_curriculum = 'range'

class BlackCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {'FL_hip_joint': 0.0, 'FL_thigh_joint': 0.8014, 'FL_calf_joint': -1.527, 'FR_hip_joint': -0.0, 'FR_thigh_joint': -0.8014, 'FR_calf_joint': 1.527, 'RL_hip_joint': 0.0, 'RL_thigh_joint': 0.8014, 'RL_calf_joint': -1.527, 'RR_hip_joint': -0.0, 'RR_thigh_joint': -0.8014, 'RR_calf_joint': 1.527}

    class control(LeggedRobotCfg.control):
        stiffness = {'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0, 'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0, 'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0}
        damping = {'FL_hip_joint': 1.2, 'RL_hip_joint': 1.2, 'FR_hip_joint': 1.2, 'RR_hip_joint': 1.2, 'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2, 'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black/black_description.urdf'
        name = 'black'
        foot_name = 'foot'
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = ['base', 'thigh']
        privileged_contacts_on = ['base', 'thigh', 'calf']
        self_collisions = 1

    class commands:
        curriculum = True
        max_curriculum = 2.0
        curriculum_threshold = 0.7
        curriculum_ema_alpha = 0.2
        curriculum_required_passes = 2
        curriculum_buffer_min = 256
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-2.0, 4.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = True
        link_mass_range = [0.75, 1.25]
        randomize_friction = True
        friction_range = [0.3, 1.35]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        randomize_inertia = True
        inertia_range = [0.5, 1.5]
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        push_robots = True
        push_interval_s = 30
        max_push_vel_xy = 2.5
        delay = True
        lag_timesteps = 3

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.08
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.1, 0.25, 0.25, 0.2, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.75

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 12
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        stuck_timeout_s = 4.0
        stuck_vel_threshold = 0.05
        stuck_command_threshold = 0.2
        stuck_grace_s = 1.0

    class rewards(LeggedRobotCfg.rewards):
        cycle_time = 0.8
        clearance_height_target = 0.08
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.8
        base_height_target = 0.43
        only_positive_rewards = False

        class terrain_adaptive:
            enabled = True
            terrain_variability_clip = 0.3

            class orientation:
                enabled = True
                mode = 'decay'
                sigma = 0.009
                min_scale = 0.15
                max_scale = 1.0

            class smoothness:
                enabled = True
                mode = 'decay'
                sigma = 0.2
                min_scale = 0.9
                max_scale = 1.0

            class action_rate:
                enabled = True
                mode = 'decay'
                sigma = 0.01
                min_scale = 0.2
                max_scale = 1.0

            class torques:
                enabled = False
                mode = 'decay'
                sigma = 0.05
                min_scale = 0.9
                max_scale = 1.0

            class progress:
                enabled = False
                mode = 'boost'
                sigma = 0.04
                min_scale = 1.0
                max_scale = 1.5

            class foot_clearance:
                enabled = True
                mode = 'margin'
                std_gain = 2.0
                max_extra_clearance = 0.15
                stance_gain = 0.5
                swing_high_penalty_weight = 0.25

        class raibert:
            nominal_front_x = 0.21
            nominal_rear_x = -0.21
            nominal_y = 0.155
            max_linear_offset_x = 0.16
            max_linear_offset_y = 0.06
            vel_error_gain = 0.3
            yaw_gain = 1.0
            max_yaw_offset = 0.1
            tracking_sigma = 0.06
            late_swing_start_x = 0.35
            late_swing_start_latyaw = 0.0
            touchdown_gain = 0.4
            approach_bonus = 0.25
            max_approach_speed = 0.4

        class scales:
            termination = -100.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5
            lin_vel_z = -1.5
            ang_vel_xy = -0.05
            orientation = -3.0
            dof_acc = -0.0
            joint_power = -1e-06
            base_height = -3.0
            foot_clearance = -10.0
            action_rate = -0.3
            smoothness = -0.01
            feet_air_time = 1.0
            collision = -0.05
            feet_stumble = -1.0
            stand_still = -1.0
            torques = -1e-07
            dof_vel = -1e-07
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -1e-05
            trot = 1.0
            hip_pos = -0.5
            all_joint_pos = -0.001
            foot_slip = -0.3
            foot_impact_vel = -10.0
            progress = 1.0
            raibert = 1.5

class BlackCfgPPO(LeggedRobotCfgPPO):

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 0.0001
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = True
        obs_permutation = [1e-05, -1, -2, -3, 4, -5, 6, -7, 8, -12, -13, -14, -9, -10, -11, -18, -19, -20, -15, -16, -17, -24, -25, -26, -21, -22, -23, -30, -31, -32, -27, -28, -29, -36, -37, -38, -33, -34, -35, -42, -43, -44, -39, -40, -41]
        act_permutation = [-3, -4, -5, -0.0001, -1, -2, -9, -10, -11, -6, -7, -8]
        frame_stack = 6
        sym_coef = 0.8

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        num_steps_per_env = 100
        experiment_name = 'rough_black_dog'
        max_iterations = 1000
