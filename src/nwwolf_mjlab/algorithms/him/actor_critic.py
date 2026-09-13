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

import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Normal
from .estimator import get_activation
from .estimator import HIMEstimator

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class HIMActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_one_step_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        action_std_groups=None,
                        action_std_group_init_noise_std=None,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)

        self.history_size = int(num_actor_obs/num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        mlp_input_dim_a = num_one_step_obs + 3 + 16#16
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.std_group_indices = None
        self.std_group_names = None
        if action_std_groups is not None:
            if len(action_std_groups) == 0:
                raise ValueError("action_std_groups must not be empty")

            flat_indices = [idx for group in action_std_groups for idx in group]
            expected_indices = list(range(num_actions))
            if sorted(flat_indices) != expected_indices:
                raise ValueError(
                    f"action_std_groups must partition [0, {num_actions - 1}], got {sorted(flat_indices)}"
                )

            if action_std_group_init_noise_std is None:
                group_init_std = [float(init_noise_std)] * len(action_std_groups)
            else:
                if len(action_std_group_init_noise_std) != len(action_std_groups):
                    raise ValueError(
                        "action_std_group_init_noise_std must have the same length as action_std_groups"
                    )
                group_init_std = [float(value) for value in action_std_group_init_noise_std]

            self.std_group_indices = [list(group) for group in action_std_groups]
            self.std_group_names = [f"group_{i}" for i in range(len(action_std_groups))]
            self.std = nn.Parameter(torch.tensor(group_init_std, dtype=torch.float))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _expanded_std(self, reference_tensor):
        if self.std_group_indices is None:
            return self.std

        expanded_std = torch.empty(self.num_actions, device=reference_tensor.device, dtype=self.std.dtype)
        group_std = self.std.to(reference_tensor.device)
        for group_id, group_indices in enumerate(self.std_group_indices):
            expanded_std[group_indices] = group_std[group_id]
        return expanded_std

    def _remap_legacy_std_state(self, state_dict):
        if "std" not in state_dict:
            return state_dict, False

        loaded_std = state_dict["std"]
        if loaded_std.shape == self.std.shape:
            return state_dict, False

        remapped_state = OrderedDict(state_dict)
        if self.std_group_indices is not None and loaded_std.numel() == self.num_actions:
            group_std = []
            for group_indices in self.std_group_indices:
                group_std.append(loaded_std[group_indices].mean())
            remapped_state["std"] = torch.stack(group_std).to(
                device=loaded_std.device,
                dtype=loaded_std.dtype,
            )
            print(
                f"Remapped legacy action std from shape {tuple(loaded_std.shape)} "
                f"to grouped shape {tuple(remapped_state['std'].shape)}."
            )
            return remapped_state, True

        if loaded_std.numel() == 1:
            remapped_state["std"] = loaded_std.reshape(1).expand_as(self.std).clone()
            print(
                f"Broadcast action std from shape {tuple(loaded_std.shape)} "
                f"to shape {tuple(remapped_state['std'].shape)}."
            )
            return remapped_state, True

        return state_dict, False

    def load_state_dict(self, state_dict, strict=True):
        remapped_state, _ = self._remap_legacy_std_state(state_dict)
        return super().load_state_dict(remapped_state, strict=strict)

    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:,:self.num_one_step_obs], vel, latent), dim=-1)
        mean = self.actor(actor_input)
        std = self._expanded_std(mean)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        vel, latent = self.estimator(obs_history)
        actions_mean = self.actor(torch.cat((obs_history[:,:self.num_one_step_obs], vel, latent), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
