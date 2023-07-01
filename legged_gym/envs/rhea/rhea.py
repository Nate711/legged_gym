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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rhea_config import RheaRoughCfg

NUM_ACTUATORS = 4
NUM_PHYSICAL_JOINTS = 6
KNEE_TO_HIP_TORQUE_RATIO = -0.5

MIN_LEG_POS = -0.35
MAX_LEG_POS = -0.15
MAX_WHEEL_VEL = 30.0
ACTION_MIN = [MIN_LEG_POS, -MAX_WHEEL_VEL, MIN_LEG_POS, -MAX_WHEEL_VEL]
ACTION_MAX = [MAX_LEG_POS, MAX_WHEEL_VEL, MAX_LEG_POS, MAX_WHEEL_VEL]

ACTION_SCALE_POS = 1.0
ACTION_SCALE_VEL = 1.0

ACTION_SCALES = [ACTION_SCALE_POS, ACTION_SCALE_VEL, ACTION_SCALE_POS, ACTION_SCALE_VEL]
DEFAULT_DOF_POS = torch.tensor([-0.3, 0.0, -0.3, 0.0], dtype=torch.float, device="cuda")

ACTION_MIN = torch.tensor(ACTION_MIN, dtype=torch.float, device="cuda")
ACTION_MAX = torch.tensor(ACTION_MAX, dtype=torch.float, device="cuda")

ACTION_SCALES = torch.tensor(ACTION_SCALES, dtype=torch.float, device="cuda")

RESET_PROJECTED_GRAVITY_Z = np.cos(0.52) # Pitch/roll angle to trigger resets

PITCH_OFFSET_RANGE = [0.0, 0.0] #[-0.05, 0.05]

class Rhea(LeggedRobot):
    cfg : RheaRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.position_joints = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.velocity_joints = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.pitch_offsets = torch_rand_float(PITCH_OFFSET_RANGE[0], PITCH_OFFSET_RANGE[1], (self.num_envs, 1), device=self.device)#.squeeze(1)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            for dof_name in self.cfg.control.joint_control_types.keys():
                if dof_name in name:
                    if self.cfg.control.joint_control_types[dof_name] == "P":
                        self.position_joints[i] = 1.0
                    elif self.cfg.control.joint_control_types[dof_name] == "V":
                        self.velocity_joints[i] = 1.0

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= torch.abs(self.projected_gravity[:, 2]) < RESET_PROJECTED_GRAVITY_Z

    def compute_observations(self):
        """ Computes observations
        """
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions
        #                             ),dim=-1)

        actions_clipped = torch.clamp(self.actions, (ACTION_MIN - DEFAULT_DOF_POS) / ACTION_SCALES, (ACTION_MAX - DEFAULT_DOF_POS) / ACTION_SCALES)
        # actions_clipped = self.actions

        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    torch.cos(self.pitch_offsets) * self.projected_gravity[:, :1] - torch.sin(self.pitch_offsets) * self.projected_gravity[:, 2:],
                                    self.projected_gravity[:, 1:2],
                                    torch.sin(self.pitch_offsets) * self.projected_gravity[:, :1] + torch.cos(self.pitch_offsets) * self.projected_gravity[:, 2:],
                                    self.commands[:, :3] * self.commands_scale,
                                    self.position_joints * (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    actions_clipped
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _process_rigid_body_props(self, props, env_id):
            if self.cfg.domain_rand.randomize_base_mass:
                rng_mass = self.cfg.domain_rand.added_mass_range
                rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
                props[0].mass += rand_mass
            else:
                rand_mass = np.zeros(1)
            if self.cfg.domain_rand.randomize_gripper_mass:
                gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
                gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
                props[self.gripper_idx].mass += gripper_rand_mass
            else:
                gripper_rand_mass = np.zeros(1)
            if self.cfg.domain_rand.randomize_base_com:
                rng_com_x = self.cfg.domain_rand.added_com_range_x
                rng_com_y = self.cfg.domain_rand.added_com_range_y
                rng_com_z = self.cfg.domain_rand.added_com_range_z
                rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
                props[0].com += gymapi.Vec3(*rand_com)
            else:
                rand_com = np.zeros(3)
            mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass])
            return props

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:13] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[13:17] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[17:21] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        # sum up the number of contacts (each foot should have at least one)
        both_feet_contact = torch.sum(1.*contacts, dim=1) == 2

        # return 1 if both feet have contact, 0 otherwise
        return 1.*both_feet_contact

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.position_joints * (self.dof_pos - self.default_dof_pos), dim=1)
        return torch.square(base_height)
    
    def _reward_alive(self):
        return 1.0

    def _compute_torques(self, actions):
        
        # actions_clipped = actions
        actions_clipped = torch.clamp(actions, (ACTION_MIN - DEFAULT_DOF_POS) / ACTION_SCALES, (ACTION_MAX - DEFAULT_DOF_POS) / ACTION_SCALES)

        pos_actions_scaled = self.position_joints * actions_clipped * self.cfg.control.action_scale
        # actions_scaled = actions_scaled @ self.actuation_matrix
        velocity_actions_scaled = self.velocity_joints * actions_clipped * self.cfg.control.velocity_action_scale

        actions_scaled  = pos_actions_scaled + velocity_actions_scaled

        # print(f'actions_scaled: {actions_scaled.size()}, default dof: {self.default_dof_pos.size()}, dgains: {self.d_gains.size()}, dof vel: {self.dof_vel.size()}')
        actuator_position_torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        actuator_position_torques = self.position_joints * actuator_position_torques

        # velocity_torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        actuator_velocity_torques = self.d_gains*(velocity_actions_scaled - self.dof_vel)
        actuator_velocity_torques = self.velocity_joints * actuator_velocity_torques

        actuator_torques = actuator_position_torques + actuator_velocity_torques
        actuator_torques = torch.clip(actuator_torques, -self.torque_limits, self.torque_limits)

        return actuator_torques

