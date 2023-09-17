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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class PupperRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.15] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'leg_back_l_1': 0.0,   # [rad]
            'leg_back_l_2': 0.0,   # [rad]
            'leg_back_l_3': 0.0,  # [rad]

            'leg_front_l_1': 0.0,   # [rad]
            'leg_front_l_2': 0.0,   # [rad]
            'leg_front_l_3': 0.0,  # [rad]

            'leg_back_r_1': 0.0,   # [rad]
            'leg_back_r_2': 0.0,   # [rad]
            'leg_back_r_3': 0.0,  # [rad]

            'leg_front_r_1': 0.0,   # [rad]
            'leg_front_r_2': 0.0,   # [rad]
            'leg_front_r_3': 0.0,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'leg': 5.0}  # [N*m/rad]
        damping = {'leg': 0.1}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class commands( LeggedRobotCfg.commands ):
        num_commands = 4# default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True
        class ranges:
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            # lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw  = [-0.5, 0.5]    # min max [rad/s]
            ang_vel_yaw  =[1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/urdf/pupper_v3.urdf'
        name = "pupper"
        foot_name = "_3"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.1
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            # dof_pos_limits = -0.1
            # ang_vel_xy = -0.005
            # action_rate = -0.001
            # dof_acc = -2.5e-8
            # lin_vel_z = -0.1

class PupperRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_pupper'

  