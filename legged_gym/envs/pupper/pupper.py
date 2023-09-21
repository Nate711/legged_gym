
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
from .pupper_config import PupperRoughCfg

class Pupper(LeggedRobot):
    cfg : PupperRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_camera()

    def compute_observations(self):
        super().compute_observations()
        self.obs_buf[:, :3] *= 0
        self._get_depth_maps()

    def _init_camera(self):
        self.num_pixel_directions = self.cfg.camera.h_res * self.cfg.camera.v_res
        pixel_directions = torch.zeros((1, self.num_pixel_directions, 3), dtype=torch.float32)
        for i in range(self.cfg.camera.h_res):
            for j in range(self.cfg.camera.v_res):
                # Horizontal component (robot y-axis)
                pixel_directions[0, i*self.cfg.camera.v_res+j, 1] = np.tan(self.cfg.camera.h_fov/2.0) * (i - self.cfg.camera.h_res/2.0) / self.cfg.camera.h_res
                
                # Vertical component (robot z-axis)
                pixel_directions[0, i*self.cfg.camera.v_res+j, 2] = np.tan(self.cfg.camera.v_fov/2.0) * (j - self.cfg.camera.v_res/2.0) / self.cfg.camera.v_res

                # Depth component (robot x-axis)
                pixel_directions[0, i*self.cfg.camera.v_res+j, 0] = 1.0
        
        self.pixel_directions = pixel_directions.to(self.device).repeat(self.num_envs, 1, 1)
        self.pixel_depths = torch.zeros((self.num_envs, self.num_pixel_directions), dtype=torch.float32, device=self.device)

        try:
            self.cone_map = torch.load(os.path.join(LEGGED_GYM_ROOT_DIR, 'cone_map.pt'))
        except:
            # Generate the cone map
            # For each pixel of the terrain map, calculate the slope of the line to every other pixel, and store the maximum slope
            H, W = self.height_samples.shape
            y_coords = torch.arange(W, dtype=torch.int, device=self.device)
            x_coords = torch.arange(H, dtype=torch.int, device=self.device)
            xx, yy = torch.meshgrid(x_coords, y_coords)

            self.cone_map = torch.zeros_like(self.height_samples).float()

            # Loop through each pixel in the original map
            for i in range(self.height_samples.shape[0]):
                # print(i)
                print("Generating cone map: {:.2f}%".format(100.0 * i / self.height_samples.shape[0]))
                for j in range(self.height_samples.shape[1]):
                    # Calculate the slope to each pixel
                    height_diffs = self.height_samples - self.height_samples[i, j]
                    x_diffs = i - xx
                    y_diffs = j - yy
                    distances = torch.sqrt(x_diffs**2 + y_diffs**2)
                    distances[distances == 0] = 1
                    slopes = height_diffs / distances
                    self.cone_map[i, j] = torch.max(slopes)

            self.cone_map *= self.terrain.cfg.vertical_scale / self.terrain.cfg.horizontal_scale

            torch.save(self.cone_map, os.path.join(LEGGED_GYM_ROOT_DIR, 'cone_map.pt'))

    def _get_depth_maps(self, env_ids=None):
        """ Renders the depth map as seen from each robot's camera.
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """

        if env_ids:
            world_pixel_directions = quat_apply(self.base_quat[env_ids].repeat(1, self.num_pixel_directions), self.pixel_directions[env_ids])
            pixel_sample_points = (self.root_states[env_ids, :3]).unsqueeze(1) + self.cfg.camera.near * world_pixel_directions
            self.pixel_depths[env_ids, :] = self.cfg.camera.near
        else:
            world_pixel_directions = quat_apply(self.base_quat.repeat(1, self.num_pixel_directions), self.pixel_directions)
            pixel_sample_points = (self.root_states[:, :3]).unsqueeze(1) + self.cfg.camera.near * world_pixel_directions
            self.pixel_depths[:, :] = self.cfg.camera.near

        pixel_directions_xy_normalized = world_pixel_directions / torch.norm(world_pixel_directions[:, :, :2], dim=2, keepdim=True).clamp(min=1e-10)
        
        for i in range(self.cfg.camera.num_sample_iters):
            # Get the height at the pixel sample points
            points = pixel_sample_points + self.terrain.cfg.border_size
            points = (points/self.terrain.cfg.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0]-2)
            py = torch.clip(py, 0, self.height_samples.shape[1]-2)

            heights = self.height_samples[px, py]
            heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

            cone_map_values = self.cone_map[px, py]
            cone_map_values = cone_map_values.view(self.num_envs, -1)

            # Use the cone map to figure out the step size
            step_size = torch.clamp((pixel_sample_points[:, :, 2] - heights) / (cone_map_values - pixel_directions_xy_normalized[:, :, 2]), min=self.cfg.camera.sample_step)

            # Determine which points are above the height
            above_height = pixel_sample_points[:, :, 2] - self.cfg.camera.tolerance > heights
            
            # Update the sample points which are above the height
            pixel_sample_points[above_height] += step_size[above_height].unsqueeze(-1) * pixel_directions_xy_normalized[above_height]
            self.pixel_depths[above_height] += step_size[above_height]
        
        # import cv2
        # img_idx = 10
        # above_height = pixel_sample_points[:, :, 2] - self.cfg.camera.tolerance > heights
        # valid_pixels = ~np.flip(above_height[img_idx].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T)

        # # Depth
        # img = self.pixel_depths[img_idx].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # img = np.flip(img)
        # img = np.log(img)
        # img = 1 - (img - img[valid_pixels].min()) / (img[valid_pixels].max() - img[valid_pixels].min())
        # img = (img * 255).astype(np.uint8)
        # img = img * valid_pixels
        # np.nan_to_num(img)
        # cv2.imwrite('depth.png', img)

        # # World coords
        # img_u = pixel_sample_points[img_idx, :, 0].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # img_u = np.flip(img_u)
        # img_u = (img_u - img_u[valid_pixels].min()) / (img_u[valid_pixels].max() - img_u[valid_pixels].min())
        # img_u = (img_u * 255).astype(np.uint8)
        # img_v = pixel_sample_points[img_idx, :, 1].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # img_v = np.flip(img_v)
        # img_v = (img_v - img_v[valid_pixels].min()) / (img_v[valid_pixels].max() - img_v[valid_pixels].min())
        # img_v = (img_v * 255).astype(np.uint8)
        # # img_w = pixel_sample_points[img_idx, :, 2].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # # img_w = np.flip(img_w) * valid_pixels
        # # img_w = (img_w - img_w.min()) / (img_w.max() - img_w.min())
        # # img_w = (img_w * 255).astype(np.uint8)
        # img_w = np.ones_like(img_u) * 128
        # img = np.stack([img_w, img_v, img_u], axis=-1)
        # img = img * np.expand_dims(valid_pixels, axis=-1)
        # np.nan_to_num(img)
        # cv2.imwrite('uv.png', img)

        # # Height
        # img = pixel_sample_points[img_idx, :, 2].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # img = np.flip(img)
        # img = (img - img[valid_pixels].min()) / (img[valid_pixels].max() - img[valid_pixels].min())
        # img = (img * 255).astype(np.uint8)
        # img = img * valid_pixels
        # np.nan_to_num(img)
        # cv2.imwrite('height.png', img)

        # # Checkerboard
        # x_coords = pixel_sample_points[img_idx, :, 0].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # y_coords = pixel_sample_points[img_idx, :, 1].view(self.cfg.camera.h_res, self.cfg.camera.v_res).cpu().numpy().T
        # # Color the image with a checkerboard pattern with red and blue squares in the world xy plane
        # img = np.zeros((self.cfg.camera.h_res, self.cfg.camera.v_res, 3), dtype=np.uint8)
        # x_repeat = np.floor(x_coords / 0.2).astype(np.int32) % 2
        # y_repeat = np.floor(y_coords / 0.2).astype(np.int32) % 2
        # img[x_repeat == y_repeat] = [255, 0, 0]
        # img[x_repeat != y_repeat] = [0, 0, 255]
        # img = np.flip(img) * np.expand_dims(valid_pixels, axis=-1)
        # np.nan_to_num(img)
        # cv2.imwrite('checkerboard.png', img)
        
        # breakpoint()