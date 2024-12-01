
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import math
from .camera_pose import CameraModel
from .model import GaussianModel
import logging

class Render:
    def __init__(self, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, default_camera_pose=None) -> None:
        self.model = pc
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier

        if default_camera_pose is None:
            raise ValueError("default_camera_pose cannot be None during initialization.")
        else:
            self.camera_model = default_camera_pose

        # tan(fov/2)
        tanfovx = math.tan(self.camera_model.FoVx * 0.5)
        tanfovy = math.tan(self.camera_model.FoVy * 0.5)

        # rasterization settings
        self.gs_raster_setting = GaussianRasterizationSettings(
            image_height=int(self.camera_model.image_height),
            image_width=int(self.camera_model.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=self.camera_model.world_view_transform,
            projmatrix=self.camera_model.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=self.camera_model.camera_center,
            prefiltered=False,
            debug=False,
        )

        self.rasterizer = GaussianRasterizer(raster_settings=self.gs_raster_setting)
        self.means2D = torch.zeros_like(pc._xyz, dtype=pc._xyz.dtype).cuda()

    def update_camera_pose(self, new_camera_pose: CameraModel):
        
        self.camera_model = new_camera_pose

        tanfovx = math.tan(self.camera_model.FoVx * 0.5)
        tanfovy = math.tan(self.camera_model.FoVy * 0.5)

        self.gs_raster_setting = GaussianRasterizationSettings(
            image_height=int(self.camera_model.image_height),
            image_width=int(self.camera_model.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=self.camera_model.world_view_transform,
            projmatrix=self.camera_model.full_proj_transform,
            sh_degree=self.model.active_sh_degree,
            campos=self.camera_model.camera_center,
            prefiltered=False,
            debug=False,
        )

        self.rasterizer.raster_settings = self.gs_raster_setting
        logging.debug("Rasterizer settings updated with new camera pose.")

    def forward(self, viewpoint_camera: CameraModel, scaling_modifier=1.0):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        self.gs_raster_setting = self.gs_raster_setting._replace(
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center
        )

        self.rasterizer.raster_settings = self.gs_raster_setting
        self.rasterizer = GaussianRasterizer(raster_settings=self.gs_raster_setting)
        logging.debug("Rasterizer settings updated in forward method.")

        scales = self.model.get_scaling
        rotations = self.model.get_rotation
        means3D = self.model.get_xyz
        opacity = self.model.get_opacity
        shs = self.model.get_features

        rendered_image, radii = self.rasterizer(
            means3D=means3D,
            means2D=self.means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        logging.debug("Rasterization completed.")
        return rendered_image
