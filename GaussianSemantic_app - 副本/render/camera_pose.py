# render/camera_pose.py
import numpy as np
import logging
import torch
import math
from dataclasses import dataclass, field


ZNEAR = 0.01
ZFAR = 100
Default_Height = 1080
Default_Width = 1920
Fx = Fy = 1200
Cy = Default_Height/2
Cx = Default_Width/2

@dataclass
class CameraModel:
    FoVx: float  # Field of view along x-axis
    FoVy: float  # Field of view along y-axis
    world_view_transform: torch.Tensor  # World to Camera transformation matrix
    full_proj_transform: torch.Tensor  # Projection matrix (combined with intrinsic)
    camera_center: torch.Tensor  # Camera position in world coordinates
    image_height: int = field(default=1960)
    image_width: int = field(default=1080)

def qvec2rotmat(qvec: torch.Tensor) -> torch.Tensor:
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def getProjectionMatrix(fovX, fovY, znear=ZNEAR, zfar=ZFAR):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4).cuda()

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# Function to create a mock-up CameraModel from COLMAP intrinsic and extrinsic data
def load_camera_model_from_colmap(intrinsic_str, extrinsic_str):
    # Parse intrinsic
    intrinsic_data = intrinsic_str.split()
    fx = float(intrinsic_data[4])
    fy = float(intrinsic_data[5])
    cx = float(intrinsic_data[6])
    cy = float(intrinsic_data[7])
    
    # Construct the intrinsic matrix (3x3)
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32, device='cuda')

    # Parse extrinsic
    extrinsic_data = extrinsic_str.split()
    q1, q2, q3, q4 = map(float, extrinsic_data[1:5])
    qvec = np.array([q1,q2,q3,q4])
    tx, ty, tz = map(float, extrinsic_data[5:8])
    
    # Assuming rotation is given in quaternion form, convert to rotation matrix
    R = torch.tensor(qvec2rotmat(qvec))
    T = torch.tensor([tx, ty, tz], dtype=torch.float32, device='cuda')
    
    # Construct world to view transform (extrinsic matrix 4x4)
    world_view_transform = torch.eye(4, dtype=torch.float32, device='cuda')
    world_view_transform[:3, :3] = R
    world_view_transform[:3, 3] = T
    
    world_view_transform = world_view_transform.T
    FoVx=2 * math.atan2(cx, fx)
    FoVy=2 * math.atan2(cy, fy)
    
    projection_matrix = getProjectionMatrix(fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    return CameraModel(
        FoVx=FoVx,
        FoVy=FoVy,
        image_height=int(intrinsic_data[3]),
        image_width=int(intrinsic_data[2]),
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center
    )

def qvec2rotmat_tensor(qvec: torch.Tensor) -> torch.Tensor:
    return torch.Tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]]).cuda()


def load_camera_model(
                      position: torch.Tensor, orientation:torch.Tensor,
                      h = Default_Height, w = Default_Width, fx = Fx, fy = Fy, cx = Cx, cy = Cy, 
                      ):
    R = qvec2rotmat_tensor(orientation)
    T = position
    
    # Construct world to view transform (extrinsic matrix 4x4)
    world_view_transform = torch.eye(4, dtype=torch.float32, device='cuda')
    world_view_transform[:3, :3] = R
    world_view_transform[:3, 3] = T
    
    world_view_transform = world_view_transform.T
    FoVx=2 * math.atan2(cx, fx)
    FoVy=2 * math.atan2(cy, fy)

    projection_matrix = getProjectionMatrix(fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    logging.debug(f"World View Transform:\n{world_view_transform}")
    logging.debug(f"Camera Position: {position}")
    logging.debug(f"Camera Orientation (Quaternion): {orientation}")
    logging.debug(f"Camera Center: {camera_center}")
    
    return CameraModel(
        FoVx=FoVx,
        FoVy=FoVy,
        image_height=h,
        image_width=w,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center
    )