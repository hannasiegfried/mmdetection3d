import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import torch


SPEED_OF_LIGHT = 299792458  # m/s
TIME_BIN_NS = 0.266
TOF_TO_M = SPEED_OF_LIGHT * TIME_BIN_NS * 1e-9 / 2

def process_pc(dist: np.array, intensity: np.array = None) -> np.array:
    """Converts the Carla point clouds to a format that PCDet accepts. For this, the
    distances are multiplied by the sensor specs."""
    view_dir = np.load("/home/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
    
    filtered_dist = filter_points(dist)

    pc = filtered_dist[..., None] * view_dir[:, :, None, :]
    pc = pc.reshape(-1, 3)

    # Add intensity
    if intensity is not None:
        pc = np.hstack((pc, intensity.reshape(-1, 1)))
    else:
        pc = np.hstack((pc, np.zeros((len(pc), 1))))
    
    # Select only rows that do not contain NaN values
    pc = pc[~np.any(np.isnan(pc), axis=1)]
    # Convert distances
    pc[:, [0, 1, 2]] = pc[:, [0, 1, 2]] * TOF_TO_M

    # Convert to kitti lidar coordinates
    pc = pc[:, [2, 0, 1, 3]]
    pc[:, [1, 2]] = -pc[:, [1, 2]]
   
    return pc

def convert_pcs(source_dir: str, destination_dir: str) -> None:
    #view_dir = np.load("/lhome/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
    
    for file_path in os.listdir(source_dir):
        file, _ending = file_path.split(".")
        dist = np.load(source_dir + file_path)['arr_0']
        pc_points = process_pc(dist)
        np.save(destination_dir + file, pc_points)

def filter_points(points: np.array) -> np.array:
    mask = points == 0
    points[mask] = np.nan 
    return points

def plot_points_image(pc: np.array) -> None:
    plt.imshow(pc[:,:,0])
    plt.colorbar()
    plt.show()

def plot_points_pc(pc: np.array) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

def process_pc_torch(dist: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
    """Converts the Carla point clouds to a format that PCDet accepts. For this, the
    distances are multiplied by the sensor specs."""
    view_dir = torch.from_numpy(np.load("/home/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")).to(dist.device)
    
    #filtered_dist = filter_points_torch(dist)
    #dist = torch.where(dist < 1, torch.tensor(0.0, device=dist.device), dist)

    pc = dist.unsqueeze(-1) * view_dir.unsqueeze(2)
    pc = pc.view(-1, 3)

    # Add intensity
    if features is not None:
        pc = torch.cat((pc, features.reshape(-1, features.shape[-1])), dim=1)
    else:
        pc = torch.cat((pc, torch.ones((pc.size(0), 1), device=pc.device)), dim=1)
    
    # Select only rows that do not contain NaN values
    #pc = pc[~torch.isnan(pc).any(dim=1)]
    pc = pc[~(pc == 0).any(dim=1)]
    # Convert distances
    pc[:, [0, 1, 2]] = pc[:, [0, 1, 2]] * TOF_TO_M

    # Convert to kitti lidar coordinates
    pc[:, [0, 1, 2]] = pc[:, [2, 0, 1]]
    pc[:, [1, 2]] = -pc[:, [1, 2]]
   
    return pc

def filter_points_torch(points: torch.Tensor) -> torch.Tensor:
    mask = points == 0
    points[mask] = float('nan')  # This is now safe
    return points
    