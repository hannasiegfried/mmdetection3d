import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 299792458  # m/s
TIME_BIN_NS = 0.266
TOF_TO_M = SPEED_OF_LIGHT * TIME_BIN_NS * 1e-9 / 2

def process_pc(dist: np.array) -> np.array:
    """Converts the Carla point clouds to a format that PCDet accepts. For this, the
    distances are multiplied by the sensor specs."""
    view_dir = np.load("/lhome/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
    
    filtered_dist = filter_points(dist)

    pc = filtered_dist[..., None] * view_dir[:, :, None, :]
    pc = pc.reshape(-1, 3)
    
    # Select only rows that do not contain NaN values
    pc = pc[~np.any(np.isnan(pc), axis=1)]
    # Convert distances
    pc = pc * TOF_TO_M

    # Convert to kitti lidar coordinates
    pc = pc[:, [2, 0, 1]]
    pc[:, [1, 2]] = -pc[:, [1, 2]]
   
    # Add intensity = 0
    pc_points = np.hstack((pc, np.zeros((len(pc),1))))
    return pc_points

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
    