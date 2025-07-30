from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from mmengine.runner import Runner
from typing import Optional
from mmdet3d.datasets.carla import pc_converter
import torch
from fw_lidar.convert_torch import to_cpu
import os
from collections import defaultdict
from open3d import geometry
import open3d as o3d
from mmengine.visualization.utils import tensor2ndarray

@HOOKS.register_module()
class BBHook(Hook):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.bboxes_metrics = defaultdict(list)
        self.points_metrics = defaultdict(list)
        self.bboxes = []

    def after_val_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None):
        """Log custom plots after each training iteration."""
        self.bb_metric(outputs)

    def after_val(self, runner: Runner):
        means = {k: np.mean(v) for k, v in self.bboxes_metrics.items()}
        sum_points = {k: np.sum(v) for k, v in self.points_metrics.items()}
        formatted = {k: round(float(v), 4) for k, v in means.items()}
        print(f"Points inside bboxes: {formatted}")
        print(f"Points inside bboxes sum: {sum_points}")
        with open('output.txt', 'w') as f:
            for item in self.bboxes:
                f.write(str(item) + '\n')

        #self.log_scalar_metrics(means, prefix="val")

    def after_test_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None):
        """Log custom plots after each training iteration."""
        self.bb_metric(outputs)

    def after_test(self, runner: Runner):
        """Log custom plots after each training iteration."""
        means = {k: np.mean(v) for k, v in self.bboxes_metrics.items()}
        sum_points = {k: np.sum(v) for k, v in self.points_metrics.items()}
        formatted = {k: round(float(v), 4) for k, v in means.items()}
        print(f"Points inside bboxes: {formatted}")
        print(f"Points inside bboxes sum: {sum_points}")

    def bb_metric(self, outputs, bins = [20,40,60], rot_axis = 2, center_mode = 'lidar_bottom'):
        bins_points = [5, 20, 50, 200]
        frame = outputs[0].lidar_path
        targets = self.load_targets_waveform_model(frame)
        gt_pc, pred_pc = self.prepare_for_eval(outputs[0].pred_points, targets)
        gt_pc = tensor2ndarray(gt_pc)
        pred_pc = tensor2ndarray(pred_pc)
        gt_pc = o3d.utility.Vector3dVector(gt_pc)
        pred_pc = o3d.utility.Vector3dVector(pred_pc)

        gt_bboxes = outputs[0].eval_ann_info["gt_bboxes_3d"]
        gt_bboxes = tensor2ndarray(gt_bboxes.tensor)
        for i in range(len(gt_bboxes)):
            center = gt_bboxes[i, 0:3]
            dim = gt_bboxes[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = gt_bboxes[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                # bottom center to gravity center
                center[rot_axis] += dim[rot_axis] / 2
            elif center_mode == 'camera_bottom':
                # bottom center to gravity center
                center[rot_axis] -= dim[rot_axis] / 2
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            indices_gt = box3d.get_point_indices_within_bounding_box(gt_pc)
            indices_pred = box3d.get_point_indices_within_bounding_box(pred_pc)
            
            dist = np.linalg.norm(center)

            self.bboxes.append((dist, len(indices_gt)))
            idx = np.searchsorted(bins, dist)
            bin_name = f"{bins[idx-1] if idx > 0 else 0.0}-{bins[idx] if idx < len(bins) else 'inf'}"
            idx_points = np.searchsorted(bins_points, len(indices_gt))
            bin_name_points = f"{bins_points[idx_points-1] if idx_points > 0 else 0.0}-{bins_points[idx_points] if idx_points < len(bins_points) else 'inf'}"
            self.points_metrics[f"points/gt/{bin_name_points}"].append(1)
            self.bboxes_metrics[f"gt/{bin_name}"].append(len(indices_gt))
            self.bboxes_metrics[f"pred/{bin_name}"].append(len(indices_pred))

    def prepare_for_eval(self, output, target):
        gt_tof = target["gt_tof"].unsqueeze(0).unsqueeze(-1)  # (b, n_rows, n_cols, n_queries)
        gt_pc = pc_converter.process_pc_torch(gt_tof, features=None)[:,:3]
        pred_pc = output["points"][:,:3]

        return to_cpu(gt_pc), to_cpu(pred_pc)

    
    def log_scalar_metrics(self, metric_dict, prefix=None):
        for key, value in metric_dict.items():
            name = key if prefix is None else f"{prefix}/{key}"
            self.writer.add_scalar(name, value)

    def load_npy_or_npz(self, path: str, frame: str) -> np.array:
        full_path = f"{path}/{frame}"
        if os.path.exists(f"{full_path}.npy"):
            x = np.load(f"{full_path}.npy")
        elif os.path.exists(f"{full_path}.npz"):
            x = np.load(f"{full_path}.npz")["arr_0"]
        else:
            print(f"{full_path=}")
            raise NotImplementedError
        return x

    def load_targets_waveform_model(self, frame: str) -> dict:
        data_root = "/media/hasiegf/data/carla_dataset_supersampled_fulllrange/"
        gt_tof = self.load_npy_or_npz(f"{data_root}/gt_tof", frame)[..., 4]

        target = {
            "gt_tof": torch.from_numpy(gt_tof),
        }
        return target