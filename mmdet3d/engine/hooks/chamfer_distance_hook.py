from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from mmengine.runner import Runner
from typing import Optional
from fw_lidar.visu_img import color_error, color_prob
from fw_lidar import metrics_pc
from mmdet3d.datasets.carla import pc_converter
import torch
from fw_lidar.convert_torch import to_cpu
from mmdet3d.evaluation.metrics.chamfer_metric import chamfer_distance_per_range
import os
from collections import defaultdict

@HOOKS.register_module()
class ChamferDistanceHook(Hook):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.distances = defaultdict(list)

    def after_val_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None):
        """Log custom plots after each training iteration."""
        distances = self.chamfer_distance(outputs)
        for k, v in distances.items():
            self.distances[k].append(v)

    def after_val(self, runner: Runner):
        means = {k: np.mean(v) for k, v in self.distances.items()}
        self.log_scalar_metrics(means, prefix="val")

    def after_test_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None):
        """Log custom plots after each training iteration."""
        distances = self.chamfer_distance(outputs)
        for k, v in distances.items():
            self.distances[k].append(v)

    def after_test(self, runner: Runner):
        """Log custom plots after each training iteration."""
        means = {k: np.mean(v) for k, v in self.distances.items()}
        formatted = {k: round(float(v), 4) for k, v in means.items()}
        print(f"Chamfer distance: {formatted}")

    def chamfer_distance(self, outputs):
        frame = outputs[0].lidar_path
        targets = self.load_targets_waveform_model(frame)
        gt_pc, pred_pc = self.prepare_for_eval(outputs[0].pred_points, targets)

        return chamfer_distance_per_range(gt_pc, pred_pc, bins=[40])

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