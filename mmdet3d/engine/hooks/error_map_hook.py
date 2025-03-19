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
import os

@HOOKS.register_module()
class ErrorMapHook(Hook):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def after_val_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None):
        """Log custom plots after each training iteration."""
        self.log_images(outputs, runner.iter)

    def log_images(self, outputs, step):
        frame = outputs[0].lidar_path
        targets = self.load_targets_waveform_model(frame)
        gt_pc, pred_pc = self.prepare_for_eval(outputs[0].pred_waveform_data, targets)

        _, error_map = metrics_pc.get_min_distance(gt_pc, pred_pc, return_map=True)  # (b, n_rows, n_cols, 9)
        error_map = np.max(error_map, axis=-1)  # (b, n_rows, n_cols)

        self.log_error_map(error_map[0, ...], step, f"error_coarse/{frame}", max_error=450)
        self.log_error_map(error_map[0, ...], step, f"error_fine/{frame}", max_error=25)

    def prepare_for_eval(self, output, target):
        view_dir = np.load("/home/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
        view_dir = torch.from_numpy(view_dir).float()
        pred_tof = output["tof"]
        pred_score = output["patch_class"][..., 0]
        pred_tof[pred_score < 0.5] = 0  # (b, n_rows, n_cols, n_patches, 1)

        sort_idx = torch.argsort(-pred_score, dim=-1)

        pred_tof_sorted = torch.take_along_dim(pred_tof, sort_idx[..., None], dim=-2).cpu()
        centroid_tof = target["centroid_tof"]  # (b, n_rows, n_cols, n_queries)

        gt_pc = centroid_tof[..., None] * view_dir[None, :, :, None, :]  # (b, n_rows, n_cols, n_queries, 3)
        pred_pc = pred_tof_sorted * view_dir[None, :, :, None, :]

        return to_cpu(gt_pc), to_cpu(pred_pc)

    def log_image(self, image, step, tag):
        image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)  # (3, H, W)
        self.writer.add_image(tag, image, global_step=step)

    def log_error_map(self, error, step, tag, max_error=5):
        self.log_image(color_error(error, vmax=max_error), step, tag)

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
        centroid_tof = self.load_npy_or_npz(f"{data_root}/centroid_tof", frame)

        target = {
            "centroid_tof": torch.from_numpy(centroid_tof),
        }
        return target