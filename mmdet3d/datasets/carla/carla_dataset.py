# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np
from pathlib import Path
import os

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from ..det3d_dataset import Det3DDataset


@DATASETS.register_module()
class CarlaDataset(Det3DDataset):
    """CARLA Dataset."""
    METAINFO = {
        'classes': ('Car'),#, 'Pedestrian', 'Cyclist'),
        'palette': [
            (255, 0, 0),  # red
            (0, 255, 0),  # green
            (0, 0, 255),  # Blue
        ]
    }
    class_mapping = {
        'Car': 0,
        'Pedestrian': -1,
        #'Cyclist': 2,
    }
        
    def parse(self, info: dict) -> Union[dict, None]:
        """Process the annotations in data info to ann_info."""

        # Directly access the `info` dictionary
        # Assume `info` has 'name' and 'gt_boxes_lidar'
        names = info.get('name', [])
        gt_boxes_lidar = info.get('gt_boxes_lidar', np.array([]))

        # Handle empty case (like no data or no annotations)
        if len(names) == 0 or len(gt_boxes_lidar) == 0:
            return None
        
        # Initialize the ann_info dictionary
        ann_info = dict()

        ann_info['gt_labels_3d'] = np.array([self.class_mapping.get(name, -1) for name in names], dtype=np.int64)
        ann_info['gt_bboxes_3d'] = gt_boxes_lidar

        # Count instances per category
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1

        return ann_info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = self.parse(info["annos"])
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        ann_info['gt_waveform_model'] = self.load_targets_waveform_model(info["lidar_path"])
        #ann_info['gt_waveform_model'] = {}
        return ann_info

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
        gt_tof = self.load_npy_or_npz(f"{self.data_root}/gt_tof", frame)
        centroid_tof = self.load_npy_or_npz(f"{self.data_root}/centroid_tof", frame)
        patch_class = self.load_npy_or_npz(f"{self.data_root}/target_proposal_64_classification", frame)
        patch_offset = self.load_npy_or_npz(f"{self.data_root}/target_proposal_64_offset", frame)

        target = {
            "centroid_tof": centroid_tof,
            "gt_tof": gt_tof,
            "patch_class": patch_class,
            "patch_offset": patch_offset,
        }
        return target