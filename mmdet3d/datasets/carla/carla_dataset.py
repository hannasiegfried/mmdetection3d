# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from ..det3d_dataset import Det3DDataset


@DATASETS.register_module()
class CarlaDataset(Det3DDataset):
    """CARLA Dataset."""
    METAINFO = {
        'classes': ('Car'),
        'palette': [
            (255, 0, 0),  # red
            (0, 255, 0),  # green
            (0, 0, 255),  # Blue
        ]
    }
    class_mapping = {
        'Pedestrian': -1,
        'Car': 0,
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
        return ann_info
