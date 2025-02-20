# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)

@METRICS.register_module()
class CarlaMetric(BaseMetric):
    """Carla evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [2, -52, -2, 90, 52, 6],
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Carla metric'
        super(CarlaMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.format_only = format_only
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def convert_gt_annos_to_kitti_annos(self, annos):
        """
        Args:
            annos:
            map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
            info_with_fakelidar:
        Returns:

        """
        gt_annos = []

        for anno in annos:
            # For lyft and nuscenes, different anno key in info
            anno = anno['annos']
            kitti_annos = {}

            kitti_annos['name'] = anno['name']
            kitti_annos['bbox'] = np.zeros((len(anno['name']), 4))
            kitti_annos['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
            kitti_annos['truncated'] = np.zeros(len(anno['name']))
            kitti_annos['occluded'] = np.zeros(len(anno['name']))
            kitti_annos['score'] = np.ones(len(anno['name']))
            if 'boxes_lidar' in anno:
                gt_boxes_lidar = anno['boxes_lidar'].copy()
            else:
                gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

            if len(gt_boxes_lidar) > 0:
                gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                kitti_annos['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                kitti_annos['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                kitti_annos['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                kitti_annos['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                dxdydz = gt_boxes_lidar[:, 3:6]
                kitti_annos['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                kitti_annos['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                kitti_annos['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + kitti_annos['rotation_y']
            else:
                kitti_annos['location'] = kitti_annos['dimensions'] = np.zeros((0, 3))
                kitti_annos['rotation_y'] = kitti_annos['alpha'] = np.zeros(0)

            gt_annos.append(kitti_annos)

        return gt_annos
    
    def transform_annotations_to_kitti_format(self, net_outputs, pklfile_prefix = "/media/hasiegf/data/mmdet3d/out/vis/test.pkl"):
        """
        Args:
            annos:
            map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
            info_with_fakelidar:
        Returns:

        """
        det_annos = []
        sample_idx_list = [result['sample_idx'] for result in net_outputs]

        for idx, pred_dicts in enumerate(
            mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            pred_dicts = pred_dicts['pred_instances_3d']
            gt_boxes_lidar = pred_dicts['bboxes_3d'].tensor.numpy()            

            if len(gt_boxes_lidar) > 0:
                anno = {}
                gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                anno['name'] = np.array([self.classes[int(label)] for label in pred_dicts['labels_3d']])
                anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                dxdydz = gt_boxes_lidar[:, 3:6]
                anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                anno['score'] = pred_dicts['scores_3d'].numpy()
                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }

            anno['sample_idx'] = np.array(
                [sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)

        return {'pred_instances_3d': det_annos}
        
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        gt_annos = self.convert_gt_annos_to_kitti_annos(pkl_infos)
        result_dict = self.transform_annotations_to_kitti_format(results)

        metric_dict = {}

        for metric in self.metrics:
            ap_dict = self.kitti_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        return metric_dict

    def kitti_evaluate(self,
                       results_dict: dict,
                       gt_annos: List[dict],
                       metric: Optional[str] = None,
                       classes: Optional[List[str]] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = dict()
        for name in results_dict:
            eval_types = ['bbox', 'bev', '3d']
            ap_result_str, ap_dict_ = kitti_eval(
                gt_annos, results_dict[name], classes, eval_types=eval_types)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)

        return ap_dict