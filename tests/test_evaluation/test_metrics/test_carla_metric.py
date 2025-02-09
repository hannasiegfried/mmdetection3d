import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmdet3d.evaluation.metrics import CarlaMetric
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

def _init_evaluate_input():
    metainfo = dict(sample_idx=0)
    predictions = Det3DDataSample()
    pred_instances_3d = InstanceData()
    pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[35.9, 10.2, -0.021, 0.83, 0.63, 1.86, 2.377]]))
    pred_instances_3d.scores_3d = torch.Tensor([0.9])
    pred_instances_3d.labels_3d = torch.Tensor([0])

    predictions.pred_instances_3d = pred_instances_3d
    predictions.pred_instances = InstanceData()
    predictions.set_metainfo(metainfo)
    predictions = predictions.to_dict()
    return {}, [predictions]

def test_carla_metric_bbox():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    kittimetric = CarlaMetric(
        '/media/hasiegf/data/carla_pcdet/out/custom_infos_test.pkl', metric=['bbox'])
    kittimetric.dataset_meta = dict(classes=['Pedestrian', 'Cyclist', 'Car'])
    data_batch, predictions = _init_evaluate_input()
    kittimetric.process(data_batch, predictions)
    ap_dict = kittimetric.compute_metrics(kittimetric.results)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(
        ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_moderate'],
        3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_hard'],
                      3.0303030303030307)
