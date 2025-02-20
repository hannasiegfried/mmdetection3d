# dataset settings
dataset_type = 'CarlaDataset'
data_root = '/media/hasiegf/data/carla_pcdet/'
class_names = ['Car']  # replace with your dataset class
point_cloud_range = [2, -52, -2, 90, 52, 6] # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromWaveform',
        data_path='/media/hasiegf/data/carla_dataset_supersampled_fulllrange/waveform_8bit/'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='PointSample', num_points=16384),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromWaveformModel',
        data_path='/media/hasiegf/data/carla_dataset_supersampled_fulllrange/waveform_8bit/'),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
eval_pipeline = [
    dict(
        type='LoadPointsFromWaveform',
        data_path='/media/hasiegf/data/carla_dataset_supersampled_fulllrange/waveform_8bit/'),
    dict(type='Pack3DDetInputs', keys=['points']),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CarlaDataset',
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file = '/media/hasiegf/data/carla_mmdet/out/custom_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CarlaDataset',
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file = '/media/hasiegf/data/carla_mmdet/out/custom_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR')
    )
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CarlaDataset',
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file = '/media/hasiegf/data/carla_mmdet/out/custom_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR')
    )
test_evaluator = dict(
    type='CarlaMetric',
    ann_file='/media/hasiegf/data/carla_mmdet/out/custom_infos_test.pkl',
    metric='bbox')
val_evaluator = dict(
    type='CarlaMetric',
    ann_file='/media/hasiegf/data/carla_mmdet/out/custom_infos_val.pkl',
    metric='bbox')