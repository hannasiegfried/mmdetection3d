_base_ = [
    '../_base_/datasets/carla.py', '../_base_/models/pointnet2_ssg.py',
    '../_base_/schedules/seg-cosine-200e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    decode_head=dict(
        num_classes=1,
        #ignore_index=0,
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        loss_decode=dict(class_weight=[1
        ])),
    test_cfg=dict(
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# data settings
train_dataloader = dict(batch_size=1)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
train_cfg = dict(val_interval=5)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
custom_hooks = []
