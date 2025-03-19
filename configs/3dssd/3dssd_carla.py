_base_ = [
    '../_base_/models/3dssd.py', '../_base_/datasets/carla.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))

# optimizer
lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'waveform_model': dict(lr_mult=0.1)}),
)
randomness = dict(seed=4)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[5,10,45,60],
        gamma=0.5)
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
custom_hooks = [dict(type='ErrorMapHook', log_dir=None)]

