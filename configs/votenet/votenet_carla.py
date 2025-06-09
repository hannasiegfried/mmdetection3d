_base_ = [
    '../_base_/datasets/carla.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=1,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[[3.9, 1.6, 1.56]])))

# optimizer
# lr = 0.0001  # max learning rate
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
#     clip_grad=dict(max_norm=35, norm_type=2),
#     #paramwise_cfg=dict(
#     #    custom_keys={'waveform_model': dict(lr_mult=0.01)}),
# )
# randomness = dict(seed=4)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# training schedule for 1x
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# learning rate
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=80,
#         by_epoch=True,
#         milestones=[15,25,45],
#         gamma=0.5)
# ]
# param_scheduler = [
#     dict(
#         type='OneCycleLR',
#         eta_max=0.02,
#         total_steps=239520
#     )

# ]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
custom_hooks = [dict(type='ChamferDistanceHook', log_dir=None), 
                dict(type='BBHook', log_dir=None)]
