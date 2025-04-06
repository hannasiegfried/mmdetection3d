_base_ = [
    '../_base_/datasets/carla.py', '../_base_/models/point_rcnn.py',
    '../_base_/default_runtime.py', '../_base_/schedules/cyclic-40e.py'
]
          
lr = 0.001  # max learning rate
# optim_wrapper = dict(
#     optimizer=dict(lr=lr, betas=(0.95, 0.85)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'waveform_model': dict(lr_mult=0.1)}),
)

randomness = dict(seed=4)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=5)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[5,10,45,60],
        gamma=0.5)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=1)
# param_scheduler = [
#     # learning rate scheduler
#     # During the first 35 epochs, learning rate increases from 0 to lr * 10
#     # during the next 45 epochs, learning rate decreases from lr * 10 to
#     # lr * 1e-4
#     dict(
#         type='CosineAnnealingLR',
#         T_max=35,
#         eta_min=lr * 10,
#         begin=0,
#         end=35,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=45,
#         eta_min=lr * 1e-4,
#         begin=35,
#         end=80,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     # momentum scheduler
#     # During the first 35 epochs, momentum increases from 0 to 0.85 / 0.95
#     # during the next 45 epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=35,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=35,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=45,
#         eta_min=1,
#         begin=35,
#         end=80,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
custom_hooks = []#dict(type='ErrorMapHook', log_dir=None)]

