_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1.0 / 1000,
#         by_epoch=False,
#         begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=24,
#         by_epoch=True,
#         milestones=[20, 23],
#         gamma=0.1)
# ]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5))