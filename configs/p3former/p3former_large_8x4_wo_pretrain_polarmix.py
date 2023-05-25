_base_ = [
    '../_base_/datasets/semantickitti_panoptic_polarmix.py', '../_base_/models/p3former.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.0005
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    voxel_encoder=dict(
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=32,
        return_point_feats=False),
    backbone=dict(
        input_channels=32,
        base_channels=48,
        more_conv=True,
        out_channels=256),
    decode_head=dict(
        num_decoder_layers=6,
        num_queries=128,
        embed_dims=256,
        cls_channels=(256, 256, 20),
        mask_channels=(256, 256, 256, 256, 256),
    ))



# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
    #     end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[30],
        gamma=0.2)
]

train_dataloader = dict(batch_size=4, )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
