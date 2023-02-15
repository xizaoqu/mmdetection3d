voxel_size = [480, 360, 32]
in_channels = 9
out_fea_dim = 256
num_input_features = 16
use_norm = True
init_size = 32
is_fix_backbone = False

norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01)

model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        point_cloud_range=[0, '-np.pi', -4, 50, 'np.pi', 2],
        voxel_size=voxel_size,
    ),
    pts_voxel_encoder=dict(
        type='SegVFE',
        voxel_size=voxel_size,
        in_channels=in_channels,
        feat_channels=out_fea_dim,
        fea_compre=num_input_features,
        norm_cfg=norm_cfg),
    backbone=dict(
        type='Asymm3DSpconv',
        output_shape=voxel_size,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=out_fea_dim,
        num_classes=20,
    ),
)
