dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='')

backend_args = None

labels_map = {
1:0,
5:0,
7:0,
8:0,
10:0,
11:0,
13:0,
19:0,
20:0,
0:0,
29:0,
31:0,
9:1,
14:2,
15:3,
16:3,
17:4,
18:5,
21:6,
2:7,
3:7,
4:7,
6:7,
12:8,
22:9,
23:10,
24:11,
25:12,
26:13,
27:14,
28:15,
30:16
}

metainfo = dict(
    classes=labels_map, seg_label_mapping=labels_map, max_label=31)


# metainfo = dict(
#     classes=class_names, seg_label_mapping=labels_map, max_label=259)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            test_mode=False))
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file='nuscenes_infos_val.pkl',
            pipeline=eval_pipeline,
            metainfo=metainfo,
            test_mode=True))
)

test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')