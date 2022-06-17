# dataset settings
dataset_type = 'LesionDataset'
data_root = 'data/DeepLesion/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_3DCE', to_float32=False, lesion_input=True, num_slice=9, zflip=True, window=[-1024, 1050]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         multiscale_mode='value',
         img_scale=[(448,448), (512, 512), (576,576)],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=9),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_3DCE', to_float32=False, lesion_input=True, num_slice=9, zflip=False, window=[-1024, 1050]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=9),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/deeplesion_train.json',
        img_prefix=data_root + 'Images_png/Images_png/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/deeplesion_val.json',
        img_prefix=data_root + 'Images_png/Images_png/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/deeplesion_test.json',
        img_prefix=data_root + 'Images_png/Images_png/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
