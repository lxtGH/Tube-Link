# dataset settings
dataset_type = 'CityscapesPanopticDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(
        type='Resize', img_scale=[(512, 1024), (2048, 4096)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/cityscapes_panoptic_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            seg_prefix=data_root + 'gtFine/cityscapes_panoptic_train/',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/cityscapes_panoptic_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/cityscapes_panoptic_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val/',
        pipeline=test_pipeline
        )
    )
evaluation = dict(interval=1, metric=['PQ'])


custom_imports = dict(
    imports=[
        'datasets.datasets.cityscapes_panoptic',
    ],
    allow_failed_imports=False
)
