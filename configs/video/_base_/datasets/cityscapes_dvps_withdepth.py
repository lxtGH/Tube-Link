dataset_type = 'CityscapesDVPSDataset'
data_root = 'data/cityscapes-dvps'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', with_depth=True, mode='direct'),
    dict(type='SeqResizeWithDepth', img_scale=(1024, 2048), ratio_range=[0.5, 2.0], keep_ratio=True),
    dict(type='SeqFlipWithDepth', flip_ratio=0.5),
    dict(type='SeqRandomCropWithDepth', crop_size=crop_size, share_params=True),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_instance_ids', 'gt_depth']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(type='VideoCollect', keys=['img']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0, 1],
            test_mode=False,
            pipeline=train_pipeline,
            with_depth=True,
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='sequence',
        ref_seq_index=[0, 1],
        test_mode=True,
        pipeline=test_pipeline,
        with_depth=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='test',
        ref_seq_len_test=6,
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        with_depth=True,
    )
)

evaluation = dict()
