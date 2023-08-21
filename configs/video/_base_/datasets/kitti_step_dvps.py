dataset_type = 'KITTISTEPDVPSDataset'
data_root = 'data/kitti-step'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
crop_size = (384, 1248)
# The kitti dataset contains 1226 x 370 and 1241 x 376
train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', with_depth=False, mode='rgb'),
    dict(type='SeqResizeWithDepth', img_scale=(384, 1248), ratio_range=[1.0, 2.0], keep_ratio=True),
    dict(type='SeqFlipWithDepth', flip_ratio=0.5),
    dict(type='SeqRandomCropWithDepth', crop_size=crop_size, share_params=True),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_instance_ids']),
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
        times=4,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0, 1],
            test_mode=False,
            pipeline=train_pipeline,
            with_depth=False,
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
        with_depth=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='test',
        ref_seq_len_test=2,
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        with_depth=False,
    )
)

evaluation = dict()
