dataset_type = 'OV_VIPSegDataset'
data_root = 'data/VIPSeg'

# cls num 58 + 66 = 124
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
crop_size = (720, 720)
train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', mode='direct', instance_only=True),
    dict(type='SeqResizeWithDepth', img_scale=(720, 100000), ratio_range=[1., 2.], keep_ratio=True),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            mode="base_novel",
            clip_vis_emb=None,
            ref_sample_mode='sequence',
            ref_seq_index=[0],
            test_mode=False,
            pipeline=train_pipeline,
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='sequence',
        ref_seq_index=[0],
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='sequence',
        ref_seq_index=[0],
        test_mode=True,
        pipeline=test_pipeline,
    )
)

evaluation = dict()
