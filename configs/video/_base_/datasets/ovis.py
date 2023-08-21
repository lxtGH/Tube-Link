# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    dict(
        type='SeqResizeWithDepth',
        multiscale_mode='value',
        share_params=True,
        img_scale=[(288, 1e6), (320, 1e6), (352, 1e6), (384, 1e6), (416, 1e6), (448, 1e6), (480, 1e6), (512, 1e6)],
        keep_ratio=True
    ),
    dict(type='SeqFlipWithDepth', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids'],
        reject_empty=True,
        num_ref_imgs=5,
    ),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqResizeWithDepth', img_scale=(640, 360), multiscale_mode='value', share_params=True,  keep_ratio=True),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(type='VideoCollect', keys=['img']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

dataset_type = 'YouTubeVISDataset'
data_root = 'data/ovis/'
dataset_version = 'ovis'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_ovis_train.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            num_ref_imgs=5,
            frame_range=[0, 4],
            filter_key_img=False,
            method='uniform'),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_ovis_valid.json',
        img_prefix=data_root + 'valid',
        ref_img_sampler=None,
        load_all_frames=True,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_ovis_valid.json',
        img_prefix=data_root + 'valid',
        ref_img_sampler=None,
        load_all_frames=True,
        pipeline=test_pipeline,
        inf_len=36,
    )
)

evaluation = dict()
