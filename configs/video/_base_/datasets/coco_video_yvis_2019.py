# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size=(384, 640)
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
        img_scale=[(320, 1e6), (352, 1e6), (384, 1e6), (416, 1e6), (448, 1e6), (480, 1e6), (512, 1e6), (544, 1e6), (576, 1e6), (608, 1e8), (640, 1e6)],
        keep_ratio=True
    ),
    dict(type='SeqFlipWithDepth', share_params=True, flip_ratio=0.5),
    dict(type='SeqRandomCropWithDepth', crop_size=crop_size, share_params=False),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids'],
        reject_empty=True,
        num_ref_imgs=2,
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
data_root = 'data/coco/'
dataset_version = '2019'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        load_as_video=False,
        ann_file=data_root + 'annotations/coco2ytvis2019_train.json',
        img_prefix=data_root + 'train2017',
        ref_img_sampler=dict(
            num_ref_imgs=2,
            frame_range=[0, 1],
            filter_key_img=False,
            method='uniform'),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        load_as_video=False,
        ann_file=data_root + 'annotations/coco2ytvis2019_val.json',
        img_prefix=data_root + 'val2017',
        ref_img_sampler=None,
        load_all_frames=False,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        load_as_video=False,
        ann_file=data_root + 'annotations/coco2ytvis2019_val.json',
        img_prefix=data_root + 'val2017',
        ref_img_sampler=None,
        load_all_frames=False,
        pipeline=test_pipeline,
    )
)

evaluation = dict()
