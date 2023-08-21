_base_ = [
    '../_base_/datasets/vspw_dvps.py',
    '../_base_/models/mask2former_video_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_schedules.py',
]

# load mask2former coco r50
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/' \
            'mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/' \
            'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0, 2, 4, 6],
            test_mode=False,
            with_depth=False,
        )
    ),
    test=dict(
        split='val',
        ref_sample_mode='test',
        ref_seq_len_test=6,
        ref_seq_index=[0, 1, 2, 3, 4, 5],
        test_mode=True,
        with_depth=False,
    )
)


num_things_classes = 0
num_stuff_classes = 124
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='Mask2FormerVideoCustomMatching',
    dataset="vip_seg",
    panoptic_head=dict(
        type='Mask2FormerVideoHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_sem_seg=None,
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        panoptic_mode='sem_seg_only_with_query',
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=False,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        object_mask_thr=0.30,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True
    ),
    init_cfg=None,
    tracker=dict(
        type='IDOL_Tracker',
        nms_thr_pre=0.7,
        nms_thr_post=0.3,
        init_score_thr=0.2,
        addnew_score_thr=0.5,
        obj_score_thr=0.1,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.5,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.5,
        match_metric='bisoftmax'
    ),
)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5,]
)
runner = dict(type='EpochBasedRunner', max_epochs=6)
