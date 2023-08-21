_base_ = [
    '../_base_/datasets/vipseg_dvps.py',
    '../_base_/models/mask2former_video_swin_b.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_schedules.py',
]

# load tube_link_vps coco swin-b
load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask2former/" \
            "mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic/" \
            "mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth"

num_things_classes = 58
num_stuff_classes = 66
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='TubeLinkVPS',
    dataset="vip_seg",
    track_link=True,
    panoptic_head=dict(
        type='Mask2FormerVideoHeadCustomNoStuff',
        in_channels=[128, 256, 512, 1024], # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_split_thing_stuff=True,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
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
        panoptic_mode='with_query',
        loss_panoptic=None,
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
    track_head=dict(
            type='QuasiDenseMaskEmbedHeadGTMask',
            num_convs=0,
            num_fcs=2,
            roi_feat_size=1,
            in_channels=256,
            fc_out_channels=256,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0),
        ),
    track_train_cfg=dict(
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
            mask_cost=dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True)),
        sampler=dict(type='MaskPseudoSampler'),
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
crop_size = (736, 736)

train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', mode='direct'),
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0, 1, 2, 3],
            pipeline=train_pipeline,
            test_mode=False,
            with_depth=False,
        )
    ),
    test=dict(
        split='val',
        ref_sample_mode='test',
        ref_seq_len_test=2,
        ref_seq_index=[0, 1],
        test_mode=True,
        with_depth=False,
    )
)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5, ]
)
runner = dict(type='EpochBasedRunner', max_epochs=6)

find_unused_parameters=True