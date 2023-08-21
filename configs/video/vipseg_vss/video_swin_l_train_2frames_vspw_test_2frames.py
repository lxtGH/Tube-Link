_base_ = [
    '../_base_/datasets/vspw_dvps.py',
    '../_base_/models/mask2former_video_swin_b.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_schedules.py',
]

# load mask2former coco swin-b
load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask2former/" \
            "mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic/" \
            "mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth"



data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0, 1],
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
    backbone=dict(
            embed_dims=192,
            num_heads=[6, 12, 24, 48],),
    panoptic_head=dict(
        type='Mask2FormerVideoHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=200,
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
    step=[5, ]
)
runner = dict(type='EpochBasedRunner', max_epochs=6)

