# note: start from 10, the swin schedule changes
_base_ = [
    '../_base_/datasets/yvis_2021.py',
    '../_base_/models/mask2former_tube_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_swin_schedules_iter.py',
]

depths = [2, 2, 18, 2]
model=dict(
    fix_backbone=False,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        pretrain_img_size=384,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
    ),
    panoptic_head=dict(
        in_channels=[192, 384, 768, 1536]
    ),
)

# load tube_link_vps coco swin-l
load_from = 'https://download.openmmlab.com/dummy/m2former_swin_larage_coco_instance.pth'

work_dir = 'work_dir/y21_swinl_010_nofix_tubemin_2_5k_5k_10k'


lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[2500, 5000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=0.001,
)

max_iters = 10000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# no use following
interval = 5000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3
)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict()

"""
g8at configs/video/exp_tubeminvis/y21_swin_l_010_tubemin_2_5k_5k_10k.py work_dir/y21_swinl_010_nofix_tubemin_2_5k_5k_10k/latest.pth --format-only --eval-options resfile_path='work_dir/y21_swinl_010_nofix_tubemin_2_5k_5k_10k'
"""
