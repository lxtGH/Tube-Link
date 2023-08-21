_base_ = [
    '../_base_/datasets/yvis_2019.py',
    '../_base_/models/mask2former_tube_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_schedules_iter.py',
]


model=dict(
    fix_backbone=False
)

# load tube_link_vps coco r50
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/' \
            'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

work_dir = 'work_dir/ytvis19/r50_tb_link'


lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[5000, 10000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=0.001,
)

max_iters = 15000
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
g8at configs/video/exp_tubeminvis/y19_r50_003_tubemin_5k_10k_15k.py work_dir/y19_r50_003_nofix_tubemin_5k_10k_15k/latest.pth --format-only --eval-options resfile_path='work_dir/y19_r50_003_nofix_tubemin_5k_10k_15k'
"""
