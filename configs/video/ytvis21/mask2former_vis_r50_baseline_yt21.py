_base_ = [
    '../_base_/datasets/yvis_2021.py',
    '../_base_/models/mask2former_vis_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_schedules.py',
]

# load tube_link_vps coco r50
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/' \
            'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

work_dir = 'work_dir/tb_vis_001_yt21'
