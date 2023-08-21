import argparse
import hashlib
import os
import cv2
import numpy as np


img_root="D:\VIPSeg\VIPSeg\imgs"
val_txt_path = "D:\VIPSeg\VIPSeg/val.txt"

print(img_root)
print(val_txt_path)

# video_clip_list = []
with open(val_txt_path, "r") as r:
    video_clip_list = r.readlines()

print(video_clip_list)
img_list = []
for clip in video_clip_list:
    clip = clip.strip()

    imgs_dir = os.path.join(img_root, clip)
    imgs_list_tmp = os.listdir(imgs_dir)
    for img in imgs_list_tmp:
        imgs = os.path.join(imgs_dir, img)
        img_list.append(imgs)

print("image list:",len(img_list))

#### baseline pred dir #####

mask_path = "D:\CVPR23_logs/tb_link/vis_results/base_vis/base_vis/vis/pred"

mask_list = os.listdir(mask_path)

print("mask list:", len(mask_list))

output_dir = "D:\CVPR23_logs/tb_link/vis_results/base_vis/base_vis/vis/output"

#### Our method pred dir #####
lam = 0.7
for i, img in enumerate(img_list):
    if i > 8000:
        break
    else:
        print("processing", img)
        ori_img = cv2.imread(img)
        mask_img = cv2.imread(os.path.join(mask_path, mask_list[i]))
        h, w, c = mask_img.shape
        resize_img = cv2.resize(ori_img, (w, h))

        color_map = cv2.addWeighted(np.array(resize_img), 0.5, np.array(mask_img), 0.5, 0)

        cv2.imwrite(os.path.join(output_dir, mask_list[i]), color_map)