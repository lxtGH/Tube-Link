import os
import cv2
import numpy as np


img_path = "D:\CVPR23_logs\kitti_step_img\mnt\lustre\lixiangtai.vendor\kitti_step_imgs"

img_list = os.listdir(img_path)

print("image list:",len(img_list))

#### baseline pred dir #####

mask_path = "D:\CVPR23_logs/kitti_step_masks/vis/pred"

mask_list = os.listdir(mask_path)

print("mask list:", len(mask_list))

output_dir = "D:\CVPR23_logs\kitti_step_ovlap"

# exit()
#### Our method pred dir #####
lam = 0.7
for i, img in enumerate(img_list):
    if i > 8000:
        break
    else:
        print("processing", img)
        img_file_path = os.path.join(img_path, img)
        ori_img = cv2.imread(img_file_path)
        mask_img = cv2.imread(os.path.join(mask_path, mask_list[i]))
        h, w, c = mask_img.shape
        resize_img = cv2.resize(ori_img, (w, h))
        
        color_map = cv2.addWeighted(np.array(resize_img), 0.5, np.array(mask_img), 0.5, 0)

        cv2.imwrite(os.path.join(output_dir, mask_list[i]), color_map)