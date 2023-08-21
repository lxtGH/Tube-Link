output_dir1 = "D:\CVPR23_logs/tb_link/vis_results/base_vis/base_vis/vis/output"
output_dir2 = "D:\CVPR23_logs/tb_link/vis_results/vis/vis/output/output"

cat_output = "D:\CVPR23_logs/tb_link/vis_results/cat_output"


import cv2
import numpy as np
import os

files1 = os.listdir(output_dir1)
files2 = os.listdir(output_dir2)

for i, f in enumerate(files1):
    f1 = f
    f2 = files2[i]
    
