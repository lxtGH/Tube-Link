import json

json_path = "D:\VideoKNet_ckpts\cityscapes_panoptic_train.json"

file_data = json.load(open(json_path, "r"))

print(file_data.keys())
# step categories json file.\

step_categories = [{'color': [128, 64, 128], 'id': 7, 'isthing': 0, 'name': 'road', 'supercategory': 'flat'},
                   {'color': [244, 35, 232], 'id': 8, 'isthing': 0, 'name': 'sidewalk', 'supercategory': 'flat'},
                   {'color': [70, 70, 70], 'id': 11, 'isthing': 0, 'name': 'building', 'supercategory': 'construction'},
                   {'color': [102, 102, 156], 'id': 12, 'isthing': 0, 'name': 'wall', 'supercategory': 'construction'},
                   {'color': [190, 153, 153], 'id': 13, 'isthing': 0, 'name': 'fence', 'supercategory': 'construction'},
                   {'color': [153, 153, 153], 'id': 17, 'isthing': 0, 'name': 'pole', 'supercategory': 'object'},
                   {'color': [250, 170, 30], 'id': 19, 'isthing': 0, 'name': 'traffic light', 'supercategory': 'object'},
                   {'color': [220, 220, 0], 'id': 20, 'isthing': 0, 'name': 'traffic sign', 'supercategory': 'object'},
                   {'color': [107, 142, 35], 'id': 21, 'isthing': 0, 'name': 'vegetation', 'supercategory': 'nature'},
                   {'color': [152, 251, 152], 'id': 22, 'isthing': 0, 'name': 'terrain', 'supercategory': 'nature'},
                   {'color': [70, 130, 180], 'id': 23, 'isthing': 0, 'name': 'sky', 'supercategory': 'sky'},
                   {'color': [220, 20, 60], 'id': 24, 'isthing': 1, 'name': 'person', 'supercategory': 'human'},
                   {'color': [255, 0, 0], 'id': 25, 'isthing': 0, 'name': 'rider', 'supercategory': 'human'},
                   {'color': [0, 0, 142], 'id': 26, 'isthing': 1, 'name': 'car', 'supercategory': 'vehicle'},
                   {'color': [0, 0, 70], 'id': 27, 'isthing': 0, 'name': 'truck', 'supercategory': 'vehicle'},
                   {'color': [0, 60, 100], 'id': 28, 'isthing': 0, 'name': 'bus', 'supercategory': 'vehicle'},
                   {'color': [0, 80, 100], 'id': 31, 'isthing': 0, 'name': 'train', 'supercategory': 'vehicle'},
                   {'color': [0, 0, 230], 'id': 32, 'isthing': 0, 'name': 'motorcycle', 'supercategory': 'vehicle'},
                   {'color': [119, 11, 32], 'id': 33, 'isthing': 0, 'name': 'bicycle', 'supercategory': 'vehicle'}]

file_data['categories'] = step_categories

json.dump(file_data, open("D:\VideoKNet_ckpts\cityscapes_step_panoptic_train.json", "w"))