import argparse
import hashlib
import os

import mmcv
import numpy as np

CLASSES = [
    {"id": 0, "name": "wall", "isthing": 0, "color": [120, 120, 120]},
    {"id": 1, "name": "ceiling", "isthing": 0, "color": [180, 120, 120]},
    {"id": 2, "name": "door", "isthing": 1, "color": [6, 230, 230]},
    {"id": 3, "name": "stair", "isthing": 0, "color": [80, 50, 50]},
    {"id": 4, "name": "ladder", "isthing": 1, "color": [4, 200, 3]},
    {"id": 5, "name": "escalator", "isthing": 0, "color": [120, 120, 80]},
    {"id": 6, "name": "Playground_slide", "isthing": 0, "color": [140, 140, 140]},
    {"id": 7, "name": "handrail_or_fence", "isthing": 0, "color": [204, 5, 255]},
    {"id": 8, "name": "window", "isthing": 1, "color": [230, 230, 230]},
    {"id": 9, "name": "rail", "isthing": 0, "color": [4, 250, 7]},
    {"id": 10, "name": "goal", "isthing": 1, "color": [224, 5, 255]},
    {"id": 11, "name": "pillar", "isthing": 0, "color": [235, 255, 7]},
    {"id": 12, "name": "pole", "isthing": 0, "color": [150, 5, 61]},
    {"id": 13, "name": "floor", "isthing": 0, "color": [120, 120, 70]},
    {"id": 14, "name": "ground", "isthing": 0, "color": [8, 255, 51]},
    {"id": 15, "name": "grass", "isthing": 0, "color": [255, 6, 82]},
    {"id": 16, "name": "sand", "isthing": 0, "color": [143, 255, 140]},
    {"id": 17, "name": "athletic_field", "isthing": 0, "color": [204, 255, 4]},
    {"id": 18, "name": "road", "isthing": 0, "color": [255, 51, 7]},
    {"id": 19, "name": "path", "isthing": 0, "color": [204, 70, 3]},
    {"id": 20, "name": "crosswalk", "isthing": 0, "color": [0, 102, 200]},
    {"id": 21, "name": "building", "isthing": 0, "color": [61, 230, 250]},
    {"id": 22, "name": "house", "isthing": 0, "color": [255, 6, 51]},
    {"id": 23, "name": "bridge", "isthing": 0, "color": [11, 102, 255]},
    {"id": 24, "name": "tower", "isthing": 0, "color": [255, 7, 71]},
    {"id": 25, "name": "windmill", "isthing": 0, "color": [255, 9, 224]},
    {"id": 26, "name": "well_or_well_lid", "isthing": 0, "color": [9, 7, 230]},
    {"id": 27, "name": "other_construction", "isthing": 0, "color": [220, 220, 220]},
    {"id": 28, "name": "sky", "isthing": 0, "color": [255, 9, 92]},
    {"id": 29, "name": "mountain", "isthing": 0, "color": [112, 9, 255]},
    {"id": 30, "name": "stone", "isthing": 0, "color": [8, 255, 214]},
    {"id": 31, "name": "wood", "isthing": 0, "color": [7, 255, 224]},
    {"id": 32, "name": "ice", "isthing": 0, "color": [255, 184, 6]},
    {"id": 33, "name": "snowfield", "isthing": 0, "color": [10, 255, 71]},
    {"id": 34, "name": "grandstand", "isthing": 0, "color": [255, 41, 10]},
    {"id": 35, "name": "sea", "isthing": 0, "color": [7, 255, 255]},
    {"id": 36, "name": "river", "isthing": 0, "color": [224, 255, 8]},
    {"id": 37, "name": "lake", "isthing": 0, "color": [102, 8, 255]},
    {"id": 38, "name": "waterfall", "isthing": 0, "color": [255, 61, 6]},
    {"id": 39, "name": "water", "isthing": 0, "color": [255, 194, 7]},
    {"id": 40, "name": "billboard_or_Bulletin_Board", "isthing": 0, "color": [255, 122, 8]},
    {"id": 41, "name": "sculpture", "isthing": 1, "color": [0, 255, 20]},
    {"id": 42, "name": "pipeline", "isthing": 0, "color": [255, 8, 41]},
    {"id": 43, "name": "flag", "isthing": 1, "color": [255, 5, 153]},
    {"id": 44, "name": "parasol_or_umbrella", "isthing": 1, "color": [6, 51, 255]},
    {"id": 45, "name": "cushion_or_carpet", "isthing": 0, "color": [235, 12, 255]},
    {"id": 46, "name": "tent", "isthing": 1, "color": [160, 150, 20]},
    {"id": 47, "name": "roadblock", "isthing": 1, "color": [0, 163, 255]},
    {"id": 48, "name": "car", "isthing": 1, "color": [140, 140, 140]},
    {"id": 49, "name": "bus", "isthing": 1, "color": [250, 10, 15]},
    {"id": 50, "name": "truck", "isthing": 1, "color": [20, 255, 0]},
    {"id": 51, "name": "bicycle", "isthing": 1, "color": [31, 255, 0]},
    {"id": 52, "name": "motorcycle", "isthing": 1, "color": [255, 31, 0]},
    {"id": 53, "name": "wheeled_machine", "isthing": 0, "color": [255, 224, 0]},
    {"id": 54, "name": "ship_or_boat", "isthing": 1, "color": [153, 255, 0]},
    {"id": 55, "name": "raft", "isthing": 1, "color": [0, 0, 255]},
    {"id": 56, "name": "airplane", "isthing": 1, "color": [255, 71, 0]},
    {"id": 57, "name": "tyre", "isthing": 0, "color": [0, 235, 255]},
    {"id": 58, "name": "traffic_light", "isthing": 0, "color": [0, 173, 255]},
    {"id": 59, "name": "lamp", "isthing": 0, "color": [31, 0, 255]},
    {"id": 60, "name": "person", "isthing": 1, "color": [11, 200, 200]},
    {"id": 61, "name": "cat", "isthing": 1, "color": [255, 82, 0]},
    {"id": 62, "name": "dog", "isthing": 1, "color": [0, 255, 245]},
    {"id": 63, "name": "horse", "isthing": 1, "color": [0, 61, 255]},
    {"id": 64, "name": "cattle", "isthing": 1, "color": [0, 255, 112]},
    {"id": 65, "name": "other_animal", "isthing": 1, "color": [0, 255, 133]},
    {"id": 66, "name": "tree", "isthing": 0, "color": [255, 0, 0]},
    {"id": 67, "name": "flower", "isthing": 0, "color": [255, 163, 0]},
    {"id": 68, "name": "other_plant", "isthing": 0, "color": [255, 102, 0]},
    {"id": 69, "name": "toy", "isthing": 0, "color": [194, 255, 0]},
    {"id": 70, "name": "ball_net", "isthing": 0, "color": [0, 143, 255]},
    {"id": 71, "name": "backboard", "isthing": 0, "color": [51, 255, 0]},
    {"id": 72, "name": "skateboard", "isthing": 1, "color": [0, 82, 255]},
    {"id": 73, "name": "bat", "isthing": 0, "color": [0, 255, 41]},
    {"id": 74, "name": "ball", "isthing": 1, "color": [0, 255, 173]},
    {"id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0, "color": [10, 0, 255]},
    {"id": 76, "name": "box", "isthing": 1, "color": [173, 255, 0]},
    {"id": 77, "name": "traveling_case_or_trolley_case", "isthing": 1, "color": [0, 255, 153]},
    {"id": 78, "name": "basket", "isthing": 1, "color": [255, 92, 0]},
    {"id": 79, "name": "bag_or_package", "isthing": 1, "color": [255, 0, 255]},
    {"id": 80, "name": "trash_can", "isthing": 0, "color": [255, 0, 245]},
    {"id": 81, "name": "cage", "isthing": 0, "color": [255, 0, 102]},
    {"id": 82, "name": "plate", "isthing": 1, "color": [255, 173, 0]},
    {"id": 83, "name": "tub_or_bowl_or_pot", "isthing": 1, "color": [255, 0, 20]},
    {"id": 84, "name": "bottle_or_cup", "isthing": 1, "color": [255, 184, 184]},
    {"id": 85, "name": "barrel", "isthing": 1, "color": [0, 31, 255]},
    {"id": 86, "name": "fishbowl", "isthing": 1, "color": [0, 255, 61]},
    {"id": 87, "name": "bed", "isthing": 1, "color": [0, 71, 255]},
    {"id": 88, "name": "pillow", "isthing": 1, "color": [255, 0, 204]},
    {"id": 89, "name": "table_or_desk", "isthing": 1, "color": [0, 255, 194]},
    {"id": 90, "name": "chair_or_seat", "isthing": 1, "color": [0, 255, 82]},
    {"id": 91, "name": "bench", "isthing": 1, "color": [0, 10, 255]},
    {"id": 92, "name": "sofa", "isthing": 1, "color": [0, 112, 255]},
    {"id": 93, "name": "shelf", "isthing": 0, "color": [51, 0, 255]},
    {"id": 94, "name": "bathtub", "isthing": 0, "color": [0, 194, 255]},
    {"id": 95, "name": "gun", "isthing": 1, "color": [0, 122, 255]},
    {"id": 96, "name": "commode", "isthing": 1, "color": [0, 255, 163]},
    {"id": 97, "name": "roaster", "isthing": 1, "color": [255, 153, 0]},
    {"id": 98, "name": "other_machine", "isthing": 0, "color": [0, 255, 10]},
    {"id": 99, "name": "refrigerator", "isthing": 1, "color": [255, 112, 0]},
    {"id": 100, "name": "washing_machine", "isthing": 1, "color": [143, 255, 0]},
    {"id": 101, "name": "Microwave_oven", "isthing": 1, "color": [82, 0, 255]},
    {"id": 102, "name": "fan", "isthing": 1, "color": [163, 255, 0]},
    {"id": 103, "name": "curtain", "isthing": 0, "color": [255, 235, 0]},
    {"id": 104, "name": "textiles", "isthing": 0, "color": [8, 184, 170]},
    {"id": 105, "name": "clothes", "isthing": 0, "color": [133, 0, 255]},
    {"id": 106, "name": "painting_or_poster", "isthing": 1, "color": [0, 255, 92]},
    {"id": 107, "name": "mirror", "isthing": 1, "color": [184, 0, 255]},
    {"id": 108, "name": "flower_pot_or_vase", "isthing": 1, "color": [255, 0, 31]},
    {"id": 109, "name": "clock", "isthing": 1, "color": [0, 184, 255]},
    {"id": 110, "name": "book", "isthing": 0, "color": [0, 214, 255]},
    {"id": 111, "name": "tool", "isthing": 0, "color": [255, 0, 112]},
    {"id": 112, "name": "blackboard", "isthing": 0, "color": [92, 255, 0]},
    {"id": 113, "name": "tissue", "isthing": 0, "color": [0, 224, 255]},
    {"id": 114, "name": "screen_or_television", "isthing": 1, "color": [112, 224, 255]},
    {"id": 115, "name": "computer", "isthing": 1, "color": [70, 184, 160]},
    {"id": 116, "name": "printer", "isthing": 1, "color": [163, 0, 255]},
    {"id": 117, "name": "Mobile_phone", "isthing": 1, "color": [153, 0, 255]},
    {"id": 118, "name": "keyboard", "isthing": 1, "color": [71, 255, 0]},
    {"id": 119, "name": "other_electronic_product", "isthing": 0, "color": [255, 0, 163]},
    {"id": 120, "name": "fruit", "isthing": 0, "color": [255, 204, 0]},
    {"id": 121, "name": "food", "isthing": 0, "color": [255, 0, 143]},
    {"id": 122, "name": "instrument", "isthing": 1, "color": [0, 255, 235]},
    {"id": 123, "name": "train", "isthing": 1, "color": [133, 255, 0]}
]

CLASSES_THING = [
    {'id': 2, 'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
    {'id': 4, 'name': 'ladder', 'isthing': 1, 'color': [4, 200, 3]},
    {'id': 8, 'name': 'window', 'isthing': 1, 'color': [230, 230, 230]},
    {'id': 10, 'name': 'goal', 'isthing': 1, 'color': [224, 5, 255]},
    {'id': 41, 'name': 'sculpture', 'isthing': 1, 'color': [0, 255, 20]},
    {'id': 43, 'name': 'flag', 'isthing': 1, 'color': [255, 5, 153]},
    {'id': 44, 'name': 'parasol_or_umbrella', 'isthing': 1, 'color': [6, 51, 255]},
    {'id': 46, 'name': 'tent', 'isthing': 1, 'color': [160, 150, 20]},
    {'id': 47, 'name': 'roadblock', 'isthing': 1, 'color': [0, 163, 255]},
    {'id': 48, 'name': 'car', 'isthing': 1, 'color': [140, 140, 140]},
    {'id': 49, 'name': 'bus', 'isthing': 1, 'color': [250, 10, 15]},
    {'id': 50, 'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
    {'id': 51, 'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
    {'id': 52, 'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
    {'id': 54, 'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
    {'id': 55, 'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
    {'id': 56, 'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
    # {'id': 60, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
    # change to city color
    {'id': 60, 'name': 'person', 'isthing': 1, 'color': (220, 20, 220)},
    {'id': 61, 'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
    {'id': 62, 'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
    {'id': 63, 'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
    {'id': 64, 'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
    {'id': 65, 'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
    {'id': 72, 'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
    {'id': 74, 'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
    {'id': 76, 'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
    {'id': 77, 'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
    {'id': 78, 'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
    {'id': 79, 'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
    {'id': 82, 'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
    {'id': 83, 'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
    {'id': 84, 'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
    {'id': 85, 'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
    {'id': 86, 'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
    {'id': 87, 'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
    {'id': 88, 'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
    {'id': 89, 'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
    {'id': 90, 'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
    {'id': 91, 'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
    {'id': 92, 'name': 'sofa', 'isthing': 1, 'color': [0, 112, 255]},
    {'id': 95, 'name': 'gun', 'isthing': 1, 'color': [0, 122, 255]},
    {'id': 96, 'name': 'commode', 'isthing': 1, 'color': [0, 255, 163]},
    {'id': 97, 'name': 'roaster', 'isthing': 1, 'color': [255, 153, 0]},
    {'id': 99, 'name': 'refrigerator', 'isthing': 1, 'color': [255, 112, 0]},
    {'id': 100, 'name': 'washing_machine', 'isthing': 1, 'color': [143, 255, 0]},
    {'id': 101, 'name': 'Microwave_oven', 'isthing': 1, 'color': [82, 0, 255]},
    {'id': 102, 'name': 'fan', 'isthing': 1, 'color': [163, 255, 0]},
    {'id': 106, 'name': 'painting_or_poster', 'isthing': 1, 'color': [0, 255, 92]},
    {'id': 107, 'name': 'mirror', 'isthing': 1, 'color': [184, 0, 255]},
    {'id': 108, 'name': 'flower_pot_or_vase', 'isthing': 1, 'color': [255, 0, 31]},
    {'id': 109, 'name': 'clock', 'isthing': 1, 'color': [0, 184, 255]},
    {'id': 114, 'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
    {'id': 115, 'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
    {'id': 116, 'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
    {'id': 117, 'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
    {'id': 118, 'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
    {'id': 122, 'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]},
    {'id': 123, 'name': 'train', 'isthing': 1, 'color': [133, 255, 0]}
]

CLASSES_STUFF = [
    {'id': 0, 'name': 'wall', 'isthing': 0, 'color': [120, 120, 120]},
    {'id': 1, 'name': 'ceiling', 'isthing': 0, 'color': [180, 120, 120]},
    {'id': 3, 'name': 'stair', 'isthing': 0, 'color': [80, 50, 50]},
    {'id': 5, 'name': 'escalator', 'isthing': 0, 'color': [120, 120, 80]},
    {'id': 6, 'name': 'Playground_slide', 'isthing': 0, 'color': [140, 140, 140]},
    {'id': 7, 'name': 'handrail_or_fence', 'isthing': 0, 'color': [204, 5, 255]},
    {'id': 9, 'name': 'rail', 'isthing': 0, 'color': [4, 250, 7]},
    {'id': 11, 'name': 'pillar', 'isthing': 0, 'color': [235, 255, 7]},
    {'id': 12, 'name': 'pole', 'isthing': 0, 'color': [150, 5, 61]},
    {'id': 13, 'name': 'floor', 'isthing': 0, 'color': [120, 120, 70]},
    {'id': 14, 'name': 'ground', 'isthing': 0, 'color': [8, 255, 51]},
    {'id': 15, 'name': 'grass', 'isthing': 0, 'color': [255, 6, 82]},
    {'id': 16, 'name': 'sand', 'isthing': 0, 'color': [143, 255, 140]},
    {'id': 17, 'name': 'athletic_field', 'isthing': 0, 'color': [204, 255, 4]},
    {'id': 18, 'name': 'road', 'isthing': 0, 'color': [255, 51, 7]},
    {'id': 19, 'name': 'path', 'isthing': 0, 'color': [204, 70, 3]},
    {'id': 20, 'name': 'crosswalk', 'isthing': 0, 'color': [0, 102, 200]},
    {'id': 21, 'name': 'building', 'isthing': 0, 'color': [61, 230, 250]},
    {'id': 22, 'name': 'house', 'isthing': 0, 'color': [255, 6, 51]},
    {'id': 23, 'name': 'bridge', 'isthing': 0, 'color': [11, 102, 255]},
    {'id': 24, 'name': 'tower', 'isthing': 0, 'color': [255, 7, 71]},
    {'id': 25, 'name': 'windmill', 'isthing': 0, 'color': [255, 9, 224]},
    {'id': 26, 'name': 'well_or_well_lid', 'isthing': 0, 'color': [9, 7, 230]},
    {'id': 27, 'name': 'other_construction', 'isthing': 0, 'color': [220, 220, 220]},
    {"id": 28, "name": "sky", "isthing": 0, "color": [9, 82, 255]},
    {'id': 29, 'name': 'mountain', 'isthing': 0, 'color': [112, 9, 255]},
    {'id': 30, 'name': 'stone', 'isthing': 0, 'color': [8, 255, 214]},
    {'id': 31, 'name': 'wood', 'isthing': 0, 'color': [7, 255, 224]},
    {'id': 32, 'name': 'ice', 'isthing': 0, 'color': [255, 184, 6]},
    {'id': 33, 'name': 'snowfield', 'isthing': 0, 'color': [10, 255, 71]},
    {'id': 34, 'name': 'grandstand', 'isthing': 0, 'color': [255, 41, 10]},
    {'id': 35, 'name': 'sea', 'isthing': 0, 'color': [7, 255, 255]},
    {'id': 36, 'name': 'river', 'isthing': 0, 'color': [224, 255, 8]},
    {'id': 37, 'name': 'lake', 'isthing': 0, 'color': [102, 8, 255]},
    {'id': 38, 'name': 'waterfall', 'isthing': 0, 'color': [255, 61, 6]},
    {'id': 39, 'name': 'water', 'isthing': 0, 'color': [255, 194, 7]},
    {'id': 40, 'name': 'billboard_or_Bulletin_Board', 'isthing': 0, 'color': [255, 122, 8]},
    {'id': 42, 'name': 'pipeline', 'isthing': 0, 'color': [255, 8, 41]},
    {'id': 45, 'name': 'cushion_or_carpet', 'isthing': 0, 'color': [235, 12, 255]},
    {'id': 53, 'name': 'wheeled_machine', 'isthing': 0, 'color': [255, 224, 0]},
    {'id': 57, 'name': 'tyre', 'isthing': 0, 'color': [0, 235, 255]},
    {'id': 58, 'name': 'traffic_light', 'isthing': 0, 'color': [0, 173, 255]},
    {'id': 59, 'name': 'lamp', 'isthing': 0, 'color': [31, 0, 255]},
    {'id': 66, 'name': 'tree', 'isthing': 0, 'color': [255, 0, 0]},
    {'id': 67, 'name': 'flower', 'isthing': 0, 'color': [255, 163, 0]},
    {'id': 68, 'name': 'other_plant', 'isthing': 0, 'color': [255, 102, 0]},
    {'id': 69, 'name': 'toy', 'isthing': 0, 'color': [194, 255, 0]},
    {'id': 70, 'name': 'ball_net', 'isthing': 0, 'color': [0, 143, 255]},
    {'id': 71, 'name': 'backboard', 'isthing': 0, 'color': [51, 255, 0]},
    {'id': 73, 'name': 'bat', 'isthing': 0, 'color': [0, 255, 41]},
    {'id': 75, 'name': 'cupboard_or_showcase_or_storage_rack', 'isthing': 0, 'color': [10, 0, 255]},
    {'id': 80, 'name': 'trash_can', 'isthing': 0, 'color': [255, 0, 245]},
    {'id': 81, 'name': 'cage', 'isthing': 0, 'color': [255, 0, 102]},
    {'id': 93, 'name': 'shelf', 'isthing': 0, 'color': [51, 0, 255]},
    {'id': 94, 'name': 'bathtub', 'isthing': 0, 'color': [0, 194, 255]},
    {'id': 98, 'name': 'other_machine', 'isthing': 0, 'color': [0, 255, 10]},
    {'id': 103, 'name': 'curtain', 'isthing': 0, 'color': [255, 235, 0]},
    {'id': 104, 'name': 'textiles', 'isthing': 0, 'color': [8, 184, 170]},
    {'id': 105, 'name': 'clothes', 'isthing': 0, 'color': [133, 0, 255]},
    {'id': 110, 'name': 'book', 'isthing': 0, 'color': [0, 214, 255]},
    {'id': 111, 'name': 'tool', 'isthing': 0, 'color': [255, 0, 112]},
    {'id': 112, 'name': 'blackboard', 'isthing': 0, 'color': [92, 255, 0]},
    {'id': 113, 'name': 'tissue', 'isthing': 0, 'color': [0, 224, 255]},
    {'id': 119, 'name': 'other_electronic_product', 'isthing': 0, 'color': [255, 0, 163]},
    {'id': 120, 'name': 'fruit', 'isthing': 0, 'color': [255, 204, 0]},
    {'id': 121, 'name': 'food', 'isthing': 0, 'color': [255, 0, 143]}
]

# stuff -> thing
NO_OBJ = 0
NO_OBJ_HB = 255
DIVISOR_PAN = 100
NUM_THING = 58
NUM_STUFF = 66


def to_coco(pan_map, divisor=0):
    pan_new = - np.ones_like(pan_map)
    # a bug here: ID in json is not matched with the ID in png file.
    vip2hb_thing = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_THING)}
    vip2hb_stuff = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_STUFF)}
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 200:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = vip2hb_thing[cls_id]
            inst_id = idx % DIVISOR_PAN
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id + 1
        else:
            cls_new_id = vip2hb_stuff[idx]
            cls_new_id += NUM_THING
            pan_new[pan_map == idx] = cls_new_id * divisor
    assert -1. not in np.unique(pan_new)
    return pan_new


def id2pale(id_map, palette):
    assert isinstance(id_map, np.ndarray)
    rgb_shape = tuple(list(id_map.shape) + [3])
    color = np.zeros(rgb_shape, dtype=np.uint8)
    for itm in np.unique(id_map):
        if itm < len(palette):
            color[id_map == itm] = palette[itm]
        else:
            if itm != NO_OBJ_HB:
                raise ValueError
    return color


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def sha256num(num):
    if num == 0:
        return 0
    hex = hashlib.sha256(str(num).encode('utf-8')).hexdigest()
    hex = hex[-6:]
    return int(hex, 16)


def sha256map(id_map):
    return np.vectorize(sha256num)(id_map)


def parse_args():
    parser = argparse.ArgumentParser(description='No description.')
    parser.add_argument('--input', type=str, default='data/VIPSeg/images/2090_qAl42oaO42M')
    parser.add_argument('--output', type=str, default='./work_dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    files = sorted(list(os.listdir(args.input)))
    img_files = [os.path.join(args.input, file) for file in files]
    mask_path = args.input.replace('images', 'panomasks')
    mask_files = [os.path.join(mask_path, file.replace('.jpg', '.png')) for file in files]
    palette = [itm['color'] for itm in CLASSES_THING] + [itm['color'] for itm in CLASSES_STUFF]
    label_names = [itm['name'] for itm in CLASSES_THING] + [itm['name'] for itm in CLASSES_STUFF]
    lam = 0.7
    for task in ['vps', 'vis', 'vss']:
        for img_path, mask_path in zip(img_files, mask_files):
            img = mmcv.imread(img_path, channel_order='rgb')
            ps_id = to_coco(mmcv.imread(mask_path, flag='unchanged').astype(float), divisor=10000).astype(np.int64)
            cls_id = ps_id // 10000
            cls_map = id2pale(cls_id, palette=palette)
            cls_map[cls_id == NO_OBJ_HB] = (0, 0, 0)
            ins_id = ps_id % 10000
            ins_map = id2rgb(sha256map(ins_id))
            if task == 'vis':
                cls_map = 0
            elif task == 'vss':
                ins_map = cls_map
            color_map = np.where(cls_id[:, :, None] == 17, ins_map, cls_map)

            color_map = (img * (1 - lam) + color_map * lam).astype(np.uint8)

            color_map = mmcv.rgb2bgr(color_map)
            out_dir = os.path.join(args.output, task, os.path.basename(mask_path))
            mmcv.imwrite(color_map, out_dir)
