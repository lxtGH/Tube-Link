import os
import random
from typing import List

import torch
from typing_extensions import Literal

import copy

import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.utils import get_root_logger

from datasets.datasets.utils import SeqObj, vpq_eval, pan_mm2hb

# meta data
# Base Novel Class Distribution
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
    {'id': 60, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
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

BASE_THING_CLASSES_33 = [{'id': 2, 'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
                      {'id': 4, 'name': 'ladder', 'isthing': 1, 'color': [4, 200, 3]},
                      {'id': 8, 'name': 'window', 'isthing': 1, 'color': [230, 230, 230]},
                      {'id': 41, 'name': 'sculpture', 'isthing': 1, 'color': [0, 255, 20]},
                      {'id': 44, 'name': 'parasol_or_umbrella', 'isthing': 1, 'color': [6, 51, 255]},
                      {'id': 48, 'name': 'car', 'isthing': 1, 'color': [140, 140, 140]},
                      {'id': 49, 'name': 'bus', 'isthing': 1, 'color': [250, 10, 15]},
                      {'id': 50, 'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
                      {'id': 51, 'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
                      {'id': 60, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
                      {'id': 62, 'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
                      {'id': 64, 'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
                      {'id': 72, 'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
                      {'id': 74, 'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
                      {'id': 76, 'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
                      {'id': 77, 'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
                      {'id': 78, 'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
                      {'id': 85, 'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
                      {'id': 86, 'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
                      {'id': 87, 'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
                      {'id': 88, 'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
                      {'id': 89, 'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
                      {'id': 90, 'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
                      {'id': 91, 'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
                      {'id': 96, 'name': 'commode', 'isthing': 1, 'color': [0, 255, 163]},
                      {'id': 100, 'name': 'washing_machine', 'isthing': 1, 'color': [143, 255, 0]},
                      {'id': 101, 'name': 'Microwave_oven', 'isthing': 1, 'color': [82, 0, 255]},
                      {'id': 102, 'name': 'fan', 'isthing': 1, 'color': [163, 255, 0]},
                      {'id': 108, 'name': 'flower_pot_or_vase', 'isthing': 1, 'color': [255, 0, 31]},
                      {'id': 109, 'name': 'clock', 'isthing': 1, 'color': [0, 184, 255]},
                      {'id': 114, 'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
                      {'id': 115, 'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
                      {'id': 122, 'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]}]

NOVEL_THING_CLASSES_25 = [{'id': 10, 'name': 'goal', 'isthing': 1, 'color': [224, 5, 255]},
                       {'id': 43, 'name': 'flag', 'isthing': 1, 'color': [255, 5, 153]},
                       {'id': 46, 'name': 'tent', 'isthing': 1, 'color': [160, 150, 20]},
                       {'id': 47, 'name': 'roadblock', 'isthing': 1, 'color': [0, 163, 255]},
                       {'id': 52, 'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
                       {'id': 54, 'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
                       {'id': 55, 'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
                       {'id': 56, 'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
                       {'id': 61, 'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
                       {'id': 63, 'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
                       {'id': 65, 'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
                       {'id': 79, 'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
                       {'id': 82, 'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
                       {'id': 83, 'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
                       {'id': 84, 'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
                       {'id': 92, 'name': 'sofa', 'isthing': 1, 'color': [0, 112, 255]},
                       {'id': 95, 'name': 'gun', 'isthing': 1, 'color': [0, 122, 255]},
                       {'id': 97, 'name': 'roaster', 'isthing': 1, 'color': [255, 153, 0]},
                       {'id': 99, 'name': 'refrigerator', 'isthing': 1, 'color': [255, 112, 0]},
                       {'id': 106, 'name': 'painting_or_poster', 'isthing': 1, 'color': [0, 255, 92]},
                       {'id': 107, 'name': 'mirror', 'isthing': 1, 'color': [184, 0, 255]},
                       {'id': 116, 'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
                       {'id': 117, 'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
                       {'id': 118, 'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
                       {'id': 123, 'name': 'train', 'isthing': 1, 'color': [133, 255, 0]}]


# Add 33 base, 25 novel cases : novel clip index to ignore.
novel_clips_seq_index_33_base_25_novel = [1, 3, 5, 6, 10, 12, 15, 18, 20, 21, 22, 23, 29, 31, 33, 35, 36, 39, 41, 45, 46, 50, 52, 57, 60, 64, 65, 67, 68, 73, 74, 75, 77, 79, 81, 84, 85, 86, 87, 88, 89, 91, 92, 93, 96, 97, 98, 99, 100, 101, 104, 105, 107, 111, 116, 119, 120, 123, 131, 135, 138, 139, 141, 142, 143, 144, 146, 148, 149, 151, 152, 153, 154, 155, 156, 157, 162, 167, 169, 170, 171, 174, 175, 177, 181, 184, 185, 188, 189, 190, 192, 193, 195, 196, 197, 199, 200, 201, 202, 204, 205, 208, 209, 210, 211, 212, 213, 215, 216, 217, 224, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 244, 245, 248, 252, 253, 255, 257, 258, 259, 265, 266, 268, 270, 278, 281, 283, 286, 292, 294, 295, 298, 300, 302, 307, 315, 319, 322, 323, 325, 329, 335, 336, 339, 340,
342, 344, 345, 346, 347, 349, 352, 354, 355, 360, 364, 365, 370, 372, 373, 374, 376, 377, 379, 380, 382, 385, 386, 390, 397, 402, 404, 405, 406, 414, 416, 418, 419, 420, 422, 423, 425, 426, 427, 428, 429, 431, 438, 440, 442, 447, 450, 453, 454, 455, 456, 458, 459, 460, 465, 468, 472, 475, 479, 483, 485, 486, 488, 490, 493, 494, 495, 498, 500, 505, 507, 510, 513, 516, 519, 520, 523, 526, 530, 531, 536, 537, 538, 539, 544, 545, 546, 550, 551, 556, 560, 567, 568, 569, 570, 571, 573, 574, 575, 576, 577, 578, 579, 581, 583, 585, 586, 587, 588, 589, 593, 599, 602, 605, 606, 607, 608, 609, 610, 612, 613, 614, 617, 618, 620, 622, 624, 631, 632, 634, 635, 637, 638, 646, 651, 652, 654, 655, 657, 658, 659, 660, 667, 668, 671, 675, 678, 681, 683, 684, 685, 689, 693, 697, 700, 710, 711, 712, 713, 721, 723, 728, 729, 731, 739, 740, 741, 742, 743, 744, 745, 747, 749, 754, 758, 759, 762, 764, 765, 768, 770, 771, 776, 777, 781, 783, 784, 785, 786, 790, 792, 794, 799, 803, 805, 808, 810, 820, 825, 828, 830, 839, 842, 843, 846, 857, 858, 862, 865, 869, 871, 873, 875, 877, 880, 881, 882, 884, 886, 887, 895, 898, 903, 907, 910, 914, 915, 916, 919, 920, 921, 924, 925, 926, 927, 929, 930, 931, 932, 934, 935, 936, 937,
940, 941, 942, 947, 948, 951, 958, 959, 960, 962, 963, 964, 972, 973, 985, 986, 987, 988, 989, 992, 993, 996, 999, 1002, 1003, 1004, 1007, 1012, 1014, 1017, 1020, 1023, 1025, 1027, 1029, 1034, 1037, 1040, 1050, 1052, 1053, 1054, 1060, 1061, 1062, 1063, 1067, 1068, 1069, 1070, 1071, 1073, 1074, 1077, 1080, 1082, 1085, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1104, 1105, 1110, 1111, 1113, 1119, 1120, 1124, 1128, 1130, 1134, 1135, 1136, 1140, 1147, 1148, 1156, 1159, 1161, 1163, 1167, 1168, 1172, 1174, 1175, 1177, 1185, 1186, 1190, 1191, 1195, 1197, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1210, 1211, 1212, 1214, 1222, 1223, 1224, 1225, 1226, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1237, 1238, 1239, 1240, 1241, 1243, 1244, 1245, 1246, 1247, 1248, 1250, 1254, 1256, 1259, 1260, 1261, 1263, 1264, 1265, 1267, 1268, 1269, 1271, 1272, 1273, 1276, 1277, 1278, 1280, 1285, 1286, 1288, 1290, 1292, 1295, 1299, 1303, 1305, 1306, 1307, 1308, 1311, 1313, 1314, 1321, 1323, 1326, 1327, 1329, 1331, 1335, 1336, 1338, 1347, 1352, 1353, 1354, 1355, 1358, 1360, 1362, 1365, 1366, 1367, 1371, 1375, 1376, 1380, 1381, 1382, 1383, 1385, 1388, 1390, 1395, 1399, 1400, 1403, 1405, 1414, 1415, 1417, 1419, 1420, 1423, 1424, 1428, 1431, 1432, 1436, 1438, 1439, 1440, 1445, 1448, 1451, 1459, 1461, 1463, 1465, 1466, 1468, 1470, 1475, 1476, 1477, 1479, 1484, 1486, 1487, 1490, 1492, 1494, 1495, 1498, 1504, 1505, 1506, 1507, 1522, 1525, 1528, 1534, 1535, 1537, 1538, 1539, 1541, 1543, 1545, 1548, 1550, 1551, 1552, 1554, 1555, 1557, 1568, 1574, 1577, 1579, 1583, 1585, 1588, 1589, 1590, 1592, 1594, 1600, 1602, 1603, 1604, 1605, 1606, 1608, 1612, 1616, 1617, 1619, 1620, 1625, 1628, 1630, 1631, 1632, 1635, 1636, 1641, 1642, 1645, 1650, 1654, 1655, 1661, 1663, 1672, 1676, 1680, 1683, 1684, 1686, 1687, 1688, 1692, 1694, 1698, 1700, 1701, 1703, 1704, 1705, 1707, 1713, 1715, 1716, 1718, 1720, 1721, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1736, 1740, 1746, 1750, 1751, 1752, 1754, 1756, 1757, 1759, 1760, 1764, 1768, 1769, 1774, 1779, 1789, 1790, 1791, 1793, 1794, 1795, 1796, 1800, 1803, 1804, 1806, 1808, 1810, 1813, 1814, 1817, 1818, 1821, 1822, 1843, 1847, 1849, 1850, 1851, 1853, 1863, 1865, 1870, 1877, 1878, 1879, 1880, 1881, 1883, 1885, 1887, 1889, 1891, 1893, 1895, 1896, 1900, 1903, 1904, 1905, 1908, 1912, 1914, 1916, 1920, 1922, 1923, 1924, 1926, 1935, 1942, 1948, 1951, 1954, 1955, 1956, 1959, 1961, 1962, 1967, 1970, 1971, 1972, 1976, 1977, 1979, 1980, 1982, 1985, 1988, 1989, 1990, 1992, 1993, 1995, 1996, 1997, 1998, 1999, 2002, 2010, 2012, 2017, 2019, 2027, 2029, 2033, 2038, 2050, 2054, 2057, 2061, 2063, 2067, 2072, 2077, 2080, 2081, 2083, 2084, 2088, 2090, 2092, 2095, 2096, 2099, 2101, 2103, 2106, 2108, 2109, 2110, 2113, 2117, 2119, 2120, 2121, 2122, 2123, 2124, 2126, 2128, 2135, 2138, 2141, 2142, 2143, 2146, 2150, 2155, 2157, 2160, 2162, 2165, 2166, 2168, 2173, 2179, 2181, 2182, 2186, 2187, 2188, 2189, 2190, 2196, 2198, 2199, 2205, 2207, 2213, 2215, 2219, 2220, 2222, 2227, 2228, 2229, 2230, 2232, 2235, 2238, 2242, 2244, 2245, 2246, 2247, 2249, 2251, 2252, 2253, 2255, 2256, 2258, 2259, 2260, 2261, 2262, 2268, 2278, 2282, 2285, 2286, 2287, 2288, 2289, 2292, 2299, 2304, 2305, 2306, 2308, 2310, 2311, 2314, 2315, 2317, 2318, 2320, 2324, 2325, 2327, 2328, 2329, 2331, 2332, 2334, 2335, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2347, 2349, 2351, 2355, 2356, 2357, 2358, 2361, 2362, 2363, 2364, 2365, 2367, 2368, 2371, 2373, 2374, 2376, 2377, 2381, 2382, 2383, 2384, 2386, 2387, 2388, 2389, 2390, 2391, 2393, 2394, 2395, 2396, 2398, 2401]

# Add 46 base, 12 novel cases:
novel_clips_seq_index_46_base_12_novel = [1, 2, 3, 5, 6, 11, 14, 17, 18, 19, 22, 23, 24, 29, 30, 39, 40, 41, 43, 48, 52, 53, 56, 57, 60, 61, 65, 68, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 84, 85, 86, 87, 90, 91, 92, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 117, 123, 131, 136, 138, 139, 142, 143, 144, 146, 147, 151, 153, 156, 159, 168, 176, 177, 179, 181, 185, 186, 188, 189, 191, 192, 193, 194, 195, 200, 201, 202, 204, 205, 207, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 224, 226, 229, 230, 231, 232, 233, 234, 237, 238, 248, 249, 251, 253, 254, 255, 257, 258, 265, 266, 268, 276, 281, 284, 289, 290, 291, 293, 294, 295, 298, 299, 302, 304, 306, 308, 317, 318, 319, 320, 322, 323, 325, 329, 334, 335, 336, 338, 339, 340, 343, 345, 346, 347, 348, 349, 352, 354, 355, 360, 361, 367, 370, 374, 375, 380, 385, 390, 391, 397, 399, 402, 404, 408, 410, 414, 415, 426, 441, 442, 444, 445, 447, 450, 451, 453, 454, 455, 456, 457, 458, 459, 460, 463, 465, 471, 472, 482, 490, 494, 495, 506,
507, 510, 511, 517, 518, 523, 524, 526, 530, 531, 535, 537, 538, 539, 540, 542, 543, 544, 545, 550, 551, 556, 560, 563, 573, 576, 585, 586, 587, 588, 589, 593, 600, 602, 605, 606, 607, 608, 610, 613, 622, 623, 624, 631, 632, 633, 634, 635, 636, 637, 638, 640, 642, 646, 658, 660, 661, 666, 667, 668, 670, 671, 672, 675, 678, 681, 682, 684, 692, 699, 700, 709, 710, 714, 718, 721, 722, 725, 728, 729, 738, 739, 740, 741, 742, 743, 744, 745, 747, 749, 750, 751, 752, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 768, 769, 770, 771, 773, 776, 777, 779, 780, 781, 782, 785, 786, 787, 803, 807, 808, 810, 814, 815, 816, 822, 823, 825, 826, 828, 830, 837, 842, 843, 845, 852, 853, 857, 862, 863, 869, 872, 875, 877, 880, 881, 882, 884, 886, 887, 895, 898, 907, 908, 910, 914, 915, 918, 919, 920, 922, 924, 925, 926, 927, 929, 930, 935, 936, 937, 940, 941, 942, 948, 951, 959, 960, 961, 962, 963, 964, 972, 979, 980, 983, 985, 986, 987, 989, 991, 992, 996, 999, 1003, 1004, 1008, 1014, 1018, 1022, 1023, 1025, 1027, 1029, 1035, 1040, 1045, 1046, 1049, 1050, 1052, 1053, 1058, 1060, 1061, 1062, 1068, 1069, 1070, 1071, 1072, 1073, 1076, 1077, 1080, 1082, 1084, 1085, 1091, 1092, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1105, 1106, 1107, 1110, 1111, 1112, 1113, 1114, 1119, 1120, 1124, 1126, 1127, 1128, 1130, 1131, 1134, 1135, 1140, 1141, 1143, 1147, 1154, 1156, 1159, 1161, 1162, 1167, 1169, 1171, 1172, 1174, 1175, 1177, 1178, 1185, 1190, 1191, 1193, 1195, 1196, 1202, 1208, 1209, 1210, 1211, 1212, 1214, 1218, 1228, 1229, 1230, 1231, 1232, 1233, 1235, 1237, 1239, 1242, 1243, 1244, 1246, 1247, 1248, 1250, 1255, 1256, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1271, 1273, 1278, 1280, 1282, 1285, 1286, 1287, 1292, 1293, 1294, 1295, 1299, 1303, 1304, 1306, 1308, 1309, 1311, 1313, 1318, 1323, 1326, 1327, 1329, 1331, 1333, 1340, 1347, 1352, 1353, 1354, 1355, 1356, 1358, 1360, 1361, 1362, 1365, 1366, 1371, 1372, 1373, 1375, 1376, 1380, 1381, 1382, 1383, 1385, 1388, 1392, 1393, 1394, 1399, 1400, 1402, 1403, 1404, 1405, 1406, 1409, 1415, 1417, 1419, 1420, 1422, 1424, 1426, 1428, 1431, 1434, 1436, 1438, 1439, 1440, 1441, 1446, 1447, 1448, 1451, 1455, 1458, 1459, 1461, 1463, 1464, 1465, 1466, 1468, 1470, 1473, 1475, 1476, 1477, 1478, 1479, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1495, 1496, 1504, 1506, 1507, 1511, 1514, 1522, 1525, 1528, 1535, 1536, 1537, 1539, 1541, 1543, 1544, 1545, 1547, 1548, 1554, 1568, 1571, 1574, 1577, 1579, 1581, 1588, 1591, 1594, 1602, 1606, 1608, 1610, 1615, 1616, 1617, 1619, 1622, 1623, 1625, 1628, 1630, 1632, 1633, 1635, 1636, 1641, 1645, 1648, 1651, 1654, 1655, 1656, 1661, 1662, 1663, 1675, 1676, 1680, 1683, 1684, 1685, 1686, 1687, 1691, 1692, 1699, 1700, 1703, 1705, 1706, 1707, 1710, 1712, 1713, 1716, 1717, 1719, 1720, 1721, 1722, 1724, 1725, 1727, 1728, 1730, 1731, 1733, 1736, 1739, 1742, 1743, 1744, 1746, 1749, 1751, 1752, 1757, 1760, 1761, 1764, 1768, 1769, 1771, 1774, 1785, 1788, 1789, 1791, 1793, 1794, 1799, 1800, 1801, 1802, 1804, 1806, 1807, 1809, 1810, 1812, 1813, 1814, 1815, 1817, 1818, 1819, 1821, 1829, 1833, 1847, 1849, 1851, 1852, 1854, 1856, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1867, 1870, 1877, 1879, 1888, 1890, 1892, 1893, 1895, 1898, 1903, 1904, 1905, 1906, 1907, 1909, 1911, 1916, 1918, 1920, 1931, 1933, 1937, 1939, 1941, 1945, 1947, 1948, 1949, 1950, 1955, 1956, 1958, 1959, 1960, 1961, 1962, 1967, 1969, 1970, 1972, 1976, 1977, 1978, 1979, 1980, 1982, 1988, 1990, 1992, 1993, 1995, 1997, 1998, 2002, 2005, 2010, 2014, 2020, 2021, 2027, 2029, 2030, 2038, 2050, 2052, 2054, 2057, 2067, 2070, 2071, 2072, 2077, 2080, 2083, 2084, 2087, 2088, 2089, 2090, 2092, 2093, 2096, 2097, 2098, 2099, 2109, 2110, 2113, 2116, 2120, 2121, 2122, 2123, 2125, 2127, 2128, 2129, 2146, 2149, 2150, 2155, 2157, 2162, 2165, 2168, 2172, 2179, 2182, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2196, 2199, 2200, 2205, 2206, 2207, 2208, 2211, 2213, 2215, 2224, 2225, 2227, 2230, 2236, 2238, 2242, 2244, 2245, 2246, 2247, 2249, 2251, 2252, 2253, 2255, 2256, 2259, 2260, 2262, 2265, 2267, 2268, 2275, 2277, 2282, 2286, 2289, 2290, 2291, 2292, 2293, 2295, 2299, 2304, 2305, 2306, 2310, 2314, 2317, 2319, 2320, 2324, 2325, 2327, 2328, 2331, 2333, 2334, 2335, 2337, 2338, 2339, 2340, 2342, 2343, 2345, 2348, 2351, 2355, 2361, 2364, 2365, 2367, 2368, 2371, 2373, 2376, 2377, 2378, 2381, 2382, 2384, 2387, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2398]


NOVEL_THING_CLASSES_12 = [{'id': 50, 'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
                          {'id': 52, 'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
                          {'id': 56, 'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
                          {'id': 61, 'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
                          {'id': 65, 'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
                          {'id': 77, 'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
                          {'id': 79, 'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
                          {'id': 87, 'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
                          {'id': 89, 'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
                          {'id': 91, 'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
                          {'id': 114, 'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
                          {'id': 123, 'name': 'train', 'isthing': 1, 'color': [133, 255, 0]}]

BASE_THING_CLASSES_46 = [{'id': 2, 'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
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
                          {'id': 51, 'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
                          {'id': 54, 'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
                          {'id': 55, 'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
                          {'id': 60, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
                          {'id': 62, 'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
                          {'id': 63, 'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
                          {'id': 64, 'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
                          {'id': 72, 'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
                          {'id': 74, 'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
                          {'id': 76, 'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
                          {'id': 78, 'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
                          {'id': 82, 'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
                          {'id': 83, 'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
                          {'id': 84, 'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
                          {'id': 85, 'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
                          {'id': 86, 'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
                          {'id': 88, 'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
                          {'id': 90, 'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
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
                         {'id': 115, 'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
                         {'id': 116, 'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
                         {'id': 117, 'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
                         {'id': 118, 'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
                          {'id': 122, 'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]}]

# set open vocabulary setting.
NO_OBJ = 0
NO_OBJ_HB = 255
DIVISOR_PAN = 100
NUM_THING = 58
NUM_STUFF = 0


# keep both the base and novel classes for testing and training the upperbound model
def to_coco(pan_map, divisor=0):
    pan_new = - np.ones_like(pan_map)
    # a bug here: the id in json is not matched with the id in png file.
    vip2hb_thing = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_THING)}
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 200:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = vip2hb_thing[cls_id]
            inst_id = idx % DIVISOR_PAN
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
        else:
            # ignore the stuff
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
    assert -1. not in np.unique(pan_new)
    return pan_new


# only keep the base class for training
def to_train_coco_base_33(pan_map, divisor=0):
    pan_new = - np.ones_like(pan_map)
    # a bug here: the id in json is not matched with the id in png file.
    vip2hb_base_thing = {itm['id'] + 1: idx for idx, itm in enumerate(BASE_THING_CLASSES_33)}
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 200:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            base_ids = vip2hb_base_thing.keys()
            # for base class
            if cls_id in base_ids:
                cls_new_id = vip2hb_base_thing[cls_id]
                inst_id = idx % DIVISOR_PAN
                pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
            else:
                # ignore the novel classes
                pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            # ignore the stuff
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
    assert -1. not in np.unique(pan_new)
    return pan_new


def to_train_coco_base_46(pan_map, divisor=0):
    pan_new = - np.ones_like(pan_map)
    # a bug here: the id in json is not matched with the id in png file.
    vip2hb_base_thing = {itm['id'] + 1: idx for idx, itm in enumerate(BASE_THING_CLASSES_46)}
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 200:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            base_ids = vip2hb_base_thing.keys()
            # for base class
            if cls_id in base_ids:
                cls_new_id = vip2hb_base_thing[cls_id]
                inst_id = idx % DIVISOR_PAN
                pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
            else:
                # ignore the novel classes
                pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            # ignore the stuff
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
    assert -1. not in np.unique(pan_new)
    return pan_new

@DATASETS.register_module()
class OV_VIPSegDataset:

    # base + novel thing classes
    CLASSES = (
        'door', 'ladder', 'window', 'goal', 'sculpture', 'flag', 'parasol_or_umbrella', 'tent', 'roadblock', 'car',
        'bus', 'truck', 'bicycle', 'motorcycle', 'ship_or_boat', 'raft', 'airplane', 'person', 'cat', 'dog', 'horse',
        'cattle', 'other_animal', 'skateboard', 'ball', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package',
        'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk',
        'chair_or_seat', 'bench', 'sofa', 'gun', 'commode', 'roaster', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan',
        'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'screen_or_television', 'computer', 'printer',
        'Mobile_phone', 'keyboard', 'instrument', 'train')

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 clip_vis_emb=None,
                 split='train',
                 mode="base_novel",
                 ref_sample_mode: Literal['random', 'sequence', 'test'] = 'sequence',
                 ref_seq_index: List[int] = None,
                 ref_seq_len_test: int = 4,
                 ):
        logger = get_root_logger()

        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        img_root = os.path.join(data_root, 'images')
        seg_root = os.path.join(data_root, 'panomasks')
        assert os.path.exists(img_root)
        assert os.path.exists(seg_root)

        # read split file
        split_file = os.path.join(data_root, "ov_" + split + '.txt')
        video_folders = mmcv.list_from_file(split_file, prefix=img_root + '/')
        ann_folders = mmcv.list_from_file(split_file, prefix=seg_root + '/')

        logger.info("[VIPSeg Open Vocabulary Dataset] There are totally {} videos in {} split.".format(len(video_folders), split))

        # 58 things and 0 stuff, totally 58 classes
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert len(CLASSES_THING) == self.num_classes
        self.thing_before_stuff = False

        self.mode = mode

        # ref_seq_index is None means no ref img
        self.ref_sample_mode = ref_sample_mode
        if ref_seq_index is None:
            ref_seq_index = []
        self.ref_seq_index = ref_seq_index

        # load clip visual embeddings
        if clip_vis_emb is not None:
            assert os.path.isfile(clip_vis_emb)
            clip_vis_emb_dict = torch.load(clip_vis_emb, map_location="cpu")
            logger.info("Loaded CLIP visual embedding for base classes")
        else:
            clip_vis_emb_dict = None

        # load the

        images = []
        # remember that both img_id and seq_id start from 0
        _tmp_seq_id = -1
        for vid_folder, ann_folder in zip(video_folders, ann_folders):
            assert os.path.basename(vid_folder) == os.path.basename(ann_folder)
            _tmp_seq_id += 1
            _tmp_img_id = -1
            imgs_cur = sorted(list(map(lambda x: str(x), mmcv.scandir(vid_folder, recursive=False, suffix='.jpg'))))
            pans_cur = sorted(list(map(lambda x: str(x), mmcv.scandir(ann_folder, recursive=False, suffix='.png'))))

            # filter the novel classes for training.
            if split == "train":
                if self.mode == "base_33_novel_25":
                    if _tmp_seq_id in novel_clips_seq_index_33_base_25_novel:
                        # ignore this
                        continue
                elif self.mode == "base_46_novel_12":
                    if _tmp_seq_id in novel_clips_seq_index_46_base_12_novel:
                        # ignore this
                        continue

            for img_cur, pan_cur in zip(imgs_cur, pans_cur):
                assert img_cur.split('.')[0] == pan_cur.split('.')[0]
                _tmp_img_id += 1
                seq_id = _tmp_seq_id
                img_id = _tmp_img_id
                item_full = os.path.join(vid_folder, img_cur)
                inst_map = os.path.join(ann_folder, pan_cur)
                if clip_vis_emb_dict is not None:
                    clip_vis_emd = clip_vis_emb_dict[str(seq_id)][str(img_id)]
                else:
                    clip_vis_emd = None
                images.append(SeqObj({
                    'seq_id': int(seq_id),
                    'img_id': int(img_id),
                    'img': item_full,
                    'ann': inst_map,
                    'depth': None,
                    'no_obj_class': NO_OBJ_HB,
                    'clip_embedding_ann': clip_vis_emd
                }))
                assert os.path.exists(images[-1]['img'])
                assert os.path.exists(images[-1]['ann'])

        # Warning from Haobo: the following codes are dangerous
        # because they rely on a consistent seed among different
        # processes. Please contact me before using it.
        reference_images = {hash(image): image for image in images}

        sequences = []
        if self.ref_sample_mode == 'random':
            for img_cur in images:
                is_seq = True
                seq_now = [img_cur.dict]
                if self.ref_seq_index:
                    for index in random.choices(self.ref_seq_index, k=1):
                        query_obj = SeqObj({
                            'seq_id': img_cur.dict['seq_id'],
                            'img_id': img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(reference_images[hash(query_obj)].dict)
                        else:
                            is_seq = False
                            break
                if is_seq:
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'sequence':
            # In the sequence mode, the first frame is the key frame
            # Note that sequence mode may have multiple pointer to one frame
            for img_cur in images:
                is_seq = True
                seq_now = []
                if self.ref_seq_index:
                    for index in reversed(self.ref_seq_index):
                        query_obj = SeqObj({
                            'seq_id': img_cur.dict['seq_id'],
                            'img_id': img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(copy.deepcopy(reference_images[hash(query_obj)].dict))
                        else:
                            is_seq = False
                            break
                if is_seq:
                    seq_now.append(copy.deepcopy(img_cur.dict))
                    seq_now.reverse()
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'test':
            if ref_seq_len_test == 0:
                sequences = [[copy.deepcopy(itm.dict)] for itm in images]
            elif ref_seq_len_test == 1:
                sequences = [[copy.deepcopy(itm.dict), copy.deepcopy(itm.dict)] for itm in images]
            else:
                seq_id_pre = -1
                seq_now = []
                for img_cur in images:
                    seq_id_now = img_cur.dict['seq_id']
                    if seq_id_now != seq_id_pre:
                        seq_id_pre = seq_id_now
                        if len(seq_now) > 0:
                            while len(seq_now) < ref_seq_len_test + 1:
                                seq_now.append(copy.deepcopy(seq_now[-1]))
                            sequences.append(seq_now)
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    elif len(seq_now) % (ref_seq_len_test + 1) == 0:
                        sequences.append(seq_now)
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    else:
                        seq_now.append(copy.deepcopy(img_cur.dict))
        else:
            raise ValueError("{} not supported.".format(self.ref_sample_mode))

        logger.info("[VIPSegDVPSDataset] There are totally {} clips in {} split for training.".format(
            len(sequences), split))

        self.sequences = sequences
        self.images = reference_images

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def pre_pipelines(self, results, train=False):
        for _results in results:
            _results['img_info'] = []
            _results['thing_lower'] = 0
            _results['thing_upper'] = self.num_thing_classes
            _results['ori_filename'] = os.path.basename(_results['img'])
            _results['filename'] = _results['img']
            # train with only base classes
            if train and self.mode == "base_33_novel_25":
                _results['pre_hook'] = to_train_coco_base_33
            elif train and self.mode == "base_46_novel_12":
                _results['pre_hook'] = to_train_coco_base_46
            # train all classes
            elif train and self.mode == "base_novel":
                _results['pre_hook'] = to_coco
            # test or train with base + novel classes
            else:
                _results['pre_hook'] = to_coco

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results, train=True)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results, train=False)
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # Copy and Modify from mmdet
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def __len__(self):
        """Total number of samples of data."""
        return len(self.sequences)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    def pre_eval(self, result, eval_dir, seq_id, img_id):
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='direct', divisor=10000, with_ps_id=True,
                 test_mode=True)
        ])
        for frame_id, _result in enumerate(result):
            pan_seg_result = _result['pan_results']
            pan_seg_result = pan_mm2hb(pan_seg_result, num_classes=self.num_classes, divisor=10000)
            torch.save(
                pan_seg_result.astype(np.uint32),
                os.path.join(eval_dir, "pred", "{:06d}_{:06d}.pth".format(seq_id, img_id + frame_id)),
            )

            img_info = copy.deepcopy(self.images[hash(SeqObj({'seq_id': seq_id, 'img_id': img_id + frame_id}))].dict)
            self.pre_pipelines([img_info])
            gt = pipeline(img_info)
            gt_pan = gt['gt_panoptic_seg']
            torch.save(
                gt_pan.astype(np.uint32),
                os.path.join(eval_dir, "gt", "{:06d}_{:06d}.pth".format(seq_id, img_id + frame_id)),
            )

    def format_custom(self, results, eval_dir=None):
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='direct', divisor=10000, with_ps_id=True)
        ])
        prog_bar = mmcv.ProgressBar(len(self.images))
        for vid_id in results:
            for frame_id, _result in enumerate(results[vid_id]):
                pan_seg_result = _result['pan_results']
                # sem_seg_result = pan_seg_result % INSTANCE_OFFSET
                # pan_seg to custom design
                pan_seg_result = pan_mm2hb(pan_seg_result, num_classes=self.num_classes, divisor=10000)
                torch.save(
                    pan_seg_result.astype(np.uint32),
                    os.path.join(eval_dir, "pred", "{:06d}_{:06d}.pth".format(vid_id, frame_id)),
                )

                img_info = self.images[hash(SeqObj({'seq_id': vid_id, 'img_id': frame_id}))].dict
                self.pre_pipelines([img_info])
                gt = pipeline(img_info)
                gt_pan = gt['gt_panoptic_seg']
                torch.save(
                    gt_pan.astype(np.uint32),
                    os.path.join(eval_dir, "gt", "{:06d}_{:06d}.pth".format(vid_id, frame_id)),
                )

                prog_bar.update()

    # The evaluate func
    def evaluate(
            self,
            results,
            **kwargs
    ):
        # only support image test now
        assert self.ref_sample_mode == 'test'
        max_ins = 10000
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='direct', divisor=max_ins, with_ps_id=True)
        ])
        pq_results = []
        for vid_id in results:
            for frame_id, _result in enumerate(results[vid_id]):
                img_info = self.images[hash(SeqObj({'seq_id': vid_id, 'img_id': frame_id}))].dict
                self.pre_pipelines([img_info])
                gt = pipeline(img_info)
                gt_pan = gt['gt_panoptic_seg'].astype(np.int64)
                gt_sem = gt['gt_semantic_seg'].astype(np.int64)
                pan_seg_result = copy.deepcopy(_result['pan_results'])
                sem_seg_result = pan_seg_result % INSTANCE_OFFSET
                pan_seg_map = - np.ones_like(pan_seg_result)
                for itm in np.unique(pan_seg_result):
                    if itm >= INSTANCE_OFFSET:
                        cls = itm % INSTANCE_OFFSET
                        ins = itm // INSTANCE_OFFSET
                        pan_seg_map[pan_seg_result == itm] = cls * max_ins + ins
                    elif itm == self.num_classes:
                        pan_seg_map[pan_seg_result == itm] = self.num_classes * max_ins
                    else:
                        pan_seg_map[pan_seg_result == itm] = itm * max_ins
                assert -1 not in pan_seg_result
                pq_result = vpq_eval([pan_seg_map, gt_pan], num_classes=self.num_classes)
                pq_results.append(pq_result)
        iou_per_class = np.stack([result[0] for result in pq_results]).sum(axis=0)[:self.num_classes]
        tp_per_class = np.stack([result[1] for result in pq_results]).sum(axis=0)[:self.num_classes]
        fn_per_class = np.stack([result[2] for result in pq_results]).sum(axis=0)[:self.num_classes]
        fp_per_class = np.stack([result[3] for result in pq_results]).sum(axis=0)[:self.num_classes]
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + epsilon)
        pq = sq * rq
        return {
            "PQ": pq,
            "PQ_all": pq.mean(),
            "PQ_th": pq[:self.num_thing_classes].mean(),
            "PQ_st": pq[self.num_thing_classes:].mean(),
        }
