import hashlib
import os
import argparse
import mmcv
import copy
import numpy as np

import pycocotools.mask as mask


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
    parser.add_argument('--input', type=str, default='data/youtube_vis_2019')
    parser.add_argument('--output', type=str, default='./work_dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    lam = 0.7
    args = parse_args()
    train_ann_path = os.path.join(args.input, 'annotations/youtube_vis_2019_train.json')
    train_img_dir = os.path.join(args.input, 'train/JPEGImages')

    train_json = mmcv.load(train_ann_path)
    images = train_json['images']
    annotations = train_json['annotations']

    img_vid = 'fffe5f8df6'
    img_img = '00050.jpg'

    image_id = None
    for img in images:
        a, b = img['file_name'].split('/')
        if a == img_vid and b == img_img:
            image_id = img['id']
    assert image_id is not None

    ann_list = []
    for ann in annotations:
        if ann['image_id'] == image_id:
            ann_cur = copy.deepcopy(ann)
            h, w = ann_cur['segmentation']['size']
            rle = mask.frPyObjects([ann_cur['segmentation']], h, w)
            m = mask.decode(rle)[:, :, 0]
            ann_cur['segmentation'] = m
            ann_list.append(ann_cur)
    

    
    img_rgb = None
    for img in images:
        if img['id'] == image_id:
            img_rgb = mmcv.imread(os.path.join(train_img_dir, img['file_name']))
            break
    
    
    inst_map = np.zeros(img_rgb.shape[:2], dtype=int)
    for ann in ann_list:
        inst_map[np.nonzero(ann['segmentation'])] = ann['instance_id']
    inst_map = id2rgb(sha256map(inst_map))
    
    color_map = (img_rgb.astype(dtype=float) * (1 - lam) + inst_map.astype(dtype=float) * lam).astype(np.uint8)
    mmcv.imwrite(color_map, os.path.join(args.output, img_vid, img_img))

