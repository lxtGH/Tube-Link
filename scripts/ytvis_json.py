import os
import mmcv
import hashlib
import copy

import pycocotools.mask as mask

import numpy as np


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



if __name__ == '__main__':
    tb = mmcv.load("logger/results/r50_best.json")
    yt_loc = 'data/youtube_vis_2019'
    img_prefix = 'valid/JPEGImages'
    gt_json = mmcv.load(os.path.join(yt_loc, 'annotations/youtube_vis_2019_valid.json'))

    tot_vid = len(gt_json['videos'])
    for i in range(tot_vid):
        vid_num = i + 1
        print("Processing video: ", vid_num)
        seg_list = []

        for item in tb:
            if item['video_id'] == vid_num:
                score = item['score']
                segs = []
                shape = None
                for seg in item['segmentations']:
                    if seg is None:
                        segs.append(None)
                    else:
                        if shape is None:
                            shape = seg['size']
                        else:
                            assert shape == seg['size']
                        
                        h, w = seg['size']
                        m = mask.decode(seg)
                        segs.append(m)
                
                if shape is None:
                    continue
                for idx in range(len(segs)):
                    if segs[idx] is None:
                        segs[idx] = np.zeros(shape)
                seg_list.append({
                    'score': score,
                    'seg': copy.deepcopy(segs),
                    'cat': int(item['category_id'])
                })

        seg_list = sorted(seg_list, key=lambda x: x['score'], reverse=True)

        # now let's read image to visualize
        video_list = []

        cnt = 0
        thr = 0.5
        lam = 0.7
        for item in gt_json['images']:
            if item['video_id'] == vid_num:
                img_rgb = mmcv.imread(os.path.join(yt_loc, img_prefix, item['file_name']))
                inst_map = np.zeros(img_rgb.shape[:2], dtype=int)
                ins_id = 0
                for seg in seg_list:
                    ins_id += 1
                    if seg['score'] > thr:
                        if np.any(seg['seg'][cnt]):
                            inst_map[np.nonzero(seg['seg'][cnt])] = ins_id
                    else:
                        break
                inst_map = id2rgb(sha256map(inst_map))
                color_map = (img_rgb.astype(dtype=float) * (1 - lam) + inst_map.astype(dtype=float) * lam).astype(np.uint8)

                mmcv.imwrite(
                    color_map,
                    f'work_dir/tubelink/{item["file_name"]}', 
                )

                cnt += 1

