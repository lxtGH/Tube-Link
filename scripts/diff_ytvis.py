import os
import mmcv
import hashlib
import copy

import pycocotools.mask as mask

import numpy as np


YTLOC = 'data/youtube_vis_2019'
IMGPRE = 'valid/JPEGImages'


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


def read_specific_video(pred_json_1, pred_json_2, gt_json, vid_name:str):
    tot_vid = len(gt_json['videos'])
    video_id = None
    for idx in range(tot_vid):
        if gt_json['videos'][idx]['name'] == vid_name:
            video_id = idx + 1
            break
    assert video_id is not None

    seg_list_1 = []
    seg_list_2 = []
    for item in pred_json_1:
        if item['video_id'] == video_id:
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
            seg_list_1.append({
                'score': score,
                'seg': copy.deepcopy(segs),
                'cat': int(item['category_id'])
            })
    
    for item in pred_json_2:
        if item['video_id'] == video_id:
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
            seg_list_2.append({
                'score': score,
                'seg': copy.deepcopy(segs),
                'cat': int(item['category_id'])
            })
    
    seg_list_1 = sorted(seg_list_1, key=lambda x: x['score'], reverse=True)
    seg_list_2 = sorted(seg_list_2, key=lambda x: x['score'], reverse=True)

    cnt = 0
    thr = 0.3
    thr2 = 0.5
    lam = 0.7

    colors = []
    for item in gt_json['images']:
        if item['video_id'] == video_id:
            img_rgb = mmcv.imread(os.path.join(YTLOC, IMGPRE, item['file_name']))
            inst_map = np.zeros(img_rgb.shape[:2], dtype=int)
            ins_id = 0
            for seg in seg_list_1:
                ins_id += 1
                if seg['score'] <= thr:
                    break
                if not np.any(seg['seg'][cnt]):
                    continue
                seg_1 = seg['seg'][cnt]
                seg_2 = np.zeros_like(seg_1)
                for seg2 in seg_list_2:
                    if seg2['score'] > thr2 and np.any(seg2['seg'][cnt]):
                        inter = np.logical_and(seg_1.astype(bool), seg2['seg'][cnt].astype(bool)).sum()
                        union = np.logical_or(seg_1.astype(bool), seg2['seg'][cnt].astype(bool)).sum()
                        iou = inter / union
                        if iou > 0.5:
                            seg_2 = seg2['seg'][cnt]
                            break
                seg_red = np.clip(seg_1 - seg_2, a_min=0, a_max=None)
                seg_blue = np.clip(seg_2 - seg_1, a_min=0, a_max=None)
                inst_map[np.nonzero(seg_red)] = 101
                inst_map[np.nonzero(seg_blue)] = 102
            inst_map = id2rgb(sha256map(inst_map))
            color_map = (img_rgb.astype(dtype=float) * (1 - lam) + inst_map.astype(dtype=float) * lam).astype(np.uint8)

            colors.append(color_map)
            cnt += 1
    return colors



if __name__ == '__main__':
    pred_json = mmcv.load("logger/results/r50_best.json")
    pred2_json = mmcv.load("logger/results/tube_link_vps.json")
    gt_json = mmcv.load(os.path.join(YTLOC, 'annotations/youtube_vis_2019_valid.json'))


    # tennis
    # vid_name = '0b97736357'
    # moto
    vid_name = '1bcd8a65de'

    # bicycle
    vid_name = '129db5183f'

    # fish
    vid_name = 'b1a8a404ad'

    out_dir = 'work_dir/diff'

    colors = read_specific_video(pred_json, pred2_json, gt_json, vid_name)
    for idx, color in enumerate(colors):
        mmcv.imwrite(color, os.path.join(out_dir, vid_name, "{:06d}".format(idx) + '.png'))
