import hashlib
import os
import os.path as osp
import sys
import argparse
from functools import partial
from typing import Iterable

import mmcv
import numpy as np
import torch
from mmcv.utils.progressbar import init_pool, ProgressBar


def sha256num(num):
    if num == 0:
        return 0
    hex = hashlib.sha256(str(num).encode('utf-8')).hexdigest()
    hex = hex[-6:]
    return int(hex, 16)

default_class_labels = (
        'door', 'ladder', 'window', 'goal', 'sculpture', 'flag', 'parasol_or_umbrella', 'tent', 'roadblock', 'car',
        'bus', 'truck', 'bicycle', 'motorcycle', 'ship_or_boat', 'raft', 'airplane', 'person', 'cat', 'dog', 'horse',
        'cattle', 'other_animal', 'skateboard', 'ball', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package',
        'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk',
        'chair_or_seat', 'bench', 'sofa', 'gun', 'commode', 'roaster', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan',
        'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'screen_or_television', 'computer', 'printer',
        'Mobile_phone', 'keyboard', 'instrument', 'train')

def sha256map(id_map):
    return np.vectorize(sha256num)(id_map)


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


def id2pale(id_map, palette):
    assert isinstance(id_map, np.ndarray)
    rgb_shape = tuple(list(id_map.shape) + [3])
    color = np.zeros(rgb_shape, dtype=np.uint8)
    for itm in np.unique(id_map):
        if itm < len(palette):
            color[id_map == itm] = palette[itm]
    return color


def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            file=sys.stdout):
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    gen = pool.starmap(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def visualize_pred(filename, vis_dir, num_classes, num_things,
                   labels_names=None, palette=None):
    ps_id = torch.load(filename)

    cls_id = ps_id // 10000
    if palette is None:
        cls_map = id2rgb(sha256map(cls_id))
    else:
        cls_map = id2pale(cls_id, palette=palette)
    cls_map[cls_id == num_classes] = (0, 0, 0)
    ins_id = ps_id % 10000
    ins_map = id2rgb(sha256map(ins_id))
    color_map = np.where(cls_id[:, :, None] < num_things, ins_map, cls_map)

    bboxes = []
    labels = []

    for idd in np.unique(ps_id):
        cls = idd // 10000
        if cls == num_classes or cls >= num_things:
            continue
        coords = np.equal(ps_id, idd).astype(float).nonzero()
        bbox = [coords[1].min().item(), coords[0].min().item(), coords[1].max().item(), coords[0].max().item()]
        bboxes.append(bbox)
        labels.append(cls)

    color_map = mmcv.rgb2bgr(color_map)
    out_dir = osp.join(vis_dir, osp.basename(filename).replace('.pth', '.png'))
    mmcv.imwrite(color_map, out_dir)


def visualization(eval_dir, num_classes, num_things, labels, palette=None, with_gt=False):
    gt_dir = os.path.join(eval_dir, 'gt')
    pred_dir = os.path.join(eval_dir, 'pred')

    gt_names = list(mmcv.scandir(gt_dir))
    gt_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), gt_names)))
    gt_dirs = list(map(lambda x: os.path.join(gt_dir, x), gt_names))

    pred_names = list(mmcv.scandir(pred_dir))
    pred_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), pred_names)))
    pred_dirs = list(map(lambda x: os.path.join(pred_dir, x), pred_names))

    assert len(gt_dirs) == len(pred_dirs), "gt_dir:"+str(len(gt_dirs))+"pred_dir:"+str(len(pred_dirs))

    print("There are totally {} frames.".format(len(pred_dirs)))

    mmcv.mkdir_or_exist(osp.join(eval_dir, 'vis'))
    pred_vis_dir = osp.join(eval_dir, 'vis', 'pred')
    mmcv.mkdir_or_exist(pred_vis_dir)
    func = partial(visualize_pred, labels_names=labels, palette=palette)
    if with_gt:
        gt_vis_dirs = osp.join(eval_dir, 'vis', 'gt')
        mmcv.mkdir_or_exist(gt_vis_dirs)
        tasks = [(gt, gt_vis_dirs, num_classes, num_things) for gt in gt_dirs]
        track_parallel_progress(
            func,
            tasks=tasks,
            nproc=128,
        )

    else:
        gt_vis_dir = None

    tasks = [(pred, pred_vis_dir, num_classes, num_things) for pred in pred_dirs]
    track_parallel_progress(
        func,
        tasks=tasks,
        nproc=128,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Dumped Video Clips')
    parser.add_argument('result_path')
    parser.add_argument('--num_classes', default=124, type=int)
    parser.add_argument('--num_thing_classes', default=58, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    visualization(args.result_path, args.num_classes, args.num_thing_classes, default_class_labels, with_gt=True)