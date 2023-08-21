import os
import random
from typing import List

import torch
from typing_extensions import Literal

import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import SeqObj, vpq_eval, pan_mm2hb

# first define the metadata here.

CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
)

THING_CLASSES = ('person', 'car')
STUFF_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'rider', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
)

kitt_step_train_id_2_cat_id = {
    0:11,
    1:13,
    2:0,
    3:1,
    4:2,
    5:3,
    6:4,
    7:5,
    8:6,
    9:7,
    10:8,
    11:9,
    12:10,
    13:12,
    14:14,
    15:15,
    16:16,
    17:17,
    18:18
}


PALETTE = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

NO_OBJ = 255
NO_OBJ_HB = 255
NUM_THING = len(THING_CLASSES)
NUM_STUFF = len(STUFF_CLASSES)


def build_classes():
    classes = []
    for cls in THING_CLASSES:
        classes.append(cls)

    for cls in STUFF_CLASSES:
        classes.append(cls)
    assert len(classes) == len(CLASSES)
    return classes


def build_palette():
    palette = []
    for cls in THING_CLASSES:
        palette.append(PALETTE[CLASSES.index(cls)])

    for cls in STUFF_CLASSES:
        palette.append(PALETTE[CLASSES.index(cls)])

    assert len(palette) == len(CLASSES)
    return palette


def to_coco(pan_map, divisor=0):
    # HB : This is to_coco situation #1
    # idx for stuff will be sem * div
    # Datasets: kitti-step
    pan_new = - np.ones_like(pan_map)

    thing_mapper = {CLASSES.index(itm): idx for idx, itm in enumerate(THING_CLASSES)}
    stuff_mapper = {CLASSES.index(itm): idx + NUM_THING for idx, itm in enumerate(STUFF_CLASSES)}
    mapper = {**thing_mapper, **stuff_mapper}
    for idx in np.unique(pan_map):
        if idx == NO_OBJ * divisor:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            cls_id = idx // divisor
            cls_new_id = mapper[cls_id]
            inst_id = idx % divisor
            # filter instance background
            if inst_id == 0 and cls_id in thing_mapper.keys():
                cls_new_id = NO_OBJ_HB
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
    assert -1. not in np.unique(pan_new)
    return pan_new


@DATASETS.register_module()
class KITTISTEPDVPSDataset:
    CLASSES = build_classes()
    PALETTE = build_palette()

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 split='train',
                 ref_sample_mode: Literal['random', 'sequence', 'test'] = 'sequence',
                 ref_seq_index: List[int] = None,
                 ref_seq_len_test: int = 4,
                 with_depth: bool = False
                 ):
        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        video_seq_dir = os.path.join(data_root, 'video_sequence', split)
        assert os.path.exists(video_seq_dir)
        assert 'leftImg8bit' not in video_seq_dir

        # Dataset information
        # 2 + 17 for KITTI STEP ; 255 for no obj
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert self.num_classes == len(self.CLASSES)
        self.no_obj_class = NO_OBJ_HB

        # ref_seq_index is None means no ref img
        self.ref_sample_mode = ref_sample_mode
        if ref_seq_index is None:
            ref_seq_index = []
        self.ref_seq_index = ref_seq_index

        filenames = list(map(lambda x: str(x), os.listdir(video_seq_dir)))
        img_names = sorted(list(filter(lambda x: 'leftImg8bit' in x, filenames)))

        # find all images
        images = []
        for itm in img_names:
            seq_id, img_id, _ = itm.split(sep="_", maxsplit=2)
            if int(seq_id) == 1 and int(img_id) in [177, 178, 179, 180] and with_depth:
                continue
            item_full = os.path.join(video_seq_dir, itm)
            images.append(SeqObj({
                'seq_id': int(seq_id),
                'img_id': int(img_id),
                'img': item_full,
                'depth': item_full.replace('leftImg8bit', 'depth') if with_depth else None,
                'ann': item_full.replace('leftImg8bit', 'panoptic'),
                'no_obj_class': self.no_obj_class
            }))
            assert os.path.exists(images[-1]['img'])
            assert images[-1]['depth'] is None or os.path.exists(images[-1]['depth']), \
                "Missing depth : {}".format(images[-1]['depth'])
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

        self.sequences = sequences
        self.images = reference_images

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def pre_pipelines(self, results):
        for _results in results:
            _results['img_info'] = []
            _results['thing_lower'] = 0
            _results['thing_upper'] = self.num_thing_classes
            _results['ori_filename'] = os.path.basename(_results['img'])
            _results['filename'] = _results['img']
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
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results)
        # if not self.ref_seq_index:
        #     results = results[0]
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # Copy and Modify from mmdet
    def __getitem__(self, idx):
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
        return len(self.sequences)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    def pre_eval(self, result, eval_dir, seq_id, img_id):
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='rgb', divisor=10000, with_ps_id=True)
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

    def evaluate(
            self,
            results,
            **kwargs
    ):
        max_ins = 10000
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='rgb', divisor=max_ins, with_ps_id=True)
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
                pq_result = vpq_eval([pan_seg_map, gt_pan])
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
