import os
import random
from typing import List, Optional, Tuple

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

from PIL import Image

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

CLASSES_THING = []

CLASSES_STUFF = CLASSES

# stuff -> thing
NO_OBJ = 0
NO_OBJ_HB = 255
DIVISOR_PAN = 100
NUM_THING = 0
NUM_STUFF = 124


def to_coco(pan_map, divisor=0):
    assert divisor > 100
    pan_new = - np.ones_like(pan_map)
    vspw2hb = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES)}
    for idx in np.unique(pan_map):
        if idx > 124 and idx != 255 and idx != 253:
            print(idx)
            raise NotImplementedError
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx == 253 or idx == 255:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            cls_new_id = vspw2hb[idx]
            pan_new[pan_map == idx] = cls_new_id * divisor
    assert -1. not in np.unique(pan_new)
    return pan_new


# The VSPWDVPSDataset is very similar to VIPSegDataset.
# The only difference is that the VSPWDVPSDataset has slightly different folder structure.
@DATASETS.register_module()
class VSPWDVPSDataset:
    CLASSES = [itm['name'] for itm in CLASSES_THING] + [itm['name'] for itm in CLASSES_STUFF]
    PALETTE = [itm['color'] for itm in CLASSES_THING] + [itm['color'] for itm in CLASSES_STUFF]

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 split='train',
                 ref_sample_mode: Literal['random', 'sequence', 'test'] = 'sequence',
                 ref_seq_index: List[int] = None,
                 ref_seq_len_test: int = 4,
                 with_depth: bool = False,
                 overlapping: int = 0,
                 output_scale: Optional[Tuple[int, int]] = None,
                 ):
        assert with_depth is False, "VIPSeg does not support depth annotation."
        logger = get_root_logger()

        if ref_sample_mode != 'test':
            assert overlapping == 0

        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        img_root = os.path.join(data_root, 'data')
        assert os.path.exists(img_root)

        # read split file
        split_file = os.path.join(data_root, split + '.txt')
        video_folders = mmcv.list_from_file(split_file, prefix=img_root + '/')
        logger.info("[VIPSegDVPSDataset] There are totally {} videos in {} split.".format(len(video_folders), split))

        # 58 things and 66 stuff, totally 124 classes
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert len(CLASSES_THING) == self.num_thing_classes
        assert len(CLASSES_STUFF) == self.num_stuff_classes
        assert len(CLASSES) == self.num_thing_classes + self.num_stuff_classes

        # ref_seq_index is None means no ref img
        self.ref_sample_mode = ref_sample_mode
        if ref_seq_index is None:
            ref_seq_index = []
        self.ref_seq_index = ref_seq_index

        images = []
        # remember that both img_id and seq_id start from 0
        _tmp_seq_id = -1
        for vid_folder in video_folders:
            _tmp_seq_id += 1
            _tmp_img_id = -1
            imgs_cur = sorted(list(map(lambda x: str(x), mmcv.scandir(os.path.join(vid_folder, 'origin'), recursive=False, suffix='.jpg'))))
            for img_cur in imgs_cur:
                _tmp_img_id += 1
                seq_id = _tmp_seq_id
                img_id = _tmp_img_id
                item_full = os.path.join(vid_folder, 'origin' , img_cur)
                inst_map = os.path.join(vid_folder, 'mask' , img_cur)
                inst_map = inst_map.replace('.jpg', '.png')
                images.append(SeqObj({
                    'seq_id': int(seq_id),
                    'img_id': int(img_id),
                    'img': item_full,
                    'ann': inst_map,
                    'depth': None,
                    'no_obj_class': NO_OBJ_HB
                }))
                assert os.path.exists(images[-1]['img'])
                if not test_mode:
                    assert os.path.exists(images[-1]['ann'])

        logger.info("[VIPSegDVPSDataset] There are totally {} images in {} split.".format(len(images), split))

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
            elif overlapping != 0:
                seq_id_pre = -1
                seq_now = []
                idx_img = 0
                while idx_img < len(images):
                    img_cur = images[idx_img]
                    seq_id_now = img_cur.dict['seq_id']
                    if seq_id_now != seq_id_pre:
                        seq_id_pre = seq_id_now
                        if len(seq_now) > 0:
                            while len(seq_now) < ref_seq_len_test + 1:
                                seq_now.append(copy.deepcopy(seq_now[-1]))
                            sequences.append(seq_now)
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    elif len(seq_now) == (ref_seq_len_test + 1):
                        sequences.append(seq_now)
                        idx_img -= overlapping
                        img_cur = images[idx_img]
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    else:
                        seq_now.append(copy.deepcopy(img_cur.dict))
                    idx_img += 1
                if len(seq_now) > 0:
                    while len(seq_now) < ref_seq_len_test + 1:
                        seq_now.append(copy.deepcopy(seq_now[-1]))
                    sequences.append(seq_now)
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
                if len(seq_now) > 0:
                    while len(seq_now) < ref_seq_len_test + 1:
                        seq_now.append(copy.deepcopy(seq_now[-1]))
                    sequences.append(seq_now)
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
        self.output_scale = output_scale

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
            dict(type='LoadAnnotationsDirect', mode='direct_pil', with_ps_id=True)
        ])
        for frame_id, _result in enumerate(result):
            sem_seg_result = _result['sem_results']
            assert 255 not in sem_seg_result
            sem_seg_result = sem_seg_result + 1
            img_info = copy.deepcopy(self.images[hash(SeqObj({'seq_id': seq_id, 'img_id': img_id + frame_id}))].dict)
            vid_name = img_info['ann'].split('/')[3]
            img_name = img_info['ann'].split('/')[-1]
            pred_path = os.path.join(eval_dir, "pred", vid_name, img_name)
            if self.output_scale:
                sem_seg_result = mmcv.imrescale(
                    img=sem_seg_result,
                    scale=self.output_scale,
                    return_scale=False,
                    interpolation='nearest',
                )
            if not os.path.exists(pred_path):
                mmcv.imwrite(sem_seg_result.astype(np.uint8), pred_path)


    def format_custom(self, results, eval_dir=None):
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=False, mode='direct', divisor=10000, with_ps_id=True)
        ])
        prog_bar = mmcv.ProgressBar(len(self.images))
        for vid_id in results:
            for frame_id, _result in enumerate(results[vid_id]):
                pan_seg_result = _result['pan_results']
                # sem_seg_result = pan_seg_result % INSTANCE_OFFSET
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
