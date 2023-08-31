import torch
from typing_extensions import Literal

import mmcv
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from PIL import Image


def bitmasks2bboxes(bitmasks):
    bitmasks_array = bitmasks.masks
    boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array((x[0], y[0], x[-1], y[-1]), dtype=np.float32)
    return boxes


def masks2bboxes(masks):
    bitmasks_array = masks
    boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]

        if len(x) > 0 and len(y) > 0:
            #
            # if int(x[0]) == int(x[-1]):
            #     x[0] = x[0] + 5
            # if int(y[0]) == int(y[-1]):
            #     y[0] = y[0] - 5

            boxes[idx, :] = np.array((x[0]-2, y[0]-2, x[-1], y[-1]), dtype=np.float32)
    return boxes


def extend_bbox(bbox, ratio=1.25, max_h=720, max_w=1280):
    boxes_new = np.zeros((bbox.shape[0], 4), dtype=np.float32)

    for i, box in enumerate(bbox):
        x0, y0, x1, y1 = box

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

        h = y1 - y0
        w = x1 - x0

        r_h = h * ratio
        r_w = w * ratio

        x1 = min(center_x + r_w // 2 + 1, max_w)
        y1 = min(center_y + r_h // 2 + 1, max_h)

        x0 = max(0, center_x - (r_w // 2 + 1))
        y0 = max(0, center_y - (r_h // 2 + 1))

        if x0 == x1:
            x0 = x0 + 1
        if y0 == y1:
            y0 = y0 - 1

        boxes_new[i, :] = np.array((x0, y0, x1, y1), dtype=np.float32)

    return boxes_new


@PIPELINES.register_module()
class LoadImgDirect:
    """Go ahead and just load image
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict requires "img" which is the img path.

        Returns:
            dict: The dict contains loaded image and meta information.
            'img' : img
            'img_shape' : img_shape
            'ori_shape' : original shape
            'img_fields' : the img fields
        """
        img = mmcv.imread(results['img'], channel_order='rgb', flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', ")
        return repr_str


@PIPELINES.register_module()
class LoadMultiImagesDirect(LoadImgDirect):
    """Load multi images from file.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.
        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class LoadAnnotationsDirect:
    """New version of DVPS dataloader
    mode :
        'rgb' : KITTI-STEP dataset.
    """

    def __init__(
            self,
            with_depth=False,
            mode: Literal['rgb', 'direct'] = 'rgb',
            divisor: int = 10000,
            instance_only: bool = False,
            with_ps_id: bool = False,
            with_clip_embed: bool = False,
            test_mode=False,
    ):
        self.with_depth = with_depth
        self.mode = mode
        self.divisor = divisor
        self.is_instance_only = instance_only
        self.with_ps_id = with_ps_id
        self.with_clip_embed = with_clip_embed
        self.test_mode = test_mode


    def __call__(self, results):
        # load depth map first
        if self.with_depth:
            depth = mmcv.imread(results['depth'], flag='unchanged').astype(np.float32) / 256.
            del results['depth']
            depth[depth >= 80.] = 80.
            results['gt_depth'] = depth
            results['depth_fields'] = ['gt_depth']

        if self.mode == 'rgb':
            id_map = mmcv.imread(results['ann'], flag='color', channel_order='rgb')
            gt_semantic_seg = id_map[..., 0].astype(np.float32)
            inst_map = id_map[..., 1].astype(np.float32) * 256 + id_map[..., 2].astype(np.float32)
            ps_id = gt_semantic_seg * self.divisor + inst_map
            ps_id = results['pre_hook'](ps_id, self.divisor)
            gt_semantic_seg = ps_id // self.divisor
            del results['ann']
            del results['pre_hook']

        elif self.mode == 'direct':
            ps_id = mmcv.imread(results['ann'], flag='unchanged').astype(np.float32)
            ps_id = results['pre_hook'](ps_id, self.divisor)
            del results['pre_hook']
            gt_semantic_seg = ps_id // self.divisor
        elif self.mode == 'direct_pil':
            ps_id = Image.open(results['ann'])
            ps_id = np.array(ps_id).astype(np.float32)
            ps_id = results['pre_hook'](ps_id, self.divisor)
            del results['pre_hook']
            gt_semantic_seg = ps_id // self.divisor
        else:
            raise NotImplementedError

        results['gt_semantic_seg'] = gt_semantic_seg.astype(np.int64)
        results['seg_fields'] = ['gt_semantic_seg']
        if self.with_ps_id:
            results['gt_panoptic_seg'] = ps_id

        classes = []
        masks = []
        instance_ids = []
        no_obj_class = results['no_obj_class']
        for pan_seg_id in np.unique(ps_id):
            classes.append(pan_seg_id // self.divisor)
            masks.append((ps_id == pan_seg_id).astype(np.int64))
            instance_ids.append(pan_seg_id)
        gt_labels = np.stack(classes).astype(np.int64)
        gt_instance_ids = np.stack(instance_ids).astype(np.int64)
        _height = results['img_shape'][0] if 'img_shape' in results else ps_id.shape[-2]
        _width = results['img_shape'][1] if 'img_shape' in results else ps_id.shape[-1]
        gt_masks = BitmapMasks(masks, height=_height, width=_width)
        # check the sanity of gt_masks
        verify = np.sum(gt_masks.masks.astype(np.int64), axis=0)
        assert (verify == np.ones(gt_masks.masks.shape[-2:], dtype=verify.dtype)).all()
        # now delete the no_obj_class
        gt_masks.masks = np.delete(gt_masks.masks, gt_labels == no_obj_class, axis=0)
        gt_instance_ids = np.delete(gt_instance_ids, gt_labels == no_obj_class)
        gt_labels = np.delete(gt_labels, gt_labels == no_obj_class)

        # no instance found
        if not self.test_mode:
            if len(gt_labels) == 0:
                return

        # only consider the thing classes cases
        if self.is_instance_only:
            gt_masks.masks = np.delete(
                gt_masks.masks,
                gt_labels >= results['thing_upper'],
                axis=0
            )
            gt_instance_ids = np.delete(
                gt_instance_ids,
                gt_labels >= results['thing_upper'],
            )
            gt_labels = np.delete(
                gt_labels,
                gt_labels >= results['thing_upper'],
            )
            if self.with_clip_embed:
                clip_embeds = list(results['clip_embedding_ann'].values())
                results['gt_clips_vis_emb'] = np.array(clip_embeds)

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        results['gt_instance_ids'] = gt_instance_ids
        results['mask_fields'] = ['gt_masks']

        # generate boxes
        boxes = bitmasks2bboxes(gt_masks)
        results['gt_bboxes'] = boxes
        results['bbox_fields'] = ['gt_bboxes']
        return results


@PIPELINES.register_module()
class LoadMultiAnnotationsDirect(LoadAnnotationsDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if _results is None:
                return None
            outs.append(_results)
        return outs


# mmtrack loader
@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.
        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


# mmtrack load ann
@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.
    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs

