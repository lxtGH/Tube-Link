# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
import torch

from mmdet.utils import get_root_logger

from mmdet.core import INSTANCE_OFFSET, bbox2result, encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector
from datasets.datasets.kitti_step_dvps import kitt_step_train_id_2_cat_id

def mapping_train_id_to_cat_id_segm(
        sem_seg_result,
        train_id_2_cat_id=kitt_step_train_id_2_cat_id
):
    labels = np.unique(sem_seg_result)
    sem_seg_result_new = np.ones_like(sem_seg_result) * 255
    for i in labels:
        if i == 255:
            continue
        else:
            masks = sem_seg_result == i
            sem_seg_result_new[masks] = train_id_2_cat_id[i]
    return sem_seg_result_new


@DETECTORS.register_module()
class Mask2FormerVideoCustom(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 dataset="kitti-step",
                 fix_backbone=False
                 ):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.dataset = dataset

        # BaseDetector.show_result default for instance segmentation
        if self.num_stuff_classes > 0:
            self.show_result = self._show_pan_result

        self.logger = get_root_logger()
        self.logger.info("[Unified Video Segmentation] Using customized tube_link_vps segmentor.")

        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            self.backbone.train(mode=False)
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                self.logger.info(name + " is fixed.")

    def forward_dummy(self, img, img_metas):
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      *,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_gt_semantic_seg=None,
                      ref_gt_instance_ids=None,
                      **kargs):
        super(SingleStageDetector, self).forward_train(ref_img[0], ref_img_metas[0])
        # add batch_input_shape in img_metas
        batch_input_shape = tuple(img.size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        for ref_meta in ref_img_metas:
            for meta in ref_meta:
                meta['batch_input_shape'] = batch_input_shape
        # step 1 : extract volume img features
        bs, num_frame, three, h, w = ref_img.size()  # (b,T,3,h,w)

        ref_video = ref_img.reshape((bs * num_frame, three, h, w))
        video_x = self.extract_feat(ref_video)
        # step 2: forward the volume features
        losses = self.panoptic_head.forward_train(
            video_x,
            ref_img_metas,
            ref_gt_bboxes,
            ref_gt_labels,
            ref_gt_masks,
            ref_gt_semantic_seg,
            ref_gt_instance_ids,
            gt_bboxes_ignore=None
        )

        return losses

    def simple_test(self, img, img_metas, ref_img, ref_img_metas, **kwargs):

        bs, num_frame, three, h, w = ref_img.size()
        # (b, t, 3, h, w)
        ref_video = ref_img.reshape((bs * num_frame, three, h, w))
        video_x = self.extract_feat(ref_video)

        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(video_x, ref_img_metas, **kwargs)

        results = [[] for _ in range(bs)]

        # for each frame results
        for frame_id in range(num_frame):
            # fuse the final panoptic segmentation results.
            result = self.panoptic_fusion_head.simple_test(
                mask_cls_results,
                mask_pred_results[:, frame_id],
                [ref_img_metas[idx][frame_id] for idx in range(bs)],
                **kwargs
            )

            for i in range(len(result)):
                if 'pan_results' in result[i]:

                    result[i]['pan_results'] = result[i]['pan_results'].detach(
                    ).cpu().numpy()

                    # add the sem_seg results for vps and vss evaluation
                    if self.dataset == "kitti-step":
                        result[i]['sem_results'] = mapping_train_id_to_cat_id_segm(result[i]['sem_results'],
                                                        train_id_2_cat_id=kitt_step_train_id_2_cat_id)

                if 'ins_results' in result[i]:
                    labels_per_image, bboxes, mask_pred_binary, _ = result[i]['ins_results']
                    # add the id in the box field.
                    bboxes = torch.cat(
                        [torch.arange(len(bboxes), dtype=bboxes.dtype, device=bboxes.device)[:, None] + 1,
                         bboxes], dim=1)
                    # sort by the score
                    inds = torch.argsort(bboxes[:, -1], descending=True)
                    labels_per_image = labels_per_image[inds][:10] # only keep final top-10 in each image
                    bboxes = bboxes[inds][:10]
                    mask_pred_binary = mask_pred_binary[inds][:10]
                    bbox_results = bbox2result(bboxes, labels_per_image, self.num_things_classes)
                    mask_results = [[] for _ in range(self.num_things_classes)]
                    for j, label in enumerate(labels_per_image):
                        mask = mask_pred_binary[j].detach().cpu().numpy()
                        mask_results[label].append(mask)
                    result[i]['ins_results'] = bbox_results, mask_results  # default format as instance segmentation.

                results[i].append(result[i])

        if self.num_stuff_classes == 0:
            # HY : starting from here, the codes are for video instance segmentation.
            # THe codes for vis does not support vps anymore.
            for i in range(len(results)):
                for j in range(len(results[i])):
                    bbox_results, mask_results = results[i][j]['ins_results']
                    results[i][j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))

        return results

    def forward_test(self, imgs, img_metas, **kwargs):
        """Currently video seg model does not support aug test.
        So we only add batch input shape here
        """
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_metas)
            for img_id in range(batch_size):
                img_metas[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        for ref_img, ref_img_meta in zip(kwargs['ref_img'], kwargs['ref_img_metas']):
            batch_size = len(kwargs['ref_img_metas'])
            for batch_id in range(batch_size):
                num_frame = len(ref_img_meta)
                for frame_id in range(num_frame):
                    kwargs['ref_img_metas'][batch_id][frame_id]['batch_input_shape'] = tuple(ref_img.size()[-2:])

        return self.simple_test(img=imgs, img_metas=img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def _show_pan_result(self,
                         img,
                         result,
                         score_thr=0.3,
                         bbox_color=(72, 101, 241),
                         text_color=(72, 101, 241),
                         mask_color=None,
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=False,
                         wait_time=0,
                         out_file=None):
        """Draw `panoptic result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes  # for VOID label
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            segms=segms,
            labels=labels,
            class_names=self.CLASSES,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        super().train(mode=mode)
        if self.fix_backbone:
            self.backbone.train(mode=False)
        return self