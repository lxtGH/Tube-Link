# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
from mmdet.utils import get_root_logger

from mmdet.core import bbox2result, encode_mask_results
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector

from scipy.optimize import linear_sum_assignment


def video_split(total, tube_size, overlap=0):
    print("{} {} {}".format(total, tube_size, overlap))
    assert tube_size > overlap
    total -= overlap
    tube_size -= overlap

    if total % tube_size == 0:
        splits = total // tube_size
    else:
        splits = (total // tube_size) + 1
    
    ind_list = []
    for i in range(splits):
        ind_list.append((i + 1) * tube_size)
    
    diff = ind_list[-1] - total

    # currently only supports diff < splits
    if diff < splits:
        for i in range(diff):
            ind_list[splits - 1 - i] -= diff - i
    else:
        ind_list[splits - 1] -= diff
        assert ind_list[splits - 1] > 0
        print("Warning: {} / {}".format(total, tube_size))
    
    for idx in range(len(ind_list)):
        ind_list[idx] += overlap
    
    print(ind_list)
    return ind_list


@DETECTORS.register_module()
class TubeLinkVIS(SingleStageDetector):

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
                 fix_backbone=False,
                 interval=4,
                 overlap=0,
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

        self.logger = get_root_logger()
        self.logger.info("[Unified Video Segmentation] Using customized tube_link_vps segmentor.")

        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            self.backbone.train(mode=False)
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                self.logger.info(name + " is fixed.")

        self.interval = interval
        self.overlap = overlap

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
    
    def _match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=-1, keepdim=True)
        tgt_embds = tgt_embds / tgt_embds.norm(dim=-1, keepdim=True)
        cos_sim = torch.bmm(cur_embds, tgt_embds.transpose(1,2))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = []
        for i in range(len(cur_embds)):
            indice = linear_sum_assignment(C[i].transpose(0, 1))  # target x current
            indice = indice[1]  # permutation that makes current aligns to target
            indices.append(indice)

        return indices

    def simple_test(self, img, img_metas, ref_img, ref_img_metas, **kwargs):
        device = img.device
        del img
        bs, num_frame, three, h, w = ref_img.size()
        tube_inds = video_split(num_frame, self.interval, self.overlap)
        # (b, t, 3, h, w)
        if num_frame > 25:
            clips = [[], [] , [], []]
            num_clip = num_frame // 25 + 1
            step_size = num_frame // num_clip + 1
            self.logger.info("#frames: {}; #clips: {}; clip_size: {}".format(num_frame, num_clip, step_size))
            for i in range(num_clip):
                start = i*step_size
                end = min(num_frame, (i+1) * step_size)
                ref_video = ref_img[:, start:end].reshape(
                    (bs * (end - start), three, h, w))
                clip_x = self.extract_feat(ref_video)
                assert len(clip_x) == 4
                for idx, item in enumerate(clip_x):
                    clips[idx].append(item.cpu())
            video_x = []
            for item in clips:
                scale = torch.cat(item, dim=0)
                assert scale.size(0) == bs * num_frame, "{} vs {}".format(scale.size(0), bs * num_frame)
                video_x.append(scale)
        else:
            ref_video = ref_img.reshape((bs * num_frame, three, h, w))
            video_x = self.extract_feat(ref_video)
        del ref_video
        del ref_img

        ind_pre = 0
        cls_list = []
        mask_list = []
        query_list = []
        flag = False
        for ind in tube_inds:
            tube_x = [itm[ind_pre:ind].to(device=device) for itm in video_x]
            tube_metas = [itm[ind_pre:ind] for itm in ref_img_metas]
            _mask_cls_results, _mask_pred_results, query_feat = self.panoptic_head.simple_test_with_query(tube_x, tube_metas, **kwargs)
            cls_list.append(_mask_cls_results)
            if not flag:
                mask_list.append(_mask_pred_results.cpu())
                # mask_list.append(_mask_pred_results)
                flag = True
            else:
                mask_list.append(_mask_pred_results[:, self.overlap:].cpu())
                # mask_list.append(_mask_pred_results[:, self.overlap:])
            query_list.append(query_feat)

            ind_pre = ind
            ind_pre -= self.overlap
        
        if isinstance(video_x, List):
            del video_x[:]
        elif isinstance(video_x, Tuple):
            del video_x
        else:
            raise ValueError("video x should be either List or Tuple.")

        num_tubes = len(tube_inds)
        
        out_cls = [cls_list[0].cpu()]
        out_mask = [mask_list[0]]
        mask_list[0] = None
        out_embd = [query_list[0]]

        for i in range(1, num_tubes):
            indices = self._match_from_embds(out_embd[-1], query_list[i])
            indices = indices[0] # since bs == 1 forevers

            out_cls.append(cls_list[i][:, indices].cpu())
            out_mask.append(mask_list[i][:, :, indices])
            mask_list[i] = None
            out_embd.append(query_list[i][:, indices])

        mask_cls_results = sum(out_cls) / num_tubes
        mask_pred_results = torch.cat(out_mask, dim=1)
        del out_embd[:]

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

                if 'ins_results' in result[i]:
                    labels_per_image, bboxes, mask_pred_binary, _ = result[i]['ins_results']
                    # add the id in the box field.
                    bboxes = torch.cat(
                        [torch.arange(len(bboxes), dtype=bboxes.dtype, device=bboxes.device)[:, None] + 1,
                         bboxes], dim=1)
                    # sort by the score
                    inds = torch.argsort(bboxes[:, -1], descending=True)
                    labels_per_image = labels_per_image[inds][:30] # only keep final top-10 in each image
                    bboxes = bboxes[inds][:30]
                    mask_pred_binary = mask_pred_binary[inds][:30]
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
    

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        super().train(mode=mode)
        if self.fix_backbone:
            self.backbone.train(mode=False)
        return self
