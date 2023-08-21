# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask import mask2bbox
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class MaskFormerFusionHead(BasePanopticFusionHead):

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 panoptic_mode="default",
                 sem_seg_on=False,
                 **kwargs):
        super().__init__(num_things_classes, num_stuff_classes, test_cfg,
                         loss_panoptic, init_cfg, **kwargs)
        self.panoptic_mode = panoptic_mode
        self.sem_seg_on = sem_seg_on

    def forward_train(self, **kwargs):
        """MaskFormerFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)


        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1

        return panoptic_seg

    def panoptic_postprocess_with_query(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        query_index = torch.arange(len(mask_cls), device=mask_cls.device)

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        query_index = query_index[keep]
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)

        query_list = []
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                query_ind_cur = query_index[k].item()
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        cur_id = pred_class + (query_ind_cur + 1) * INSTANCE_OFFSET
                        panoptic_seg[mask] = cur_id
                        query_list.append((query_ind_cur, cur_id))

        return panoptic_seg, query_list

    def panoptic_postprocess_sort_score(self, mask_cls,  mask_pred):

        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        overlap_thr = self.test_cfg.get('overlap_thr', 0.6)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        cur_mask_ids = cur_prob_masks.argmax(0)

        sorted_inds = torch.argsort(-cur_scores)
        current_segment_id = 0
        for k in sorted_inds:
            pred_class = cur_classes[k].item()
            isthing = pred_class < self.num_things_classes
            if isthing and cur_scores[k] < object_mask_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < overlap_thr:
                    continue
                current_segment_id += 1
                if not isthing:
                    panoptic_seg[mask] = pred_class
                else:
                    panoptic_seg[mask] = (pred_class + current_segment_id * INSTANCE_OFFSET)

        return panoptic_seg

    def panoptic_postprocess_sort_score_query(self, mask_cls,  mask_pred):

        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.3)
        overlap_thr = self.test_cfg.get('overlap_thr', 0.6)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        query_index = torch.arange(len(mask_cls))
        keep = labels.ne(self.num_classes)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        query_index = query_index[keep]

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        cur_mask_ids = cur_prob_masks.argmax(0)

        sorted_inds = torch.argsort(-cur_scores)

        query_list = []

        for k in sorted_inds:
            pred_class = cur_classes[k].item()
            query_ind_cur = query_index[k].item()
            isthing = pred_class < self.num_things_classes
            if isthing and cur_scores[k] < object_mask_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < overlap_thr:
                    continue

                if not isthing:
                    panoptic_seg[mask] = pred_class
                else:
                    cur_id = pred_class + (query_ind_cur + 1) * INSTANCE_OFFSET
                    panoptic_seg[mask] = cur_id
                    query_list.append((query_ind_cur, cur_id))

        return panoptic_seg, query_list

    def panoptic_postprocess_sort_score_query_sem_seg_only(self, mask_cls,  mask_pred):

        cls_score = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # directly obtain the semantic segmentation results to replace the panoptic segmentation results.
        seg_logits = torch.einsum('qc,qhw->chw', cls_score, mask_pred)
        seg_labels = seg_logits.argmax(0)

        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.3)
        overlap_thr = self.test_cfg.get('overlap_thr', 0.6)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        query_index = torch.arange(len(mask_cls))
        keep = labels.ne(self.num_classes)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        query_index = query_index[keep]

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        cur_mask_ids = cur_prob_masks.argmax(0)

        sorted_inds = torch.argsort(-cur_scores)

        query_list = []

        for k in sorted_inds:
            pred_class = cur_classes[k].item()
            query_ind_cur = query_index[k].item()
            isthing = pred_class < self.num_things_classes
            if isthing and cur_scores[k] < object_mask_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < overlap_thr:
                    continue

                if not isthing:
                    panoptic_seg[mask] = pred_class
                else:
                    cur_id = pred_class + (query_ind_cur + 1) * INSTANCE_OFFSET
                    panoptic_seg[mask] = cur_id
                    query_list.append((query_ind_cur, cur_id))

        return seg_labels, query_list

    def panoptic_postprocess_focal_sort_score_sperate(self, mask_cls,  mask_pred):
        # mask_cls:
        # mask_pred:
        cls_scores, _ = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid().squeeze()
        thing_scores = cls_scores[:-self.num_stuff_classes][:, :self.
            num_classes]
        thing_mask_preds = mask_pred[:-self.num_stuff_classes]
        thing_scores, topk_indices = thing_scores.flatten(0, 1).topk(
            self.test_cfg.max_per_image, sorted=True)
        mask_indices = topk_indices // self.num_things_classes
        thing_labels = topk_indices % self.num_things_classes
        thing_masks = thing_mask_preds[mask_indices]

        stuff_scores = cls_scores[
                       -self.num_proposals:][:, self.num_things_classes:].diag()
        stuff_scores, stuff_inds = torch.sort(stuff_scores, descending=True)
        stuff_masks = mask_pred[self.num_proposals:][stuff_inds]
        stuff_labels = stuff_inds + self.num_things_classes  # (sum the index)

        total_masks = torch.cat([thing_masks, stuff_masks], dim=0)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        # final seg masks
        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        # hyper parameters set
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.5)
        overlap_thr = self.test_cfg.get('overlap_thr', 0.6)
        h, w = mask_pred.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=mask_pred.device)

        # sort to get the final maps
        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_things_classes
            if isthing and total_scores[k] < object_mask_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < overlap_thr:
                    continue
                current_segment_id += 1
                if not isthing:
                    panoptic_seg[mask] = pred_class
                else:
                    panoptic_seg[mask] = (pred_class + current_segment_id * INSTANCE_OFFSET)

        return panoptic_seg

    def semantic_postprocess(self, mask_cls, mask_pred):
        """Semantic segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Semantic segment result of shape \
                (cls_out_channels, h, w).
        """
        # TODO add semantic segmentation result
        raise NotImplementedError

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary, query_indices

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        query_lists = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution, default false.
                ori_height, ori_width = meta['ori_shape'][:2]
                # print("rescale")
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                if self.panoptic_mode == "sort":
                    pan_results = self.panoptic_postprocess_sort_score(
                        mask_cls_result, mask_pred_result)
                elif self.panoptic_mode == "joint_focal":
                    pan_results = self.panoptic_postprocess_focal_sort_score_jointly(mask_cls_result, mask_pred_results)
                elif self.panoptic_mode == 'sperate_focal':
                    pan_results = self.panoptic_postprocess_focal_sort_score_sperate(mask_cls_result, mask_pred_results)
                elif self.panoptic_mode == 'with_query':
                    pan_results, query_list = self.panoptic_postprocess_with_query(mask_cls_result, mask_pred_result)
                    query_lists.append(query_list)
                elif self.panoptic_mode == 'sort_with_query':
                    pan_results, query_list = self.panoptic_postprocess_sort_score_query(mask_cls_result, mask_pred_result)
                    query_lists.append(query_list)
                elif self.panoptic_mode == 'sem_seg_only_with_query':
                    pan_results, query_list = self.panoptic_postprocess_sort_score_query_sem_seg_only(mask_cls_result,
                                                                                     mask_pred_result)
                    query_lists.append(query_list)
                else:
                    pan_results = self.panoptic_postprocess(
                        mask_cls_result, mask_pred_result)

                result['pan_results'] = pan_results


                # obtain semantic results from panoptic results
                pan_results_cpu = pan_results.cpu().numpy()
                ids = np.unique(pan_results_cpu)[::-1]
                legal_indices = ids != self.num_classes  # for VOID label
                ids = ids[legal_indices]
                labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
                sem_seg_results = np.ones_like(pan_results_cpu) * 255

                for i, label in enumerate(labels):
                    mask = pan_results_cpu == ids[i]
                    sem_seg_results[mask] = label

                result['sem_results'] = sem_seg_results

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results
            results.append(result)

        if self.panoptic_mode in ['sem_seg_only_with_query', 'with_query', 'sort_with_query']:
            return results, query_lists

        return results


