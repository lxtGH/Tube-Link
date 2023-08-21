# Author: Xiangtai Li
# Implement Temporal Contrastive Loss with Link

import copy

import mmcv
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmcv.ops import point_sample
import numpy as np
import torch
import torch.nn as nn

from mmdet.utils import get_root_logger
from mmdet.core import build_assigner, build_sampler
import torch.nn.functional as F

from mmdet.core import INSTANCE_OFFSET, bbox2result, encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, build_transformer_layer

from tracker.qdtrack.builder import build_tracker
from tracker.unitrack.utils.mask import tensor_mask2box


@DETECTORS.register_module()
class TubeLinkVPS2Frames(SingleStageDetector):

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 track_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 dataset="kitti-step",
                 tracker=None,
                 mlp_only=False,
                 # default split_index is 2, for the 4 frames inputs.
                 split_index=2,
                 num_emb_fcs=1,
                 ref_mode=False,
                 track_link=None,
                 dynamic_conv_cfg=None,
                 track_train_cfg=None,
                 bbox_roi_extractor=None,
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

        self.tracker = None
        self.tracker_cfg = tracker
        self.track_train_cfg = track_train_cfg
        self.frame_id = -1
        self.init_track_assigner_sampler()

        # split index (default is )
        self.split_index = split_index
        # add embedding fcs for the final stage queries for cross window tracking
        self.num_emb_fcs = num_emb_fcs
        act_cfg = dict(type='ReLU', inplace=True)
        self.mlp_only = mlp_only

        if not self.mlp_only:
            in_channels = 256
            out_channels = 256
            self.embed_fcs = nn.ModuleList()
            for _ in range(self.num_emb_fcs):
                self.embed_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
                self.embed_fcs.append(
                        build_norm_layer(dict(type='LN'), in_channels)[1])
                self.embed_fcs.append(build_activation_layer(act_cfg))

            self.fc_embed = nn.Linear(in_channels, out_channels)

        if track_head is not None:
            self.track_head = build_head(track_head)

            if bbox_roi_extractor is not None:
                self.track_roi_extractor = build_roi_extractor(
                    bbox_roi_extractor)

        self.ref_mode = ref_mode
        # add things self attention heads.
        self.track_link = track_link

        if track_link:
            _num_head = 8
            _dropout = 0.
            _in_channels = 256
            feedforward_channels = 1024
            num_ffn_fcs = 2
            ffn_act_cfg = dict(type='ReLU', inplace=True)
            # add tracking MHSA
            self.attention_previous_track = MultiheadAttention(
                _in_channels,
                _num_head,
                _dropout,
            )
            _, self.attention_previous_norm_track = build_norm_layer(
                dict(type='LN'),
                _in_channels
            )
            # add link ffn
            self.link_ffn_track = FFN(
                _in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=_dropout)
            self.link_ffn_norm_track = build_norm_layer(dict(type='LN'), _in_channels)[1]

            self.pre_thing_query = None

        # add dynamic conv cfg
        self.use_update_conv = False
        if dynamic_conv_cfg:
            self.use_update_conv = True
            self.kernel_update_conv = build_transformer_layer(dynamic_conv_cfg)

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.track_roi_assigner = build_assigner(
            self.track_train_cfg.assigner)
        self.track_share_assigner = False
        self.track_roi_sampler = build_sampler(
            self.track_train_cfg.sampler, context=self)
        self.track_share_sampler = False

    def forward_dummy(self, img, img_metas):
        raise NotImplementedError

    def add_ref_loss(self, loss_dict):
        track_loss ={}
        for k, v in loss_dict.items():
            track_loss[str(k)+"_ref"] = v
        return track_loss

    def _get_splited_index(self, ref_gt_labels, inter_frame_id=2):
        index = []
        ref_index = []
        thing_index = []
        ref_thing_index = []
        for i, label in enumerate(ref_gt_labels):
            if label[0] < inter_frame_id:
                index.append(i)
                if label[1] < self.num_things_classes:
                    thing_index.append(i)
            else:
                ref_index.append(i)
                if label[1] < self.num_things_classes:
                    ref_thing_index.append(i)

        return index, ref_index, thing_index, ref_thing_index

    def _generate_key_instance_index(self, gt_labels):
        # instance_ids_new = []
        instance_ids_thing_index = []
        for i, label in enumerate(gt_labels):
            if label[1] < self.num_things_classes:
                instance_ids_thing_index.append(i)
        return instance_ids_thing_index

    def link_thing_query(self, key_query_embs, ref_query_embs):

        key_query_embs = key_query_embs.permute(1, 0, 2)
        ref_query_embs = ref_query_embs.permute(1, 0, 2)

        key_query_embs = self.attention_previous_norm_track(
            self.attention_previous_track(
                query=key_query_embs,
                key=ref_query_embs,
                value=ref_query_embs,
                identity=key_query_embs
            ),
        )
        key_query_embs = key_query_embs.permute(1, 0, 2)

        key_query_embs_linked = self.link_ffn_norm_track(self.link_ffn_track(key_query_embs))

        return key_query_embs_linked

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      gt_instance_ids=None,
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
        super(SingleStageDetector, self).forward_train(ref_img[0], ref_img_metas[0])

        # step 0: split the tube into key and reference tube.
        # pre-pare the video instance id gts.
        # prepare the gt_match_indices
        key_index, ref_index, key_th_index, ref_th_index = self._get_splited_index(ref_gt_labels[0],
                                                                                   inter_frame_id=self.split_index)
        # we make the ref frame as key frames
        ref_gt_labels_new = [ref_gt_labels[0][key_index]]
        ref_gt_bboxes_new = [ref_gt_bboxes[0][key_index]]
        ref_gt_masks_new = [ref_gt_masks[0][:self.split_index]]
        ref_gt_semantic = ref_gt_semantic_seg[:, :self.split_index, ] if ref_gt_semantic_seg is not None else None
        ref_img_meta = [ref_img_metas[0][:self.split_index]]
        ref_instance_ids = [ref_gt_instance_ids[0][key_index]]
        ref_instance_ids_match = [ref_gt_instance_ids[0][key_th_index]]

        # print(ref_instance_ids_match)

        # cat the index
        index = torch.zeros(gt_labels[0].size()).to(gt_labels[0].device).unsqueeze(1).float()
        gt_labels_new = torch.cat([index, gt_labels[0].unsqueeze(1)], dim=-1)
        gt_instance_ids_new = torch.cat([index, gt_instance_ids[0].unsqueeze(1)], dim=-1)
        key_gt_labels = [gt_labels_new]
        key_gt_bboxes = gt_bboxes
        key_gt_masks = [gt_masks]
        key_gt_semantic = gt_semantic_seg.unsqueeze(1) if gt_semantic_seg is not None else None
        key_img_meta = [img_metas]
        key_instance_ids = [gt_instance_ids_new]
        key_instance_ids_match_index = self._generate_key_instance_index(key_gt_labels[0])
        key_instance_ids_match = [key_instance_ids[0][key_instance_ids_match_index]]

        # print(key_instance_ids_match)

        # filter repeated ids for matching
        ref_instance_ids_new = [[]]
        for ref_ids in ref_instance_ids_match[0]:
            ref_ids_new = ref_ids[1].long().cpu().data.numpy()
            if ref_ids_new not in ref_instance_ids_new[0]:
                ref_instance_ids_new[0].append(ref_ids_new)
        ref_instance_ids_new[0] = sorted(ref_instance_ids_new[0])

        key_instance_ids_new = [[]]
        for key_ids in key_instance_ids_match[0]:
            key_ids_new = key_ids[1].long().cpu().data.numpy()
            if key_ids_new not in key_instance_ids_new[0]:
                key_instance_ids_new[0].append(key_ids_new)
        key_instance_ids_new[0] = sorted(key_instance_ids_new[0])

        ref_gt_instance_id_list = ref_instance_ids_new
        key_gt_instance_id_list = key_instance_ids_new

        gt_pids_list = []

        for i in range(len(ref_gt_instance_id_list)):
            ref_ids = ref_gt_instance_id_list[i]
            gt_ids = key_gt_instance_id_list[i]
            gt_pids = [ref_ids.index(i) if i in ref_ids else -1 for i in gt_ids]
            gt_pids_list.append(torch.LongTensor([gt_pids]).to(img.device)[0])

        # used for training tracking heads
        gt_match_indices = gt_pids_list

        # step 1 : extract volume img features for current and reference images respectively.
        ref_img_input = ref_img[:, :self.split_index, ]
        # do not change the key frames
        key_img_input = img
        key_video_x = self.extract_feat(key_img_input)

        bs, num_frame, three, h, w = ref_img_input.size()  # (b,T,3,h,w)
        ref_img_input = ref_img_input.reshape((bs * num_frame, three, h, w))
        ref_video_x = self.extract_feat(ref_img_input)

        # step 2: forward the volume features for both current and reference volume features
        losses, scores, mask_preds, query_preds, thing_gt_labels, thing_gt_masks, feature_fpn = self.panoptic_head.forward_train(
            key_video_x,
            key_img_meta,
            key_gt_bboxes,
            key_gt_labels,
            key_gt_masks,
            key_gt_semantic,
            key_instance_ids,
            gt_bboxes_ignore=None
        )

        assert len(thing_gt_masks) == len(key_gt_instance_id_list[0])
        _ref_gt_bboxes_new = copy.deepcopy(ref_gt_bboxes_new)
        # _ref_gt_bboxes_new[0][:, 0] = _ref_gt_bboxes_new[0][:, 0]
        _ref_gt_labels_new = copy.deepcopy(ref_gt_labels_new)
        # _ref_gt_labels_new[0][:, 0] = _ref_gt_labels_new[0][:, 0]
        _ref_instance_ids = copy.deepcopy(ref_instance_ids)
        # _ref_instance_ids[0][:, 0] = _ref_instance_ids[0][:, 0]
        ref_losses, ref_scores, ref_mask_preds, ref_query_preds, ref_thing_gt_labels, ref_thing_gt_masks, ref_feature_fpn = \
            self.panoptic_head.forward_train(
                ref_video_x,
                ref_img_meta,
                _ref_gt_bboxes_new,
                _ref_gt_labels_new,
                ref_gt_masks_new,
                ref_gt_semantic,
                _ref_instance_ids,
                gt_bboxes_ignore=None,
                ref_mode=self.ref_mode
            )

        if not self.ref_mode:
            # add ref loss
            ref_losses = self.add_ref_loss(ref_losses)
            # add reference volume loss
            losses.update(ref_losses)

        # step 3: sample the matched indexes
        # score: [(b,n,h,w)], mask_preds: [ (b, t, n, h, w)]
        if self.num_stuff_classes == 0:
            object_score = scores[-1]  # (b, n, c)
            object_mask_pred = mask_preds[-1]  # (b, t, n, h, w)
            thing_object_query = query_preds.permute(1, 0, 2)  # (b, n, c)

            ref_object_score = ref_scores[-1]  # (b, n, c)
            ref_object_mask_pred = ref_mask_preds[-1]  # (b, t, n, h, w)
            ref_thing_object_query = ref_query_preds.permute(1, 0, 2)  # (b, n, c)
        else:
            object_score = scores[-1][:, :-self.num_stuff_classes, ]  # (b, n, c)
            object_mask_pred = mask_preds[-1][:, :, :-self.num_stuff_classes, ]  # (b, t, n, h, w)
            thing_object_query = query_preds.permute(1, 0, 2)[:, :-self.num_stuff_classes, ]  # (b, n, c)

            ref_object_score = ref_scores[-1][:, :-self.num_stuff_classes, ]  # (b, n, c)
            ref_object_mask_pred = ref_mask_preds[-1][:, :, :-self.num_stuff_classes, ]  # (b, t, n, h, w)
            ref_thing_object_query = ref_query_preds.permute(1, 0, 2)[:, :-self.num_stuff_classes, ]  # (b, n, c)

        # check whether has match
        gt_match_indices_array = [i.cpu().data.numpy() for i in gt_match_indices]
        nomatch = (np.array(gt_match_indices_array) == -1).all()

        if thing_gt_masks.size()[0] != 0 and not nomatch and ref_thing_gt_masks.size()[0] != 0:
            # for gt mask/labels and gt ref mask/labels
            # ref_thing_gt_labels: n_{th_gt}
            # ref_thing_gt_masks: n_{th_gt}, T, h, w

            object_mask_pred = object_mask_pred.permute(0, 2, 1, 3, 4).flatten(2, 3) # (b, n, T*h, W)
            thing_gt_masks = thing_gt_masks.flatten(1, 2) #  (n_{th_gt}, T*h, w)
            object_mask_feature = feature_fpn[-1]

            ref_object_mask_pred = ref_object_mask_pred.permute(0, 2, 1, 3, 4).flatten(2, 3)  # (b, n, T*h, W)
            ref_thing_gt_masks = ref_thing_gt_masks.flatten(1, 2)  # (n_{th_gt}, T*h, w)
            ref_object_mask_feature = ref_feature_fpn[-1]

            # ==== Dynamic Conv Update Query ==== #
            if self.use_update_conv:
                # adding dynamic conv to enhance query learning:
                sigmoid_masks = object_mask_pred.sigmoid()
                nonzero_inds = sigmoid_masks > 0.5
                sigmoid_masks = nonzero_inds.float()
                x_feat = torch.einsum('bnhw, bchw->bnc', sigmoid_masks, object_mask_feature)

                # update key thing query:
                thing_object_query = self.kernel_update_conv(x_feat, thing_object_query.unsqueeze(-2))
                thing_object_query = thing_object_query.permute(1, 0, 2)
                # update ref thing query:

                sigmoid_masks = ref_object_mask_pred.sigmoid()
                nonzero_inds = sigmoid_masks > 0.5
                sigmoid_masks = nonzero_inds.float()
                ref_x_feat = torch.einsum('bnhw, bchw->bnc', sigmoid_masks, ref_object_mask_feature)

                # update key thing query:
                ref_thing_object_query = self.kernel_update_conv(ref_x_feat, ref_thing_object_query.unsqueeze(-2))
                ref_thing_object_query = ref_thing_object_query.permute(1, 0, 2)

            # ==== Link Query Part ==== #
            if self.track_link:
                thing_object_query = self.link_thing_query(thing_object_query, ref_thing_object_query)

            # ===== Tracking Part -==== #
            # assign both key frame and reference frame tracking targets
            # we only consider two nealy-by frames
            key_sampling_results, ref_sampling_results = [], []
            # for key frames
            num_queries = object_score.size()[1]
            num_gts = thing_gt_masks.size()[0]
            point_coords = torch.rand((1, self.panoptic_head.num_points, 2), device=thing_gt_masks.device)
            mask_points_pred = point_sample(object_mask_pred[0].unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(
                1)
            # shape (num_gts, num_points)
            gt_points_masks = point_sample(thing_gt_masks.unsqueeze(1).float(),
                                           point_coords.repeat(num_gts, 1, 1)).squeeze(1)

            assign_result = self.track_roi_assigner.assign(
                object_score[0].detach(), mask_points_pred.detach(),
                thing_gt_labels, gt_points_masks, img_meta=None)

            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                object_mask_pred[0].detach(),
                thing_gt_masks)
            key_sampling_results.append(sampling_result)

            ## for the ref frames
            num_queries = ref_object_score.size()[1]
            num_gts = ref_thing_gt_masks.size()[0]
            point_coords = torch.rand((1, self.panoptic_head.num_points, 2), device=ref_thing_gt_masks.device)
            ref_mask_points_pred = point_sample(ref_object_mask_pred[0].unsqueeze(1),
                                            point_coords.repeat(num_queries, 1, 1)).squeeze(
                1)
            # shape (num_gts, num_points)
            ref_gt_points_masks = point_sample(ref_thing_gt_masks.unsqueeze(1).float(),
                                           point_coords.repeat(num_gts, 1, 1)).squeeze(1)

            assign_result = self.track_roi_assigner.assign(
                ref_object_score[0].detach(), ref_mask_points_pred.detach(),
                ref_thing_gt_labels, ref_gt_points_masks, img_meta=None)

            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                ref_object_mask_pred[0].detach(),
                ref_thing_gt_masks)
            ref_sampling_results.append(sampling_result)

            # step 4: forward the track head and calculate the match and tracking loss
            # for key emb
            key_emb = thing_object_query
            if not self.mlp_only:
                for emb_layer in self.embed_fcs:
                    key_emb = emb_layer(key_emb)
                key_obj_emb = self.fc_embed(key_emb)
            else:
                key_obj_emb = key_emb

            # sampling predicted GT mask
            key_emb_indexs = [res.pos_inds for res in key_sampling_results]
            object_feats_embed_list = []
            for i in range(len(key_emb_indexs)):
                object_feats_embed_list.append(key_obj_emb[:, key_emb_indexs[i], :].squeeze(0))
            key_feats = self._track_forward(object_feats_embed_list)
            # for ref emb
            ref_emb = ref_thing_object_query
            if not self.mlp_only:
                for emb_layer in self.embed_fcs:
                    ref_emb = emb_layer(ref_emb)
                ref_obj_emb = self.fc_embed(ref_emb)
            else:
                ref_obj_emb = ref_emb

            # sampling ref emb
            ref_emb_indexs = [res.pos_inds for res in ref_sampling_results]
            ref_object_feats_embed_list = []
            for i in range(len(ref_emb_indexs)):
                ref_object_feats_embed_list.append(ref_obj_emb[:, ref_emb_indexs[i], :].squeeze(0))

            ref_feats = self._track_forward(ref_object_feats_embed_list)

            match_feats = self.track_head.match(key_feats, ref_feats,
                                                key_sampling_results,
                                                ref_sampling_results)

            asso_targets = self.track_head.get_track_targets(
                gt_match_indices, key_sampling_results, ref_sampling_results)
            loss_track = self.track_head.loss(*match_feats, *asso_targets)

            losses.update(loss_track)

        else:
            loss_track = object_score.new_tensor(0.0)
            losses.update({"loss_track": loss_track})
            losses.update({"loss_track_aux": loss_track})
        return losses


    def init_memory(self):
        self.logger.info("[Unified Video Segmentation] Reset tracker.")
        self.tracker = build_tracker(self.tracker_cfg)
        self.frame_id = 0

    def _track_forward(self, track_feats, x=None, mask_pred=None):
        """Track head forward function used in both training and testing.
        We use mask pooling to get the fine grain features"""
        # if not self.training:
        #     mask_pred = [mask_pred]
        track_feats = torch.cat(track_feats, 0)

        track_feats = self.track_head(track_feats)

        return track_feats

    def simple_test(self, img, img_metas, ref_img, ref_img_metas, **kwargs):

        bs, num_frame, three, h, w = ref_img.size()
        # (b, t, 3, h, w)
        ref_video = ref_img.reshape((bs * num_frame, three, h, w))
        video_x = self.extract_feat(ref_video)

        mask_cls_results, mask_pred_results, query_feats, feature_fpn = \
            self.panoptic_head.simple_test_with_query(video_x, ref_img_metas, **kwargs)

        # whether the first frame
        frame_id = ref_img_metas[0][0]["img_id"]
        is_first = frame_id == 0

        self.frame_id = frame_id

        if self.use_update_conv:
            object_mask_pred = mask_pred_results.permute(0, 2, 1, 3, 4).flatten(2, 3)
            object_mask_feature = feature_fpn[-1]
            object_mask_pred = F.interpolate(object_mask_pred, size=object_mask_feature.size()[2:])
            sigmoid_masks = object_mask_pred.sigmoid()
            nonzero_inds = sigmoid_masks > 0.5
            sigmoid_masks = nonzero_inds.float()
            x_feat = torch.einsum('bnhw, bchw->bnc', sigmoid_masks, object_mask_feature)
            query_feats = query_feats.permute(1, 0, 2) # (b,n,c)
            # update key thing query:
            query_feats = self.kernel_update_conv(x_feat, query_feats.unsqueeze(-2))

        if is_first:
            pass
        else:
            if self.track_link:
                query_feats = self.link_thing_query(query_feats, self.pre_thing_query)

        results = [[] for _ in range(bs)]

        # for each frame results
        for frame_id in range(num_frame):
            # fuse the final panoptic segmentation results.
            assert self.panoptic_fusion_head.panoptic_mode in ['with_query', 'sort_with_query']

            result, query_lists = self.panoptic_fusion_head.simple_test(
                mask_cls_results,
                mask_pred_results[:, frame_id],
                [ref_img_metas[idx][frame_id] for idx in range(bs)],
                **kwargs
            )

            for i in range(len(result)):
                if 'pan_results' in result[i]:

                    result[i]['pan_results'] = result[i]['pan_results'].detach(
                    ).cpu().numpy()
                    result[i]['query_list'] = query_lists[i]

                if 'ins_results' in result[i]:
                    labels_per_image, bboxes, mask_pred_binary = result[i]['ins_results']
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

        results = self.match_panoptic(results, query_feats, mask_cls_results.cpu(), mask_pred_results.cpu())

        self.pre_thing_query = query_feats

        return results

    def match_panoptic(self, results, query_feats, mask_cls_results, mask_pred_results):
        # mask_cls_results, mask_pred_results are for bbox generation
        assert len(results) == 1
        assert self.frame_id != -1, "Not initialized"
        result = results[0]
        query_feats = query_feats[:, 0]
        clip_query_inds = None
        clip_pan_ids = None

        for frame in result:
            query_list = frame.pop('query_list')
            query_inds = torch.tensor([itm[0] for itm in query_list])
            pan_ids = torch.tensor([itm[1] for itm in query_list])
            if clip_query_inds is None and len(query_list) != 0:
                clip_query_inds = torch.unique(query_inds)
                clip_pan_ids = torch.unique(pan_ids)
            elif len(query_list) != 0:
                clip_query_inds = torch.unique(torch.cat([clip_query_inds, query_inds]))
                clip_pan_ids = torch.unique(torch.cat([clip_pan_ids, pan_ids]))
            else:
                clip_query_inds = None
                clip_pan_ids = None

        if clip_query_inds is None:

            return results

        clip_obj_feats = query_feats[clip_query_inds]
        clip_labels = clip_pan_ids % INSTANCE_OFFSET

        # get bbox for tracking
        mask_cls_results = mask_cls_results.softmax(dim=-1)
        frame_bbox = -1
        bbox = torch.zeros((len(clip_labels), 5), dtype=torch.float)
        bbox[:, 4] = torch.tensor(mask_cls_results[0][clip_query_inds][torch.arange(len(clip_labels)), clip_labels])
        # Modified by HB 19, OCT : query mask to pan mask
        # tracking_masks = mask_pred_results[0][frame_bbox][clip_query_inds]
        # tracking_masks = tracking_masks.sigmoid() >= 0.5
        tracking_masks = [torch.tensor(np.equal(result[frame_bbox]['pan_results'], itm), dtype=torch.float32)
                          for itm in clip_pan_ids]
        tracking_masks = torch.stack(tracking_masks)
        bbox[:, :4] = torch.tensor(tensor_mask2box(tracking_masks))

        # get tracking embedding
        track_emb = clip_obj_feats.unsqueeze(0)

        if not self.mlp_only:
            for emb_layer in self.embed_fcs:
                track_emb = emb_layer(track_emb)
            track_emb = self.fc_embed(track_emb)

        track_emb = track_emb.squeeze(0)

        track_feats = self._track_forward([track_emb])

        bboxes, labels, new_ids = self.tracker.match(
            bboxes=bbox,
            labels=clip_labels,
            masks=tracking_masks.unsqueeze(1),
            track_feats=track_feats,
            frame_id=self.frame_id)

        new_ids = new_ids + 1
        new_ids[new_ids == -1] = 0

        new_ids_len = len(new_ids)

        for result_per_frame in result:
            new_pan_map = copy.deepcopy(result_per_frame['pan_results'])
            for idx, clip_pan_id in enumerate(clip_pan_ids):
                clip_label = clip_pan_id % INSTANCE_OFFSET
                if idx < new_ids_len:  # only keep the remaining tracked id.
                    new_pan_map[result_per_frame['pan_results'] == clip_pan_id.item()] = \
                        clip_label.item() + new_ids[idx].item() * INSTANCE_OFFSET
                else:
                    new_pan_map[result_per_frame['pan_results'] == clip_pan_id.item()] = \
                        clip_label.item() + 0 * INSTANCE_OFFSET

            result_per_frame['pan_results'] = new_pan_map

        results[0] = result

        return results


    def generate_tracked_panoptic_seg(self, pan_results, tracking_maps, ids):

        if len(ids) == 0:
            return pan_results

        for i, id in enumerate(ids):
            pan_id = pan_results[tracking_maps[i]][0]
            semantic_seg_label = pan_id % INSTANCE_OFFSET
            index = tracking_maps[i] == 1
            id = id.int()
            pan_results[index] = semantic_seg_label + id * INSTANCE_OFFSET

        return pan_results

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