# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from PIL import Image

import numpy as np
import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results

# cityscapes
city_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(city_palette)
for i in range(zero_pad):
    city_palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(city_palette)
    return new_mask


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    pre_eval=False,
                    eval_dir=None):
    model.eval()
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    sequence_len = {}
    sequence_result = {}
    img_id_last = -1
    for _idx, data in enumerate(data_loader):
        assert len(data['img_metas']) == 1, "Only support single seq input"
        if data_loader.dataset.__class__.__name__ == 'YouTubeVISDataset':
            # Special case here
            seq_id = data['img_metas'].data[0][0]['ori_filename'].split('/')[0]
            img_id = 0
            assert seq_id not in sequence_len
            sequence_len[seq_id] = len(data['ref_img_metas'].data[0][0])
            sequence_result[seq_id] = None
            clip_len = data_loader.dataset.inf_len
            assert clip_len != -1
            clip_num = sequence_len[seq_id] // clip_len if sequence_len[seq_id] % clip_len == 0 \
                else (sequence_len[seq_id] // clip_len) + 1
            if hasattr(model.module, 'init_memory'):
                model.module.init_memory()
            result = []
            for idx in range(clip_num):
                start = idx * clip_len
                end = min(sequence_len[seq_id], (idx + 1) * clip_len)
                cur_ref_img = copy.deepcopy(data['ref_img'])
                cur_ref_img.data[0] = cur_ref_img.data[0][:, start:end]

                cur_ref_metas = copy.deepcopy(data['ref_img_metas'])
                cur_ref_metas.data[0][0] = cur_ref_metas.data[0][0][start:end]
                for _fid in range(start, end):
                    cur_ref_metas.data[0][0][_fid - start]['img_id'] = _fid
                data_cur = {
                    'img': data['img'],
                    'img_metas': data['img_metas'],
                    'ref_img': cur_ref_img,
                    'ref_img_metas': cur_ref_metas
                }
                with torch.no_grad():
                    result.extend(model(return_loss=False, rescale=True, **data_cur)[0])
            prog_bar.update(1)
            sequence_result[seq_id] = result
            continue

        # The img_id must be started from 0.
        # This is firm and no further change.
        _sample_id = 0
        _frame_id = -1
        seq_id = data['ref_img_metas'].data[0][_sample_id][_frame_id]['seq_id']
        img_id = data['ref_img_metas'].data[0][_sample_id][_frame_id]['img_id']

        if seq_id not in sequence_len:
            sequence_len[seq_id] = 0
            sequence_result[seq_id] = []
        sequence_len[seq_id] = img_id + 1

        if data['ref_img_metas'].data[0][_sample_id][0]['img_id'] == 0:
            if hasattr(model.module, 'init_memory'):
                model.module.init_memory()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        if show or out_dir:
            assert batch_size == 1
            # dump the image frames
            video_clip_tensor = data['ref_img'].data[0][0]
            video_metas = data['ref_img_metas'].data[0][0]
            imgs = tensor2imgs(video_clip_tensor, **video_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(video_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, video_metas)):
                img_id = img_meta['img_id']
                if img_id == img_id_last:
                    continue
                else:
                    img_id_last = img_id
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, "panoptic_vis", img_meta['ori_filename'])
                else:
                    out_file = None

                cat_path = os.path.join(out_dir, 'panoptic', str(seq_id),
                                        '{:06d}_{:06d}_cat.png'.format(seq_id, img_id))
                ins_path = os.path.join(out_dir, 'panoptic', str(seq_id),
                                        '{:06d}_{:06d}_ins.png'.format(seq_id, img_id))
                color_path = os.path.join(out_dir, 'color', str(seq_id),
                                        '{:06d}_{:06d}_color.png'.format(seq_id, img_id))

                sseg_results = result[0][i]["sem_results"]
                sseg_results = mmcv.imresize(sseg_results, (ori_w, ori_h), interpolation="nearest")

                color_sseg = colorize_mask(sseg_results)
                track_maps = np.zeros_like(sseg_results)

                mmcv.imwrite(sseg_results.astype(np.uint16), cat_path)
                mmcv.imwrite(track_maps.astype(np.uint16), ins_path)

                if not os.path.exists(os.path.join(out_dir, 'color', str(seq_id))):
                    os.makedirs(os.path.join(out_dir, 'color', str(seq_id)))

                color_sseg.save(color_path)

                # shot the panoptic results
                # model.module.show_result(
                #     img_show,
                #     result[0][i],
                #     bbox_color=PALETTE,
                #     text_color=PALETTE,
                #     mask_color=PALETTE,
                #     show=show,
                #     out_file=out_file,
                #     score_thr=show_score_thr)

        if isinstance(result[0][0], dict) and 'ins_results' in result[0][0]:
            for jj in range(len(result)):
                for j in range(len(result[jj])):
                    bbox_results, mask_results = result[jj][j]['ins_results']
                    result[jj][j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))

        if pre_eval:
            _sample_id = 0
            _frame_id = 0
            _seq_id = data['ref_img_metas'].data[0][_sample_id][_frame_id]['seq_id']
            _img_id = data['ref_img_metas'].data[0][_sample_id][_frame_id]['img_id']
            _len = img_id - _img_id + 1
            data_loader.dataset.pre_eval(result[0][:_len], eval_dir, _seq_id, _img_id)
        else:
            sequence_result[seq_id].extend(result[0])

        for _ in range(batch_size):
            prog_bar.update()

    if not pre_eval:
        for seq_id in sequence_result:
            sequence_result[seq_id] = sequence_result[seq_id][:sequence_len[seq_id]]
    else:
        sequence_result = None
    return sequence_result
