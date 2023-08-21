import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class ConcatVideoReferences(object):
    """Concat video references.

    If the input list contains at least two dicts, concat the input list of
    dict to one dict from 2-nd dict of the input list.

    Args:
        results (list[dict]): List of dict that contain keys such as 'img',
            'img_metas', 'gt_masks','proposals', 'gt_bboxes',
            'gt_bboxes_ignore', 'gt_labels','gt_semantic_seg',
            'gt_instance_ids'.

    Returns:
        list[dict]: The first dict of outputs is the same as the first
        dict of `results`. The second dict of outputs concats the
        dicts in `results[1:]`.
    """

    def __call__(self, results):
        assert (isinstance(results, list)), 'results must be list'
        outs = results[:1]
        for i, result in enumerate(results[1:], 1):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 1:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    outs[1]['img'] = np.concatenate(
                        (outs[1]['img'], np.expand_dims(img, -1)), axis=-1)
            for key in ['img_metas', 'gt_masks']:
                if key in result:
                    if i == 1:
                        result[key] = [result[key]]
                    else:
                        outs[1][key].append(result[key])
            for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_instance_ids',
            ]:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full(
                    (N, 1), i - 1, dtype=np.float32), value),
                    axis=1)
                if i == 1:
                    result[key] = value
                else:
                    outs[1][key] = np.concatenate((outs[1][key], value), axis=0)
            if 'gt_semantic_seg' in result:
                if i == 1:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][..., None, None]
                else:
                    outs[1]['gt_semantic_seg'] = np.concatenate(
                        (outs[1]['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)

            if 'gt_depth' in result:
                if i == 1:
                    result['gt_depth'] = result['gt_depth'][..., None, None]
                else:
                    outs[1]['gt_depth'] = np.concatenate(
                        (outs[1]['gt_depth'],
                         result['gt_depth'][..., None, None]),
                        axis=-1)
            if i == 1:
                outs.append(result)
        return outs


@PIPELINES.register_module()
class SeqDefaultFormatBundle(object):
    """Sequence Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "img_metas", "proposals", "gt_bboxes", "gt_instance_ids",
    "gt_match_indices", "gt_bboxes_ignore", "gt_labels", "gt_masks" and
    "gt_semantic_seg". These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_instance_ids: (1) to tensor, (2) to DataContainer
    - gt_match_indices: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor, \
                       (3) to DataContainer (stack=True)

    Args:
        ref_prefix (str): The prefix of key added to the second dict of input
            list. Defaults to 'ref'.
    """

    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        """Sequence Default formatting bundle call function.

        Args:
            results (list[dict]): List of two dicts.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle. Each key in the second dict of the input list
            adds `self.ref_prefix` as prefix.
        """
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            outs.append(_results)

        data = {}
        if self.ref_prefix == 'ref':
            # origin frames
            data.update(outs[0])
            # reference frames
            if len(outs) == 1:
                # for k in outs[0]:
                #     data[f'{self.ref_prefix}_{k}'] = None
                pass
            else:
                for k, v in outs[1].items():
                    data[f'{self.ref_prefix}_{k}'] = v
        elif self.ref_prefix is None:
            # origin frames
            data.update(outs[0])

        return data

    def default_format_bundle(self, results):
        """Transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle.
        """
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in [
            'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
            'gt_instance_ids', 'gt_match_indices', 'gt_clips_vis_emb'
        ]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['img_metas', 'gt_masks']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
        if 'gt_semantic_seg' in results:
            semantic_seg = results['gt_semantic_seg']
            if len(semantic_seg.shape) == 2:
                semantic_seg = semantic_seg[None, ...]
            else:
                semantic_seg = np.ascontiguousarray(
                    semantic_seg.transpose(3, 2, 0, 1))
            results['gt_semantic_seg'] = DC(
                to_tensor(semantic_seg), stack=True)
        if 'gt_depth' in results:
            gt_depth = results['gt_depth']
            if len(gt_depth.shape) == 2:
                gt_depth = gt_depth[None, ...]
            else:
                gt_depth = np.ascontiguousarray(
                    gt_depth.transpose(3, 2, 0, 1))
            results['gt_depth'] = DC(
                to_tensor(gt_depth), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class VideoCollect(object):
    """Collect data from the loader relevant to the specific task.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str]): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 keys,
                 meta_keys=None,
                 reject_empty=False,
                 split_ind=-1,
                 num_ref_imgs=0,
                 # no_obj_class is added for handling non-0  no-obj class
                 default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor',
                                    'flip', 'flip_direction', 'img_norm_cfg',
                                    'seq_id', 'img_id', 'is_video_data', 'no_obj_class')):
        self.keys = keys
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys,)
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

        self.reject_empty = reject_empty
        self.split_ind = split_ind
        self.num_ref_imgs = num_ref_imgs

    def __call__(self, results):
        """Call function to collect keys in results.

        The keys in ``meta_keys`` and ``default_meta_keys`` will be converted
        to :obj:mmcv.DataContainer.

        Args:
            results (list[dict] | dict): List of dict or dict which contains
                the data to collect.

        Returns:
            list[dict] | dict: List of dict or dict that contains the
            following keys:

            - keys in ``self.keys``
            - ``img_metas``
        """
        results_is_dict = isinstance(results, dict)
        if results_is_dict:
            results = [results]
        outs = []
        for _results in results:
            _results = self._add_default_meta_keys(_results)
            _results = self._collect_meta_keys(_results)
            outs.append(_results)

        if results_is_dict:
            outs[0]['img_metas'] = DC(outs[0]['img_metas'], cpu_only=True)

        if self.num_ref_imgs > 0:
            if len(results) != self.num_ref_imgs + 1:
                return None

        if self.reject_empty:
            if self.split_ind != -1:
                # potential bug here.
                if len(results[1]['gt_labels']) == 0 or len(results[1 + self.split_ind]['gt_labels']) == 0:
                    # print("None")
                    return None
            else:
                if len(results[0]['gt_labels']) == 0:
                    return None

        return outs[0] if results_is_dict else outs

    def _collect_meta_keys(self, results):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results['img_info']:
                img_meta[key] = results['img_info'][key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results


@PIPELINES.register_module()
class LabelConsistencyChecker:
    """This module is to make the annotations are consistent in each video.
    """

    def __init__(self, num_frames=5):
        self.num_frames = num_frames

    def __call__(self, results):
        ref_gt_instance_ids = results['ref_gt_instance_ids'].data
        ins_mul_nframe = ref_gt_instance_ids.size(0)
        if ins_mul_nframe % self.num_frames != 0:
            return None
        num_ins = ins_mul_nframe // self.num_frames
        ins_id_bucket = torch.zeros((num_ins,), dtype=torch.float)
        for i in range(ins_mul_nframe):
            frame_cur = i // num_ins
            ins_cur = i % num_ins
            if ref_gt_instance_ids[i][0] != frame_cur:
                return None
            if frame_cur == 0:
                ins_id_bucket[ins_cur] = ref_gt_instance_ids[i][1]
            else:
                if ref_gt_instance_ids[i][1] != ins_id_bucket[ins_cur]:
                    return None
        return results
