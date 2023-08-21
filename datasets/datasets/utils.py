from typing import Dict

import numpy as np

from mmdet.core import INSTANCE_OFFSET


class SeqObj:
    """
    This is a seq object class for querying the image for constructing sequence.
    DIVISOR : This divisor is orthogonal with panoptic class-instance divisor (should be large enough).
    """
    DIVISOR = 1000000

    def __init__(self, the_dict: Dict):
        self.dict = the_dict
        assert 'seq_id' in self.dict and 'img_id' in self.dict

    def __hash__(self):
        return self.dict['seq_id'] * self.DIVISOR + self.dict['img_id']

    def __eq__(self, other):
        return self.dict['seq_id'] == other.dict['seq_id'] and self.dict['img_id'] == other.dict['img_id']

    def __getitem__(self, attr):
        return self.dict[attr]


def vpq_eval(element, num_classes=19, max_ins=10000, ign_id=255):
    import six
    pred_ids, gt_ids = element
    offset = 1e9
    num_cat = num_classes + 1

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        if gt_cat != pred_cat:
            continue
        union = (
                gt_areas[gt_id] + pred_areas[pred_id] - int_area -
                prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return iou_per_class, tp_per_class, fn_per_class, fp_per_class


def pan_mm2hb(pred_pan_map, num_classes, divisor=10000):
    pan_seg_map = - np.ones_like(pred_pan_map)
    for itm in np.unique(pred_pan_map):
        if itm >= INSTANCE_OFFSET:
            # cls labels (from segmentation maps)
            cls = itm % INSTANCE_OFFSET
            # id labels (from tracking maps)
            ins = itm // INSTANCE_OFFSET
            pan_seg_map[pred_pan_map == itm] = cls * divisor + ins
        elif itm == num_classes:
            pan_seg_map[pred_pan_map == itm] = num_classes * divisor
        else:
            pan_seg_map[pred_pan_map == itm] = itm * divisor
    assert -1 not in pan_seg_map
    return pan_seg_map


def stq(element, num_classes=19, max_ins=10000, ign_id=255, num_things=8, label_divisor=1e4, ins_divisor=1e7):
    y_pred, y_true = element
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # semantic eval
    semantic_label = y_true // max_ins
    semantic_prediction = y_pred // max_ins
    semantic_label = np.where(semantic_label != ign_id,
                              semantic_label, num_classes)
    semantic_prediction = np.where(semantic_prediction != ign_id,
                                   semantic_prediction, num_classes)
    semantic_ids = np.reshape(semantic_label, [-1]) * label_divisor + np.reshape(semantic_prediction, [-1])

    # instance eval
    instance_label = y_true % max_ins
    label_mask = np.less(semantic_label, num_things)
    prediction_mask = np.less(semantic_label, num_things)
    is_crowd = np.logical_and(instance_label == 0, label_mask)

    label_mask = np.logical_and(label_mask, np.logical_not(is_crowd))
    prediction_mask = np.logical_and(prediction_mask, np.logical_not(is_crowd))

    seq_preds = y_pred[prediction_mask]
    seg_labels = y_true[label_mask]

    non_crowd_intersection = np.logical_and(label_mask, prediction_mask)
    intersection_ids = (y_true[non_crowd_intersection] * ins_divisor + y_pred[non_crowd_intersection])
    return semantic_ids, seq_preds, seg_labels, intersection_ids
