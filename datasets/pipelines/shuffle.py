from mmdet.datasets import PIPELINES
import random


@PIPELINES.register_module()
class VideoShuffle:
    """Go ahead and just load image
    """

    def __init__(self, ignore_fist=True):
        self.ignore_fist = ignore_fist

    def __call__(self, results):
        instance_ids = []
        res = results[1:] if self.ignore_fist else results
        for _result in res:
            cur_ins_ids = _result['gt_instance_ids']
            cur_ins_ids = cur_ins_ids[_result['gt_labels'] < _result['thing_upper']]
            instance_ids.append(set(cur_ins_ids.tolist()))
        instance_ids = set.intersection(*instance_ids)
        if len(instance_ids) == 0:
            return None

        start = 1 if self.ignore_fist else 0
        for idx in range(start, len(results)):
            # randint a <= N <= b
            to_switch = random.randint(idx, len(results) - 1)
            if idx != to_switch:
                results[idx], results[to_switch] = results[to_switch], results[idx]
        return results
