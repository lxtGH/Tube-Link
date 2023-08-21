# Copyright (c) OpenMMLab. All rights reserved.
import torch


def preprocess_video_panoptic_gt(
        gt_labels,
        gt_masks,
        gt_semantic_seg,
        gt_instance_ids,
        num_things,
        num_stuff,
        img_metas,
):
    num_classes = num_things + num_stuff
    num_frames = len(img_metas)

    thing_masks_list = []
    for frame_id in range(num_frames):
        thing_masks_list.append(gt_masks[frame_id].pad(
                img_metas[frame_id]['batch_input_shape'][:2], pad_val=0).to_tensor(
                dtype=torch.bool, device=gt_labels.device)
            )
    instances = torch.unique(gt_instance_ids[:, 1])
    things_masks = []
    labels = []
    for instance in instances:
        pos_ins = torch.nonzero(torch.eq(gt_instance_ids[:, 1], instance), as_tuple=True)[0]  # 0 is for redundant tuple
        labels_instance = gt_labels[:, 1][pos_ins]
        assert torch.allclose(labels_instance, labels_instance[0])
        labels.append(labels_instance[0])
        instance_frame_ids = gt_instance_ids[:, 0][pos_ins].to(dtype=torch.int32).tolist()
        instance_masks = []
        for frame_id in range(num_frames):
            frame_instance_ids = gt_instance_ids[gt_instance_ids[:, 0] == frame_id, 1]
            if frame_id not in instance_frame_ids:
                empty_mask = torch.zeros(
                    (img_metas[frame_id]['batch_input_shape'][:2]),
                    dtype=thing_masks_list[frame_id].dtype, device=thing_masks_list[frame_id].device
                )
                instance_masks.append(empty_mask)
            else:
                pos_inner_frame = torch.nonzero(torch.eq(frame_instance_ids, instance), as_tuple=True)[0].item()
                frame_mask = thing_masks_list[frame_id][pos_inner_frame]
                instance_masks.append(frame_mask)
        things_masks.append(torch.stack(instance_masks))

    things_masks = torch.stack(things_masks)
    things_masks = things_masks.to(dtype=torch.long)
    labels = torch.stack(labels)
    labels = labels.to(dtype=torch.long)

    return labels, things_masks