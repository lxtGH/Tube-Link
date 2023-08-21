import torch

ckpt_path = "./vitae-base-vsa_in21k-mmdet.pth"

mm_cls_dic = torch.load(ckpt_path)

new_dic = {}

for k, v in mm_cls_dic.items():
    new_key = "backbone." + k
    print(new_key)
    new_dic[new_key] = v

torch.save(new_dic, "/vitae-base-vsa_in21k-mmdet_new.pth")