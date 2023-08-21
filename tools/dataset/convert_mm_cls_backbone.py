import torch

ckpt_path = "/mnt/lustre/lixiangtai.vendor/pretrained/vitae/ViTAEv2_VSA_widePCM_B-22k-384finetune.pth.tar"

mm_cls_dic = torch.load(ckpt_path)["state_dict"]

new_dic = {}

for k, v in enumerate(mm_cls_dic):
    if "backbone" in v:
        new_key = v[9:]
    else:
        new_key = v
    print(new_key)
    new_dic[new_key] = mm_cls_dic[v]

torch.save(new_dic, "/home/lxt/pretrained_models/convnext/convnext-base_in21k-mmdet.pth")