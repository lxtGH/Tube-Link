## Pretrained Models

Please download the pre-trained Mask2Former model from MMDetection official website

or they will be downloaded automatically into your .cache folder.


## Training and Inference Scripts

You make sure you have more disk space to 

### [VPS-VIPSeg]

Training with ResNet50 backbone
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/vipseg/vipseg_tb_r50_8e.py --work-dir ./your_path/ --no-validate
```

Training with Swin-B backbone
```bash
# train vipseg vps SwinB model
GPUS=32 bash tools/slurm_train.sh $PARTITION job_name configs/video/vipseg/vipseg_tb_swinb_6e.py --work-dir ./your_path/ --no-validate
```

Test and evaluate the trained model with STQ and VPQ.

```bash
PYTHONPATH=. python tools/test_video.py configs/video/mask2former_vipseg/video_r50_2frames_matching.py  ./your_path_to_trained_model.pth --eval-dir work_dirs/vipseg/r50_2frames_results  --pre-eval --eval-offline VPQ STQ
```


### [VSS-VIPSeg]
Train VIPSeg-VSS Swin Large model
```bash
GPUS=32 bash tools/slurm_train.sh $PARTITION job_name configs/video/vipseg_vss/video_swin_l_train_2frames_vspw_test_2frames.py --work-dir ./your_path/ --no-validate
```

Test and evaluate the trained model with SQ (mIoU).

```bash
PYTHONPATH=. python tools/test_video.py configs/video/mask2former_vipseg/video_r50_2frames_matching.py  ./your_path_to_trained_model.pth --eval-dir ./your_dump_file_path --pre-eval --eval-offline STQ
```


### [VIS-Youtube-2019/2021]
Train Youtube-VIS Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/exp_tubeminvis/y19_swin_l_010_tubemin_2_5k_5k_10k.py --work-dir ./your_path/ --no-validate
```

Inference the model for submission.
```bash
GPUS=8 bash tools/slurm_test.py $PARTITION job_name configs/video/exp_tubeminvis/y19_swin_l_010_tubemin_2_5k_5k_10k.py  ./your_path_to_trained_model.pth --format-only --eval-options resfile_path=/path/to/submission
```

### [VSS-VSPW]
Train VSPW Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/vspw/video_swin_l_train_6frames_6e_test_6frames_f3.py --work-dir ./your_path/ --no-validate
```

Inference the model.
```bash
PYTHONPATH=. python tools/test_video.py configs/video/vspw/video_swin_l_train_6frames_6e_test_6frames_f3.py --pre-eval --retrun-direct --eval-dir ./your_dump_file_path
```

## Model Zoo 


The trained checkpoints are all available at this [Google Drive](https://drive.google.com/drive/folders/1o18DY7B0r9OgJwQyao9nDGZu7lzJz806?usp=sharing)

The corresponding configs are in configs folder.

You can download and inference for reproducing the results in our paper.

Note that the model results on VIPSeg-VPS is a little higher than our paper reported. 