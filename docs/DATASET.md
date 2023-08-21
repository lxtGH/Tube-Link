Please prepare the data structure as the following instruction:

The final dataset folder should be like this. 
```
root 
├── data
│   ├──  kitti-step
│   ├──  coco
│   ├──  VIPSeg
│   ├──  youtube_vis_2019
│   ├──  youtube_vis_2021
│   ├──  ovis
│   ├──  cityscapes
```


### [VPS] VIPSeg

Download the origin dataset from the official repo.\
Following official repo, we use resized videos for training and evaluation (The short size of the input is set to 720 while the ratio is keeped).

```
├── VIPSeg
│   ├──  images
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.jpg
│   ├──  panomasks 
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.png
│   ├──  panomasksRGB 
```



### [VIS] Youtube-VIS-2019
We use pre-processed json file according to mmtracking codebase.
see the "tools/dataset/youtubevis2coco.py".

```
├── youtube_vis_2019
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   │   ├── youtube_vis_2019_train.json
│   │   ├── youtube_vis_2019_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```


### [VIS] Youtube-VIS-2021

Follow the same procedure as Youtube-VIS-2019.

```
├── youtube_vis_2021
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   │   ├── youtube_vis_2021_train.json
│   │   ├── youtube_vis_2021_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```

### [VIS] OVIS

Follow the same procedure as Youtube-VIS-2019/2021.


### [VSS] VSPW-480P

```
├── VSPW
│   ├──  data
│   │   ├── 1812_5cs_pqWAcHY
│   │   ├── 1803_H1jujj-OTvA       
│   │   ├── ...
│   ├── data.txt
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
```

### [VPS] KITTI-STEP

Download the KITTI-STEP from the official website. 

Then run the scripts in scripts/kitti_step_prepare.py.
You will get such format.
You can get the pre-process format in https://huggingface.co/LXT/VideoK-Net/tree/main

```
├── kitti-step
│   ├──  video_sequence
│   │   ├── train
            ├──00018_000331_leftImg8bit.png
            ├──000018_000331_panoptic.png
            ├──****
│   │   ├── val
│   │   ├── test 
```

