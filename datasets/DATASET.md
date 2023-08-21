Please prepare dataset in the following format.

Image DataSet for Pre-training (COCO, Cityscapes STEP, Mapillary).
Video DataSet for training and evaluation (KITII-STEP, VIPSeg, Cityscapes-DVPS datasets, Youtube-VIS, Our Open-Vocabulary VIS)


### COCO dataset

default setting as mmdet.


```
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   ├── train2017
│   ├── val2017
│   ├── panoptic_{train,val}2017/  # png annotations
```



### Cityscapes (STEP) dataset

```
├── cityscapes
│   ├── annotations
│   │   ├── instancesonly_filtered_gtFine_train.json # coco instance annotation file(COCO format)
│   │   ├── instancesonly_filtered_gtFine_val.json
│   │   ├── cityscapes_panoptic_train.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val.json  
│   ├── leftImg8bit
│   ├── gtFine
│   │   ├──cityscapes_panoptic_{train,val}/  # png annotations
│   │   
```



### Mapillary dataset


```
├── mapillary
│   ├── annotations
│   │   ├── panoptic_train.json # from the original annotation
│   │   ├── panoptic_val.json  
│   ├── training # training images and segmentation annotations
│   │   ├──images # origin images 
│   │   ├──instances # semantic segmentation annotations
│   │   ├──labels
│   │   ├──panoptic # panoptic png annotations
│   ├── validation # validation images and segmentation annotations
│   │   ├──images # origin images 
│   │   ├──instances # semantic segmentation annotations
│   │   ├──labels
│   │   ├──panoptic # panoptic png annotations
```



### KITTI-STEP dataset




### Youtube-VIS (2019/2021)

```
├── youtubevis
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```

### VIPSeg dataset



### Cityscape DVPS (with Depth) Dataset



### Open Vocabulary VIS 



### Notations on Panoptic Segmentation file format

