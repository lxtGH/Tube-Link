# TubeLink 


### Train CMD



### Test CMD

```bash
PYTHONPATH=. python tools/test_video.py configs/video/mask2former_vipseg/video_r50_2frames_matching.py  ~/pretrained_models/uv_seg/m2former_r50_2frames_vipseg.pth --eval-dir work_dirs/vipseg/r50_2frames_results  --pre-eval --eval-offline PQ
```


### OFFLine Evaluation

```bash
PYTHONPATH=. python video_inference/eval_video.py results_path --eval_metrics VPQ --num_classes 124 --num_thing_classes 58
```

### Image Baseline


### Tube-Link

The default tube is 2 out of 4 frames. 




### To Do List 

1. Ablation Studies. (Xiangtai Li)
    1.1 Backbone ()
    1.2 Window Size (1,2,3,4) () (train/inference) why flexible 
    1.3 Tracker Loss Design (sampled reference frames design)
    1.4 TubeLink Tracker Head (masked/ROI align based)
    1.5 Tracker hyper-parameters ()
    1.6 Tracker head design ()

2. Visualization (VIP-Seg, KITTI-STEP, Cityscapes-DVPS), **base configs** (Haobo Yuan)
    Cityscape color map.
    Depth color maps.

3. Code Review (Dataloader -> Loss) (Haobo Yuan/Xiangtai Li)

4. More Results On KITTI-STEP, Cityscapes DVPS/VPS datasets (main demo for the paper.)
   4.1 Achieve more stronger results 
   4.2 Achieve the unified evaluation interface.

5. Add CopyPast Augmentation for KITTI STEP datasets and VSPW datasets. 

6. GT-RoI Align For Cityscapes-VPS/DVPS with w4/w6 training for better performance.




#### To Do List For ICCV version


1, Add YT-VIS baseline (VITA) and general VIS

2, Improved the VSPW dataset learning. 

3, Add clip-level copy paste for VSPW. 







#### Bug History 
This part record the errors that we meet during the development of this codebase. 


1. stuff/thing query index error 

2. per-tube instance id error 

3. instance id and gt-mask not matching error 
