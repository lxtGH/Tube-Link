import argparse
import numpy as np
import os
from PIL import Image
from scripts.test_vspw.utils import Evaluator
import sys
import mmcv

eval_ = Evaluator(124)
eval_.reset()

def parse_args():
    parser = argparse.ArgumentParser(description='No description.')
    parser.add_argument('--gtdir', type=str, default='data/VSPW/')
    parser.add_argument('--preddir', type=str, default='./work_dirs/tmpdebugger')
    parser.add_argument('--eval-res', type=int, default=-1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    split = 'val.txt'
    args = parse_args()
    gtdir = args.gtdir
    predir = os.path.join(args.preddir, 'pred')

    eval_res = args.eval_res

    with open(os.path.join(gtdir,split),'r') as f:
        lines = f.readlines()
        for line in lines:
            videolist = [line[:-1] for line in lines]
    for video in videolist:
        for tar in os.listdir(os.path.join(gtdir,'data',video,'mask')):
            pred = os.path.join(predir,video,tar)
            tar_ = Image.open(os.path.join(gtdir,'data',video,'mask',tar))
            tar_ = np.array(tar_)
            if eval_res > 0:
                tar_ = mmcv.imrescale(
                    img=tar_,
                    scale=(eval_res, 100000),
                    return_scale=False,
                    interpolation='nearest',
                )
            tar_ = tar_[np.newaxis,:]
            pred_ = Image.open(pred)
            pred_ = np.array(pred_)
            pred_ = pred_[np.newaxis,:]

            if tar_.shape[-1] != pred_.shape[-1]:
                assert tar_.shape[-2] == pred_.shape[-2]
                assert abs(tar_.shape[-1] - pred_.shape[-1]) == 1
                if tar_.shape[-1] > pred_.shape[-1]:
                    tar_ = tar_[:,:,:-1]
                else:
                    pred_ = pred_[:,:,:-1]
            eval_.add_batch(tar_,pred_)

    Acc = eval_.Pixel_Accuracy()
    Acc_class = eval_.Pixel_Accuracy_Class()
    mIoU = eval_.Mean_Intersection_over_Union()
    FWIoU = eval_.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))