import argparse
import numpy as np
import os
from PIL import Image
#from utils import Evaluator
import sys


def get_common(list_,predlist,clip_num,h,w):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))

                 
        for j in range(1,clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs


def parse_args():
    parser = argparse.ArgumentParser(description='No description.')
    parser.add_argument('--gtdir', type=str, default='data/VSPW/')
    parser.add_argument('--preddir', type=str, default='./work_dirs/tmpdebugger')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gtdir = args.gtdir
    predir = os.path.join(args.preddir, 'pred')
    split = 'val.txt'

    with open(os.path.join(gtdir,split),'r') as f:
        lines = f.readlines()
        for line in lines:
            videolist = [line[:-1] for line in lines]
    total_acc=[]

    clip_num=16


    for video in videolist:
        if video[0]=='.':
            continue
        imglist = []
        predlist = []

        images = sorted(os.listdir(os.path.join(gtdir,'data',video,'mask')))

        if len(images)<=clip_num:
            continue
        for imgname in images:
            if imgname[0]=='.':
                continue
            img = Image.open(os.path.join(gtdir,'data',video,'mask',imgname))
            w,h = img.size
            img = np.array(img)
            imglist.append(img)
            pred = Image.open(os.path.join(predir,video,imgname))
            pred = np.array(pred)
            predlist.append(pred)
            
        accs = get_common(imglist,predlist,clip_num,h,w)
        print(sum(accs)/len(accs))
        total_acc.extend(accs)
    Acc = np.array(total_acc)
    Acc = np.nanmean(Acc)
    print(predir)
    print('*'*10)
    print('VC{} score: {} on {} set'.format(clip_num,Acc,split))
    print('*'*10)

