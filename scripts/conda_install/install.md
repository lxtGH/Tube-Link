# install with conda

install torch:
```commandline
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.2" libcusolver-dev
```

install mmcv from source (1.6.1):
```
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@d409eedc816fccfb1c8d57e5eed5f03bd075f327
```


install other packages (seaborn is required by TAO tracker; tqdm, ftfy, regex required by CLIP):
```
python -m pip install terminaltables pycocotools scipy seaborn tqdm ftfy regex
python -m pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
```
