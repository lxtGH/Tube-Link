# Pytorch 1.13 CUDA 11.7
FROM nvcr.io/nvidia/pytorch:22.06-py3

# install ujson
RUN pip install ujson

# handle the timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="PRC" apt-get install tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && unlink /etc/localtime && ln -s /usr/share/zoneinfo/PRC /etc/localtime

# mmcv : 1.6.1
RUN until MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+git://github.com/open-mmlab/mmcv.git@d409eedc816fccfb1c8d57e5eed5f03bd075f327; do sleep 0.1; done

# git config
RUN git config --global --add safe.directory /opt/project

# install mmdet requirements
RUN until python -m pip install terminaltables; do sleep 0.1; done

# panopticapi
RUN until python -m pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0; do sleep 0.1; done

WORKDIR /opt/project

