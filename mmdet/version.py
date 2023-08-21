# Copyright (c) OpenMMLab. All rights reserved.
# The whole folder mmdet is borrowed from mmdet@3b72b12
# No other change to mmdet@3b72b12 except for:
# [220829] HY : modified datasets/coco_panoptic.py L47-49
# for compatibility with nv pycocotools.
__version__ = '2.25.1_3b72b12'
short_version = __version__


def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
