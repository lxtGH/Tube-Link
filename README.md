<br />
<p align="center">
  <h1 align="center">Tube-Link: A Flexible Cross Tube Baseline for Universal Video Segmentation</h1>
  <p align="center">
    Arxiv, 2023
    <br />
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://yuanhaobo.me/"><strong>Haobo Yuan</strong></a>
    ·
    <a href="https://zhangwenwei.cn/"><strong>Wenwei Zhang</strong></a>
    ·
    <a href="https://sites.google.com/view/guangliangcheng"><strong>Guangliang Cheng</strong></a>
    <br />
    <a href="https://oceanpang.github.io/"><strong>Jiangmiao Pang</strong></a>
    .
    <a href="https://www.mmlab-ntu.com/person/ccloy/"><strong>Chen Change Loy*</strong></a>
  </p>

  <p align="center">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
<br />

**Universal Video Segmentation Model For VSS, VPS and VIS**

![avatar](./assets/figs/teaser.png)


[[Paper]](https://arxiv.org/abs/2303.12782) [[CODE]](https://github.com/lxtGH/Tube-Link)


## Features

### $\color{#2F6EBA}{Universal\ Video\ Segmentation\ Model}$ 

- A new framework unifies online video segmentation methods and near online video segmentation methods.
- A new unified solution for three video segmentation tasks: VSS, VIS and VPS.

### $\color{#2F6EBA}{Explore\ the\ Cross\-Tube\ Relation}$ 

- The first video segmentation method that explores the cross tube relation.
- Proposed Tube-wise matching performs better Frame-wise matching.

### $\color{#2F6EBA}{Strong\ Performance}$  

- Achieves the strong performance on VIS, VSS and VPS datasets (five datasets) in one unified architecture.
- AchievesEven better performance than those specific architectures.

### Dataset 

See [Dataset.md](docs/DATASET.md)


### Install

See [Install.md](docs/INSTALL.md)


### Training, Evaluation, and Models

See [Tran.md](docs/TRAIN_EVALUATION_MODELS.md)


## Visualization Results

### [VIS] Youtube-VIS 2019
<details open>
<summary>Demo</summary>

![vis_demo_1](assets/figs/vis/vis_001.gif) 

![vis_demo_2](assets/figs/vis/vis_002.gif)

</details>

### [VPS] VIP-Seg

<details open>
<summary>Demo</summary>

![vps_demo_1](assets/figs/vps/vps_01.gif) 

![vps_demo_2](assets/figs/vps/vps_02.gif)

</details>

### [VSS] VSPW
<details open>
<summary>Demo</summary>

![vss_demo](assets/figs/vss/vspw.gif)

</details>

### [VPS] KITTI-STEP
<details open>
<summary>Demo</summary>

![vps_demo_3](assets/figs/vps/vps_03.gif)

</details>


## Citation

If you think both Tube-Link and its codebase are useful for your research, please consider to refer Tube-Link:

```bibtex
@article{li2023tube,
  title={Tube-link: A flexible cross tube baseline for universal video segmentation},
  author={Li, Xiangtai and Yuan, Haobo and Zhang, Wenwei and Cheng, Guangliang and Pang, Jiangmiao and Loy, Chen Change},
  journal={ICCV},
  year={2023}
}
```
