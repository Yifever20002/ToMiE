# ToMiE: Towards Modular Growth in Enhanced SMPL Skeleton for 3D Human with Animatable Garments 
### [Project Page](https://arxiv.org/abs/2410.08082) | [Paper](https://arxiv.org/abs/2410.08082)
>ToMiE: Towards Modular Growth in Enhanced SMPL Skeleton for 3D Human with Animatable Garments\
>[Yifan Zhan](https://yifever20002.github.io/), [Qingtian Zhu](https://qtzhu.me/), [Muyao Niu](https://myniuuu.github.io/), Mingze Ma, Jiancheng Zhao \
>[Zhihang Zhong](https://zzh-tech.github.io/), Xiao Sun, Yu Qiao, Yinqiang Zheng

![image](https://github.com/Yifever20002/ToMiE/blob/main/images/teaser.png)

In this paper, we highlight a critical yet often overlooked factor in most 3D human tasks, namely modeling humans with complex garments. It is known that the parameterized formulation of SMPL is able to fit human skin; while complex garments, e.g., hand-held objects and loose-fitting garments, are difficult to get modeled within the unified framework, since their movements are usually decoupled with the human body. To enhance the capability of SMPL skeleton in response to this situation, we propose a modular growth strategy that enables the joint tree of the skeleton to expand adaptively. Specifically, our method, called ToMiE, consists of parent joints localization and external joints optimization. For parent joints localization, we employ a gradient-based approach guided by both LBS blending weights and motion kernels. Once the external joints are obtained, we proceed to optimize their transformations in SE(3) across different frames, enabling rendering and explicit animation. ToMiE manages to outperform other methods across various cases with garments, not only in rendering quality but also by offering free animation of grown joints, thereby enhancing the expressive ability of SMPL skeleton for a broader range of applications.

## A. Prerequisite
### `Configure environment`
Create a virtual environment and install the required packages 

    conda create -n tomie python=3.7
    conda activate tomie
    pip install -r requirements.txt

Install submodules:

    export CUDA_HOME=/usr/local/cuda
    pip install submodules/depth-diff-gaussian-rasterization
    pip install submodules/simple-knn

Install other requirements:

    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    pip install -r requirement_gauhuman.txt


### `Download SMPL(-X) model`

Download the SMPL(-X) from [here](https://drive.google.com/drive/folders/1ULFP2r1RLq5dBrvkK_R_4MTMOB8ej2V4?usp=drive_link) and put it under the main path.

### `Dataset`

For DNA-Rendering dataset, please download it from [here](https://dna-rendering.github.io/inner-download.html). You only need to download the ``xxxx_xx.smc'' and ``xxxx_xx_annots.smc'' files for each case. Our code will automatically preprocess the data during the first run and save the images, bkgd_masks, and model in the corresponding directory.

For ZJU_Mocap dataset, please refer to [mlp_maps](https://github.com/zju3dv/mlp_maps/blob/master/INSTALL.md).


## B. Experiments on DNA-Rendering Dataset

### Training

    sh scripts/dna_2_0041_10.sh

#### Rendering

    sh scripts/dna_2_0041_10_render.sh

Here is a description of the ``dna_x_xxxx_xx.sh'' file:

## C. Experiments on ZJU-MoCap Dataset

### Data preparation
Fill in the [form](https://docs.google.com/forms/d/1QcTp5qIbIBn8PCT-EQgG-fOB4HZ9khpRkT3q2OnH2bs) to download the dataset.

Create a soft link:

    ln -s /path/to/zju_mocap data/zju

Then preprocess the data. Take Subject-390 for example:

    tar -xvf CoreView_390.tar.gz
    cd tools/prepare_zju_mocap
    python prepare_dataset.py --cfg=configs/390_train.yaml
    python prepare_dataset.py --cfg=configs/390_novelview.yaml
    python prepare_dataset.py --cfg=configs/390_novelpose.yaml

### Train and Test

#### + Conditions

    sh scripts/zju_mocap/313/313_posedelta.sh
    sh scripts/zju_mocap/313/313_posedelta_test.sh



