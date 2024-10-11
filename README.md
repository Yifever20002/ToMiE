# ToMiE: Towards Modular Growth in Enhanced SMPL Skeleton for 3D Human with Animatable Garments 
### [Project Page](https://ai4sports.opengvlab.com/Dyco/) | [Paper](https://arxiv.org/abs/2410.08082)
>ToMiE: Towards Modular Growth in Enhanced SMPL Skeleton for 3D Human with Animatable Garments\
>[Yifan Zhan](https://yifever20002.github.io/), [Qingtian Zhu](https://qtzhu.me/), [Muyao Niu](https://myniuuu.github.io/), Mingze Ma, Jiancheng Zhao, [Zhihang Zhong](https://zzh-tech.github.io/), Xiao Sun, Yu Qiao, Yinqiang Zheng\

![image](https://github.com/Yifever20002/ToMiE/blob/main/images/teaser.png)

## A. Prerequisite
### `Configure environment`
Create a virtual environment and install the required packages 

For Cuda11.3:

    conda create --name Dyco python=3.7 && conda activate Dyco
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

For Cuda11.8:

    conda create --name Dyco python=3.8 && conda activate Dyco
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

Install other requirements:

    pip install -r requirements.txt
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

### `Download SMPL model`

Copy the smpl model.

    SMPL_DIR=/path/to/smpl

    MODEL_DIR=$SMPL_DIR/smplify_public/code/models

    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

### `Download vgg.pth`

Download the vgg.pth from [here](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1).

    VGG_DIR=/path/to/vgg.pth

    cp $VGG_DIR third_parties/lpips/weights/v0.1/

### `Download RAFT model`
    cd RAFT && ./download_models.sh

## B. Experiments on I3D-Human Dataset
The I3D-Human Dataset focuses on capturing variations in clothing appearance under approximately identical poses. Compared with existing benchmarks, we outfit the subjects in loose clothing such as dresses and light jackets and
encourage movements involving acceleration or deceleration, such as sudden stops
after spinning, swaying, and flapping sleeves. Our capturing equipment consists of
10 DJI Osmo Action cameras, shooting at a frame rate of 100fps while synchronized
with an audio signal. The final processed dataset records 10k frames of sequence
from 6 subjects in total. Click [here](https://drive.google.com/drive/folders/1bj1pmr_FZE5ClR5ZCWpXD79ESaM8Ivvg?usp=sharing) to download our I3D-Human Dataset and copy it to /path/to/Dyco's parent/dataset/.

![image](https://github.com/Yifever20002/Dyco/blob/main/images/dataset.png)


### Train and Test

#### Baseine

    sh scripts/I3D-Human/ID1_1/ID1_1_humannerf.sh
    sh scripts/I3D-Human/ID1_1/ID1_1_humannerf_test.sh

#### + Conditions

    sh scripts/I3D-Human/ID1_1/ID1_1_posedelta.sh
    sh scripts/I3D-Human/ID1_1/ID1_1_posedelta_test.sh

You can also download our pretrained models [here](https://drive.google.com/drive/folders/1bj1pmr_FZE5ClR5ZCWpXD79ESaM8Ivvg?usp=sharing). Just put the "experiments" under /path/to/Dyco/experiments and you can simply run

    sh scripts/I3DHuman/ID1_1/ID1_1_posedelta_test.sh

to get the per-subject score reported in our paper:

|         | Novel Views |        |       | Novel Poses |        |       |
|:-------:|:-----------:|:------:|:-----:|:-----------:|:------:|:-----:|
|   Data  |     PSNR    |  SSIM  | LPIPS |     PSNR    |  SSIM  | LPIPS |
| ID1_1   |    31.31    | 0.9747 | 29.66 |    30.34    | 0.9706 | 34.41 |
| ID1_2   |    31.20    | 0.9731 | 32.67 |    30.45    | 0.9710 | 33.19 |
| ID2_1   |    29.80    | 0.9742 | 35.24 |    27.91    | 0.9653 | 45.44 |
| ID3_1   |    32.57    | 0.9733 | 40.60 |    31.77    | 0.9694 | 45.17 |
| Average |    31.22    | 0.9738 | 34.54 |    30.12    | 0.9691 | 39.55 |

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



