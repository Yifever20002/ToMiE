# (ICCV2025) ToMiE: Towards Explicit Exoskeleton for the Reconstruction of Complicated 3D Human Avatars 
### [Project Page](https://arxiv.org/abs/2410.08082) | [Paper](https://arxiv.org/abs/2410.08082)
>ToMiE: Towards Explicit Exoskeleton for the Reconstruction of Complicated 3D Human Avatars\
>[Yifan Zhan](https://yifever20002.github.io/yifanzhan.github.io/), [Qingtian Zhu](https://qtzhu.me/), [Muyao Niu](https://myniuuu.github.io/), Mingze Ma, Jiancheng Zhao \
>[Zhihang Zhong](https://zzh-tech.github.io/), Xiao Sun, Yu Qiao, Yinqiang Zheng

![image](https://github.com/Yifever20002/ToMiE/blob/main/images/teaser.png)

In this paper, we highlight a critical yet often overlooked factor in most 3D human tasks, namely modeling complicated 3D human with with hand-held objects or loose-fitting clothing. It is known that the parameterized formulation of SMPL is able to fit human skin; while hand-held objects and loose-fitting clothing, are difficult to get modeled within the unified framework, since their movements are usually decoupled with the human body. To enhance the capability of SMPL skeleton in response to this situation, we propose a growth strategy that enables the joint tree of the skeleton to expand adaptively. Specifically, our method, called ToMiE, consists of parent joints localization and external joints optimization.  For parent joints localization, we employ a gradient-based approach guided by both LBS blending weights and motion kernels. Once the external joints are obtained, we proceed to optimize their transformations in SE(3) across different frames, enabling rendering and explicit animation. ToMiE manages to outperform other methods across various cases with hand-held objects and loose-fitting clothing, not only in rendering quality but also by offering free animation of grown joints, thereby enhancing the expressive ability of SMPL skeleton for a broader range of applications.

## A. Prerequisite
### `Configure environment`
Create a virtual environment and install the required packages 

    conda create -n tomie python=3.7
    conda activate tomie
    pip install -r requirements.txt

Install submodules:

    export CUDA_HOME=/usr/local/cuda
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn

Install other requirements:

    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    pip install -r requirement_gauhuman.txt


### `Download SMPL(-X) model`

Download the SMPL(-X) from [here](https://drive.google.com/drive/folders/1ULFP2r1RLq5dBrvkK_R_4MTMOB8ej2V4?usp=drive_link) and put it under the main path.

### `Dataset`

For DNA-Rendering dataset, please download it from [here](https://dna-rendering.github.io/inner-download.html). You only need to download the ``xxxx_xx.smc'' and ``xxxx_xx_annots.smc'' files for each case. Our code will automatically preprocess the data during the first run and save the images, bkgd_masks, and model in the corresponding directory.

For ZJU_MoCap dataset, please refer to this [form](https://docs.google.com/forms/d/1QcTp5qIbIBn8PCT-EQgG-fOB4HZ9khpRkT3q2OnH2bs).


## B. Experiments

### Training

    sh scripts/dna_2_0041_10.sh

### Rendering

    sh scripts/dna_2_0041_10_render.sh

Here is a description of the ``dna_x_xxxx_xx.sh'' file:

    export CUDA_HOME=/usr/local/cuda
    export CUDA_VISIBLE_DEVICES=2
    id_name=2_0041_10                                    # case id, DNA-part2_id-number_sequence-number
    
    dataset=../dataset/DNA-Rendering/${id_name}/         # dataset path, change to yours
    iterations=30000
    smpl_type=simple_smplx                               # we use a simplified SMPLX without MANO
    use_extrapose_tuner=True                             # modular growth or not
    non_rigid_flag=True                                  # non-rigid deformation or not
    non_rigid_use_extra_condition_flag=False             
    joints_opt_flag=True                                 # optimizing joint locations or not
    extra_joints_batch=-1                                # fix joint number or gradient guided
    
    exp_name=dna_github/${id_name}_uet_${use_extrapose_tuner}_nr_\
    ${non_rigid_flag}_nruec_${non_rigid_use_extra_condition_flag}_jo_${joints_opt_flag}_ejb_${extra_joints_batch}
    
    python train.py -s $dataset --eval \
        --exp_name $exp_name \
        --motion_offset_flag \
        --smpl_type ${smpl_type} \
        --actor_gender neutral \
        --iterations ${iterations} \
        --use_extrapose_tuner ${use_extrapose_tuner} \
        --non_rigid_flag ${non_rigid_flag} \
        --non_rigid_use_extra_condition_flag ${non_rigid_use_extra_condition_flag}  \
        --joints_opt_flag ${joints_opt_flag} \
        --extra_joints_batch ${extra_joints_batch} \
        --port 6005 \
        --is_continue \
        --wandb_disable

We also have similar scripts for ZJU_MoCap dataset.

## C. Results

### Monocular Rendering

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**Ground Truth** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**ToMiE** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**Gauhuman**

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0007_04_gt.gif" alt="m0007_04_gt" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0007_04_tomie.gif" alt="m0007_04_to" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0007_04_gauhuman.gif" alt="m0007_04_ga" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0041_10_gt.gif" alt="m0041_10_gt" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0041_10_tomie.gif" alt="m0041_10_to" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0041_10_gauhuman.gif" alt="m0041_10_ga" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0800_07_gt.gif" alt="m0800_07_gt" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0800_07_tomie.gif" alt="m0800_07_to" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0800_07_gauhuman.gif" alt="m0800_07_ga" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0811_06_gt.gif" alt="m0811_06_gt" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0811_06_tomie.gif" alt="m0811_06_to" width="276" height="376"><img src="https://github.com/Yifever20002/ToMiE/blob/main/images/mono/0811_06_gauhuman.gif" alt="m0811_06_ga" width="276" height="376">

### 360° Rendering

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**ToMiE** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**Gauhuman**

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0007_04_tomie_rotate.gif" alt="m0007_04_gt" width="276" height="376"> <img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0007_04_gauhuman_rotate.gif" alt="m0007_04_to" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0041_10_tomie_rotate.gif" alt="m0041_10_gt" width="276" height="376"> <img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0041_10_gauhuman_rotate.gif" alt="m0041_10_to" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0800_07_tomie_rotate.gif" alt="m0800_07_gt" width="276" height="376"> <img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0800_07_gauhuman_rotate.gif" alt="m0800_07_to" width="276" height="376">

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0811_06_tomie_rotate.gif" alt="m0811_06_gt" width="276" height="376"> <img src="https://github.com/Yifever20002/ToMiE/blob/main/images/rotate/0811_06_gauhuman_rotate.gif" alt="m0811_06_to" width="276" height="376">

### Garments Animating

<img src="https://github.com/Yifever20002/ToMiE/blob/main/images/animating.gif" alt="animating" width="1138">

## D. Citation

If you find our work useful, please cite it as

    @misc{zhan2024tomiemodulargrowthenhanced,
      title={ToMiE: Towards Modular Growth in Enhanced SMPL Skeleton for 3D Human with Animatable Garments}, 
      author={Yifan Zhan and Qingtian Zhu and Muyao Niu and Mingze Ma and Jiancheng Zhao and Zhihang Zhong and Xiao Sun and Yu Qiao and Yinqiang Zheng},
      year={2024},
      eprint={2410.08082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.08082}, 
    }

## E. Acknowledgments

We appreciate for [GauHuman](https://github.com/skhu101/GauHuman), upon which we build our code implementation. 
We would also appreciate the code release of 
[3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release),
[GART](https://github.com/JiahuiLei/GART), 
[Im4D](https://github.com/zju3dv/im4d)
for comparison and evaluation.

Please also cite SMPL(-X) following

    @article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
    
    @inproceedings{SMPL-X:2019,
      title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
      author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      pages     = {10975--10985},
      year = {2019}
    }

