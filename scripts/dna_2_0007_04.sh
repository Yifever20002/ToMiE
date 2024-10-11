export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=2
# ok
id_name=2_0007_04

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=10000
smpl_type=simple_smplx
use_extrapose_tuner=True
non_rigid_flag=True
non_rigid_use_extra_condition_flag=False
joints_opt_flag=True
extra_joints_batch=-1

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