export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=2

id_name=CoreView_393
dataset=../dataset/zju_mocap/${id_name}/
iterations=3000
smpl_type=smpl
use_extrapose_tuner=True
non_rigid_flag=True
non_rigid_use_extra_condition_flag=False
joints_opt_flag=True
extra_joints_batch=-1     # grad 3e-6

exp_name=zju/${id_name}_uet_${use_extrapose_tuner}_nr_\
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


