export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=0

id_name=CoreView_387
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

outdir=output/${exp_name}

python render.py -m $outdir \
   --motion_offset_flag \
   --smpl_type ${smpl_type} \
   --actor_gender neutral \
   --iteration ${iterations} \
   --use_extrapose_tuner ${use_extrapose_tuner} \
   --non_rigid_flag ${non_rigid_flag} \
   --non_rigid_use_extra_condition_flag ${non_rigid_use_extra_condition_flag}  \
   --joints_opt_flag ${joints_opt_flag} \
   --vis_extrapose \
   --skip_train \
#   --mono_test
