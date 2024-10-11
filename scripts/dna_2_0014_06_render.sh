export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0

id_name=2_0014_06

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=30000
smpl_type=simple_smplx
use_extrapose_tuner=True
non_rigid_flag=True
non_rigid_use_extra_condition_flag=False
joints_opt_flag=True
extra_joints_batch=-1

exp_name=dna_github/${id_name}_uet_${use_extrapose_tuner}_nr_\
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