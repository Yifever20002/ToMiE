export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0

id_name=2_0014_06

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=30000
smpl_type=simple_smplx
use_extrapose_tuner=False

exp_name=dna/${id_name}_use_extrp_tuner_${use_extrapose_tuner}

outdir=output/${exp_name}

python render.py -m $outdir \
   --motion_offset_flag \
   --smpl_type ${smpl_type} \
   --actor_gender neutral \
   --iteration ${iterations} \
   --use_extrapose_tuner ${use_extrapose_tuner} \
   --vis_extrapose \
   --skip_train