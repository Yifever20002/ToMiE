export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=1

id_name=2_0010_04

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=30000
smpl_type=simple_smplx

use_extrapose_tuner=False

exp_name=dna/${id_name}_use_extrp_tuner_${use_extrapose_tuner}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --use_extrapose_tuner ${use_extrapose_tuner} \
    --port 6001 \
#    --wandb_disable


#outdir=output/${exp_name}

#python render.py -m $outdir \
#     --motion_offset_flag \
#     --smpl_type ${smpl_type} \
#     --actor_gender neutral \
#     --iteration ${iterations} \
#     --skip_train