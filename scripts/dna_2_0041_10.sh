export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=1

id_name=2_0041_10

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=30000
smpl_type=simple_smplx
use_extrapose_tuner=True

exp_name=dna/${id_name}_use_extrp_tuner_${use_extrapose_tuner}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --use_extrapose_tuner ${use_extrapose_tuner} \
    --port 6005 \
#    --start_checkpoint output/${exp_name}/chkpnt18000.pth \
#    --wandb_disable

#outdir=output/${exp_name}
#
#python render.py -m $outdir \
#   --motion_offset_flag \
#   --smpl_type ${smpl_type} \
#   --actor_gender neutral \
#   --iteration ${iterations} \
#   --use_extrapose_tuner ${use_extrapose_tuner} \
#   --skip_train