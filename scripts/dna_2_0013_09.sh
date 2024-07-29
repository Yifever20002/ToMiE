export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=2

id_name=2_0013_09

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=30000
smpl_type=simple_smplx

exp_name=dna/${id_name}_100_pose_correction_lbs_offset_split_clone_merge_prune

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --port 6002 \
#    --wandb_disable

outdir=output/${exp_name}

#python render.py -m $outdir \
#     --motion_offset_flag \
#     --smpl_type smplx \
#     --actor_gender neutral \
#     --iteration ${iterations} \
#     --skip_train