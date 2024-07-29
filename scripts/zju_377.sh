export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0

id_name=CoreView_377
dataset=../dataset/ZJU_Mocap/${id_name}/
iterations=3000

exp_name=zju/${id_name}_100_pose_correction_lbs_offset_split_clone_merge_prune

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type smpl \
    --actor_gender neutral \
    --iterations ${iterations} \
    --port 6010

outdir=output/${exp_name}

python render.py -m $outdir \
     --motion_offset_flag \
     --smpl_type smpl \
     --actor_gender neutral \
     --iteration ${iterations} \
     --skip_train
