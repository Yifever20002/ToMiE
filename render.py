#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
import imageio
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, vis_extrapose):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    bkgd_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "bkgd_mask")
    attention_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "attention_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(bkgd_mask_path, exist_ok=True)
    makedirs(attention_mask_path, exist_ok=True)

    # Load data (deserialize)
    # with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
    #     smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    bkgd_masks = []
    attention_masks = []
    extra_vrecs = []
    alphas = []
    elapsed_time = 0

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        gt = view.original_image[0:3, :, :].cuda()
        
        bound_mask = view.bound_mask
        # not use bound mask
        bound_mask = bound_mask.fill_(1)
        # transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']
        
        # Start timer
        start_time = time.time()
        
        # do not give transform
        transforms, translation = None, None
        render_output = render(view, gaussians, pipeline, background, \
                        transforms=transforms, translation=translation, \
                        test=True, vis_extrapose=vis_extrapose, get_mask_attention=False)#, fixed_pose=views[20])
        rendering = render_output["render"]
        attention_mask = render_output["render_mask_attention"]
        alpha = render_output["render_alpha"]
        
        attention_masks.append(attention_mask)
        
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        alphas.append(alpha)
        rgbs_gt.append(gt)
        
        bkgd_mask = view.bkgd_mask
        # bkgd_mask = bkgd_mask.squeeze()
        # bkgd_mask = (bkgd_mask > 0.).int()
        # bkgd_mask = bkgd_mask.numpy().astype(np.uint8) * 255
        bkgd_masks.append(bkgd_mask)
      
        if vis_extrapose:
            extra_vrec = render_output["extra_vrec"]
            extra_vrecs.append(extra_vrec)

    if len(extra_vrecs) and extra_vrecs[0] is not None:
        import matplotlib.pyplot as plt
        
        hist_path = os.path.join(model_path, name, "ours_{}".format(iteration), "joint_hist")
        makedirs(hist_path, exist_ok=True)
        traj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "3D_traj")
        makedirs(traj_path, exist_ok=True)
        
        extra_vrecs = torch.stack(extra_vrecs)
        num_frames, num_joints, _ = extra_vrecs.shape

        for joint_index in range(num_joints):
            joint_data = extra_vrecs[:, joint_index, :]
    
            x = joint_data[:, 0].detach().cpu().numpy()
            y = joint_data[:, 1].detach().cpu().numpy()
            z = joint_data[:, 2].detach().cpu().numpy()

            plt.figure(figsize=(15, 5))
    
            plt.subplot(1, 3, 1)
            plt.hist(x, bins=30, color='r', alpha=0.7)
            plt.title(f'Joint {joint_index + 1} - X Component Histogram')
            plt.xlabel('X Values')
            plt.ylabel('Frequency')
    
            plt.subplot(1, 3, 2)
            plt.hist(y, bins=30, color='g', alpha=0.7)
            plt.title(f'Joint {joint_index + 1} - Y Component Histogram')
            plt.xlabel('Y Values')
            plt.ylabel('Frequency')
    
            plt.subplot(1, 3, 3)
            plt.hist(z, bins=30, color='b', alpha=0.7)
            plt.title(f'Joint {joint_index + 1} - Z Component Histogram')
            plt.xlabel('Z Values')
            plt.ylabel('Frequency')
    
            plt.tight_layout()

            output_path = os.path.join(hist_path, f'joint_{joint_index + 1}_histograms.png')
            plt.savefig(output_path)
            plt.close()

            # 归一化到单位球面
            norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            x_sphere = x / norm
            y_sphere = y / norm
            z_sphere = z / norm

            # 绘制3D散点图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x_sphere, y_sphere, z_sphere, c=norm, cmap='viridis', alpha=0.6)

            # 设置标签和标题
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Sphere Distribution of Joint {joint_index + 1}')
            fig.colorbar(sc, ax=ax, label='Magnitude')

            # 保存图表
            output_path = os.path.join(traj_path, f'joint_{joint_index + 1}_3d_trajectory.png')
            plt.savefig(output_path)
            plt.close()

        print(f'Histograms saved in directory: {hist_path}')
        
    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0
    gt_frames = []
    render_frames = []
    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        bkgd_mask = bkgd_masks[id]
        alpha = alphas[id]
        #attention_mask = attention_masks[id]
        
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        bkgd_mask = torch.clamp(bkgd_mask, 0.0, 1.0)
        #attention_mask = torch.clamp(attention_mask, 0.0, 1.0)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(bkgd_mask, os.path.join(bkgd_mask_path, '{0:05d}'.format(id) + ".png"))
        #torchvision.utils.save_image(attention_mask, os.path.join(attention_mask_path, '{0:05d}'.format(id) + ".png"))
        
        gt_frames.append(imageio.imread(os.path.join(gts_path, '{0:05d}'.format(id) + ".png")))
        render_frames.append(imageio.imread(os.path.join(render_path, '{0:05d}'.format(id) + ".png")))

        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    imageio.mimwrite(os.path.join(gts_path, 'video.mp4'), gt_frames, fps=15)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), render_frames, fps=15)
    
    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, \
                skip_test : bool, mono_test : bool, non_rigid_flag, non_rigid_use_extra_condition_flag, \
                joints_opt_flag, vis_extrapose : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, \
                                  non_rigid_flag, non_rigid_use_extra_condition_flag, joints_opt_flag, \
                                  dataset.actor_gender, model_path=dataset.model_path, load_iteration=iteration)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mono_test=mono_test)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, \
                scene.getTrainCameras(), gaussians, pipeline, background, vis_extrapose)

        if not skip_test:
            if mono_test:
                 render_set(dataset.model_path, "test_mono", scene.loaded_iter, \
                    scene.getTestCameras(), gaussians, pipeline, background, vis_extrapose)
            else:
                render_set(dataset.model_path, "test", scene.loaded_iter, \
                           scene.getTestCameras(), gaussians, pipeline, background, vis_extrapose)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--mono_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis_extrapose", action="store_true")
    parser.add_argument("--use_extrapose_tuner", type=str2bool, nargs='?', const=True, default=False,
                        help="Activate extrapose tuner (default: False)")
    parser.add_argument("--non_rigid_flag", type=str2bool, nargs='?', const=True, default=False,
                        help="Activate non-rigid MLP (default: False)")
    parser.add_argument("--non_rigid_use_extra_condition_flag", type=str2bool, nargs='?', const=True, default=False,
                        help="Activate non-rigid extra condition (default: False)")
    parser.add_argument("--joints_opt_flag", type=str2bool, nargs='?', const=True, default=False,
                        help="Activate joint optimization (default: False)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, \
                args.mono_test, args.non_rigid_flag, args.non_rigid_use_extra_condition_flag, \
                args.joints_opt_flag, args.vis_extrapose)