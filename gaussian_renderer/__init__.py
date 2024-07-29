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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import gc

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, \
            scaling_modifier = 1.0, pc_grad = None, override_color = None, \
            return_smpl_rot=False, transforms=None, translation=None, test=False, vis_extrapose=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means3D_original = means3D
    
    if not pc.motion_offset_flag:
        _, means3D, joints3D, _, transforms, _, distance = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset 纠正错误的 pose
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:] #[1, 162]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']  # [1, 54, 3, 3]

            extra_vrec = None
            if pc.num_extra_joints != 0:
                t = viewpoint_camera.pose_id
                t = torch.tensor(t, device=dst_posevec.device).view(1, 1).\
                        expand(pc.num_extra_joints, 1).float() / 100
                p = torch.arange(1, pc.num_extra_joints + 1, device='cuda').\
                        view(pc.num_extra_joints, 1).float()
                extra_pose_out = pc.extrapose_tuner(t, p)
                extra_correct_Rs = extra_pose_out['Rs_extra'].unsqueeze(0)
                correct_Rs = torch.cat([correct_Rs, extra_correct_Rs], dim=1)
                
                if vis_extrapose:
                    extra_vrec = extra_pose_out['rvec_extra']
            
            # if pc.num_extra_joints != 0:
            #     extra_posevec = torch.zeros((1, 3*pc.num_extra_joints), device=dst_posevec.device)
            #     dst_posevec = torch.cat((dst_posevec, extra_posevec), dim=-1)
            #     pose_out = pc.pose_decoder(dst_posevec, t, pc.num_extra_joints)
            # else:
            #     pose_out = pc.pose_decoder(dst_posevec, t)
            
            
            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)   # [1, 10475, 55]

            # transform points
            smpl_3D, means3D, joints3D, bweights, transforms, translation, distance = \
                pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs,
                return_transl=return_smpl_rot, pc_grad=pc_grad)

        else:
            correct_Rs = None
            joints3D = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation

    smpl_3D = smpl_3D.squeeze()
    means3D = means3D.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity
    joints3D = joints3D.squeeze()
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # rendered_cano, radii, depth, alpha = rasterizer(
    #     means3D = smpl_3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    
    if not test:
        dist = torch.sqrt(torch.sum((means3D.unsqueeze(1) - joints3D.unsqueeze(0)) ** 2, dim=2))  # [N, 55]
        
        # update accumulation
        acc_dist, acc_sqre, acc_iter = pc.get_dis_sqre
        acc_dist += dist
        acc_sqre += dist ** 2
        acc_iter += 1
        pc.acc_dist = acc_dist
        pc.acc_sqre = acc_sqre
        pc.acc_iter = acc_iter
        # compute std
        mean_dist = acc_dist / acc_iter
        var_dist = (acc_sqre / acc_iter) - (mean_dist ** 2) + 1e-5
        pc.std_dist = torch.sqrt(var_dist)  # [N, 55]
        std_weights = 1. / (pc.std_dist + 1e-6)
        # print('var_dist', torch.max(var_dist), torch.min(var_dist))
        # print('pcdist', torch.max(pc.std_dist), torch.min(pc.std_dist))
        # print(torch.max(std_weights), torch.min(std_weights))
        std_weights = std_weights / std_weights.sum(dim=1, keepdim=True)
        pc.control_joint = torch.argmax(std_weights, dim=1)
        std_weights = torch.zeros_like(std_weights).scatter_(1, std_weights.argmax(dim=1, keepdim=True), 1)
    
    
    maxgrad_index = None
    distance_mask = None
    new_joint_init = None
    
    if pc_grad is not None:
        # pc_grad [] bweights [1, N, 55]
        pc_grad_flat = pc_grad.unsqueeze(0)
        grad_bweights = bweights.permute(0, 2, 1)
        
        ###
        pc_d_grad = pc_grad_flat*distance
        hb_lamda = 0.6
        std_weights = std_weights.unsqueeze(0).permute(0, 2, 1)
        hybrid_weights = hb_lamda*grad_bweights+(1-hb_lamda)*std_weights

        # thresh = 0.2
        # distance_mask = distance > thresh
        # distance_mask = distance_mask.squeeze(0).squeeze(-1)
        # masked_pc_grad = pc_grad_flat[:, distance_mask, :]
        #
        # masked_grad_bweights = grad_bweights[:, :, distance_mask]
        # masked_std_weights = std_weights.unsqueeze(0).permute(0, 2, 1)[:, :, distance_mask]
        # #print(masked_grad_bweights.shape, masked_std_weights.shape)
        # hb_lamda = 0.6
        # hybrid_weights = hb_lamda*masked_grad_bweights+(1-hb_lamda)*masked_std_weights
        
        pc_ones = torch.ones_like(pc_d_grad)
        perjoint_grad = torch.matmul(hybrid_weights, pc_d_grad).squeeze()
        perjoint_ones = torch.matmul(hybrid_weights, pc_ones).squeeze()
        perjoint_grad_avg = perjoint_grad / perjoint_ones
        
        maxgrad_index = torch.argmax(perjoint_grad_avg)

        with torch.no_grad():
            # find weighted point for joint maxgrad_index
            distance_mask = distance > 0.2
            distance_mask = distance_mask.squeeze()
            weights_for_maxgrad = hybrid_weights[:, maxgrad_index, :].squeeze()  # [1, 10475]
            # remember I'm testing not to use distance mask
            # weights_for_maxgrad = weights_for_maxgrad*distance_mask.float()
            normalized_weights_for_maxgrad = weights_for_maxgrad / weights_for_maxgrad.sum()
            # also notice this part should use means3D_original
            weighted_positions = means3D_original * normalized_weights_for_maxgrad.unsqueeze(-1)  # [1, 10475, 3]
            new_joint_init = weighted_positions.sum(dim=0).unsqueeze(0)  # [1, 3]
    
    grad_image = None
    joint_image = None
    outer_image = None
    control_image = None
    bweight_image = None
    joint_index = 21
    if maxgrad_index:
        joint_index = maxgrad_index
    
    if not test:
        color_array = torch.zeros(means3D.shape[0], 3, device=opacity.device).float()
        white_indices = torch.where(pc.control_joint == joint_index)[0]
        color_array[white_indices] = torch.tensor([1.0, 1.0, 1.0], device=opacity.device)
        
        with torch.no_grad():
            control_image, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=color_array,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
    
        bweight_control_joint = torch.argmax(bweights.squeeze(), dim=1)
        color_array = torch.zeros(means3D.shape[0], 3, device=opacity.device).float()
        white_indices = torch.where(bweight_control_joint == joint_index)[0]
        color_array[white_indices] = torch.tensor([1.0, 1.0, 1.0], device=opacity.device)
        with torch.no_grad():
            bweight_image, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=color_array,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
          
        if pc_grad is not None:
    
            pc_grad = torch.norm(pc_grad[...,:2], dim=-1, keepdim=True)
            with torch.no_grad():
                grad_image, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=None,
                    colors_precomp=pc_grad*torch.ones(pc_grad.shape[0], 3, device=opacity.device),
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            grad_image = 1e2*grad_image[:1]
            # print('find grad max index:', maxgrad_index)
            
            # joints3D = joints3D[joint_index:joint_index+1]
            ## delete
            # joints3D = torch.cat((joints3D, new_joint_init), dim=0)
            joints3D = torch.cat((joints3D[joint_index:joint_index+1], \
                                  joints3D[-1:]), dim=0)
            joint_num = joints3D.shape[0]
            scales_j = 0.02*torch.ones(joint_num, 3, device=opacity.device)
            base_rot = torch.tensor([1, 0, 0, 0], device=opacity.device)
            rotations_j = base_rot.repeat(joint_num, 1)
            opacity_j = torch.ones(joint_num, 1, device=opacity.device)
            joints2D = torch.zeros_like(joints3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            jointcolor = torch.ones(joint_num, 3, device=opacity.device).float()
            jointcolor[0, 1:] = 0.0
            jointcolor[1, :2] = 0.0
            
            with torch.no_grad():
                joint_image, _, _, _ = rasterizer(
                    means3D=joints3D,
                    means2D=joints2D,
                    shs=None,
                    colors_precomp=jointcolor,
                    opacities=opacity_j.float(),
                    scales=scales_j.float(),
                    rotations=rotations_j.float(),
                    cov3D_precomp=None)
    
            color_array = torch.zeros(means3D.shape[0], 3, device=opacity.device).float()
            color_array[distance_mask] = torch.tensor([1.0, 1.0, 1.0], device=opacity.device)
            with torch.no_grad():
                outer_image, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=None,
                    colors_precomp=color_array,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {"render": rendered_image,
            "render_depth": depth,
            "render_alpha": alpha,
            "render_grad": grad_image,              # gradient of each point
            "render_joints": joint_image,           # chosen joint to be adapted
            "render_control": control_image,        # control area of adapted joint by distance var
            "render_bweight": bweight_image,        # control area of adapted joint by bweight
            "render_outer": outer_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "maxgrad_index": maxgrad_index,
            "new_joint_init": new_joint_init,
            "extra_vrec": extra_vrec}
