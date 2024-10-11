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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from knn_cuda import KNN
import pickle
import torch.nn.functional as F
from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from nets.mlp_tuned_extra_joints import ExtraPoseTuner
from nets.mlp_delta_non_rigid import NonrigidDeformer
from nets.mlp_delta_extra_joints import ExtraJointsDeformer



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, smpl_type : str, motion_offset_flag : bool, non_rigid_flag : bool, \
                 non_rigid_use_extra_condition_flag : bool, joints_opt_flag : bool, actor_gender: str,  \
                 model_path=None, load_iteration=None, extra_joints_batch=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.num_extra_joints = 0
        self.extra_joints = None
        self.extra_parents = None
        self.extra_joints_batch = extra_joints_batch
        self.setup_functions()

        self.device=torch.device('cuda', torch.cuda.current_device())

        # load SMPL model
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('assets', f'SMPL_{actor_gender.upper()}.pkl')
            self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        elif smpl_type == 'smplx':
            neutral_smpl_path = os.path.join('assets/models/smplx', f'SMPLX_{actor_gender.upper()}.npz')
            params_init = dict(np.load(neutral_smpl_path, allow_pickle=True))
            self.SMPL_NEUTRAL = SMPL_to_tensor(params_init, device=self.device)
        elif smpl_type == 'simple_smplx':
            neutral_smpl_path = os.path.join('assets/models/smplx', f'SMPLX_{actor_gender.upper()}.npz')
            params_init = dict(np.load(neutral_smpl_path, allow_pickle=True))
            self.SMPL_NEUTRAL = SMPL_to_tensor(params_init, device=self.device)
            self.SMPL_NEUTRAL['weights'] = self.SMPL_NEUTRAL['weights'][:, :25]
            self.SMPL_NEUTRAL['J_regressor'] = self.SMPL_NEUTRAL['J_regressor'][:25, :]
            self.SMPL_NEUTRAL['kintree_table'] = self.SMPL_NEUTRAL['kintree_table'][:, :25]
            self.SMPL_NEUTRAL['posedirs'] = self.SMPL_NEUTRAL['posedirs'][:,:,:216]
            # print(self.SMPL_NEUTRAL['kintree_table'])
            # for key_ in self.SMPL_NEUTRAL.keys():
            #     print(key_, self.SMPL_NEUTRAL[key_].shape)
            # assert False

        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.knn_near_2 = KNN(k=2, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        self.non_rigid_flag = non_rigid_flag
        self.non_rigid_use_extra_condition_flag = non_rigid_use_extra_condition_flag
        self.joints_opt_flag = joints_opt_flag

        
        if self.motion_offset_flag:
            # load pose correction module
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
    
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2)
            self.pose_decoder.to(self.device)
            
            # load extra pose tuning module
            self.extrapose_tuner = ExtraPoseTuner(total_bones=total_bones, mlp_width=128, mlp_depth=2)
            self.extrapose_tuner.to(self.device)
            if load_iteration and model_path:
                model_path = os.path.join(model_path, "mlp_ckpt", "iteration_" + str(load_iteration), "ckpt.pth")
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=self.device)
                    self.num_extra_joints = ckpt['num_extra_joints']
                    self.extra_joints = ckpt['extra_joints']
                    self.extra_parents = ckpt['extra_parents']

            # load lbs weight module
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones+self.num_extra_joints)
            self.lweight_offset_decoder.to(self.device)
            
            self.non_rigid_deformer = None
            if self.non_rigid_flag:
                self.non_rigid_deformer = NonrigidDeformer(device=self.device)
                self.non_rigid_deformer.to(self.device)
              
            self.joints_deformer = None
            if self.joints_opt_flag:
                self.joints_deformer = ExtraJointsDeformer(device=self.device)
                self.joints_deformer.to(self.device)
            
            # define the accumulated distance for each point
            self.total_bones = total_bones
            self.acc_dist = torch.empty(0)
            self.acc_sqre = torch.empty(0)
            self.std_dist = torch.empty(0)
            self.control_joint = torch.empty(0)
            self.acc_iter = 0

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.num_extra_joints,
            self.extra_joints,
            self.extra_parents,
            self.use_extrapose_tuner,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
            self.extrapose_tuner,
            self.non_rigid_deformer,
            self.joints_deformer,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        self.num_extra_joints,
        self.extra_joints,
        self.extra_parents,
        self.use_extrapose_tuner,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.pose_decoder,
        self.lweight_offset_decoder,
        self.extrapose_tuner,
        self.non_rigid_deformer,
        self.joints_deformer) = model_args
        self.training_setup(training_args, self.use_extrapose_tuner)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.update_std_dist()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_dis_sqre(self):
        return self.acc_dist, self.acc_sqre, self.acc_iter
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, transform=None, d_rotation=None, d_scaling=None):
        
        if d_rotation is not None:
            scaling = self.get_scaling + d_scaling
            rotation = self._rotation + d_rotation
        else:
            scaling = self.get_scaling
            rotation = self._rotation
        
        return self.covariance_activation(scaling, scaling_modifier, rotation, transform)
    
    def update_std_dist(self):
        print('update std dist...')
        num_pts = self._xyz.shape[0]
        self.acc_dist = torch.zeros((num_pts, self.total_bones+self.num_extra_joints), device="cuda")
        self.acc_sqre = torch.zeros((num_pts, self.total_bones+self.num_extra_joints), device="cuda")
        self.std_dist = torch.zeros((num_pts, self.total_bones+self.num_extra_joints), device="cuda")
        self.control_joint = torch.zeros((num_pts, 1), device="cuda")
        self.acc_iter = 0
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() #[10475, 3]
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self.acc_dist = torch.zeros((fused_point_cloud.shape[0], self.total_bones+self.num_extra_joints), device="cuda")
        self.acc_sqre = torch.zeros((fused_point_cloud.shape[0], self.total_bones+self.num_extra_joints), device="cuda")
        self.std_dist = torch.zeros((fused_point_cloud.shape[0], self.total_bones+self.num_extra_joints), device="cuda")
        self.control_joint = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, use_extrapose_tuner):
        self.use_extrapose_tuner = use_extrapose_tuner
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if not self.motion_offset_flag:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
            ]
        if self.use_extrapose_tuner:
            l += [{'params': self.extrapose_tuner.parameters(), 'lr': training_args.extrapose_tuner_lr,
                 "name": "extrapose_tuner"},]
        if self.non_rigid_flag:
            l += [{'params': self.non_rigid_deformer.parameters(), 'lr': training_args.non_rigid_deformer_lr,
                 "name": "non_rigid_deformer"},]
        if self.joints_opt_flag:
            l += [{'params': self.joints_deformer.parameters(), 'lr': training_args.joints_deformer_lr,
                 "name": "joints_deformer"},]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        print('reset opacity...')
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if len(group["params"]) == 1:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def kl_densify_and_clone(self, grads, grad_threshold, scene_extent, kl_threshold=0.4):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())     
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()
        
        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)
        
        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl clone]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask])
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def kl_densify_and_split(self, grads, grad_threshold, scene_extent, kl_threshold=0.4, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())     
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl split]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def kl_merge(self, grads, grad_threshold, scene_extent, kl_threshold=0.1):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())     
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div < kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl merge]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        if selected_pts_mask.sum() >= 1:

            selected_point_ids = point_ids[0][selected_pts_mask]
            new_xyz = self.get_xyz[selected_point_ids].mean(1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_point_ids][:,0] / 0.8)
            new_rotation = self._rotation[selected_point_ids][:,0]
            new_features_dc = self._features_dc[selected_point_ids].mean(1)
            new_features_rest = self._features_rest[selected_point_ids].mean(1)
            new_opacity = self._opacity[selected_point_ids].mean(1)

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

            selected_pts_mask[selected_point_ids[:,1]] = True
            # prune_filter = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, kl_threshold=0.4, t_vertices=None, iter=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)
        self.kl_densify_and_clone(grads, max_grad, extent, kl_threshold)
        self.kl_densify_and_split(grads, max_grad, extent, kl_threshold)
        self.kl_merge(grads, max_grad, extent, 0.1)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # use smpl prior to prune points 
        # distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
        # distance = distance.view(distance.shape[0], -1)
        # threshold = 0.05
        # pts_mask = (distance > threshold).squeeze()
        # we not use pts_mask
        # prune_mask = prune_mask | pts_mask

        print('total points num: ', self._xyz.shape[0], 'prune num: ', prune_mask.sum().item())
        
        self.prune_points(prune_mask)
        
        self.update_std_dist()

        torch.cuda.empty_cache()

    def kl_div(self, mu_0, rotation_0_q, scaling_0_diag, mu_1, rotation_1_q, scaling_1_diag):

        # claculate cov_0
        rotation_0 = build_rotation(rotation_0_q)
        scaling_0 = build_scaling(scaling_0_diag)
        L_0 = rotation_0 @ scaling_0
        cov_0 = L_0 @ L_0.transpose(1, 2)

        # claculate inverse of cov_1
        rotation_1 = build_rotation(rotation_1_q)
        scaling_1_inv = build_scaling(1/scaling_1_diag)
        L_1_inv = rotation_1 @ scaling_1_inv
        cov_1_inv = L_1_inv @ L_1_inv.transpose(1, 2)

        # difference of mu_1 and mu_0
        mu_diff = mu_1 - mu_0

        # calculate kl divergence
        # kl_div_0 = torch.vmap(torch.trace)(cov_1_inv @ cov_0)
        # for older pytorch there is no .vmap
        # cov_1_inv and cov_0 has shape [10475, 3, 3]
        ###
        bmm_result = torch.einsum('...ij,...jk->...ik', cov_1_inv, cov_0)
        trace_result = bmm_result.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
        kl_div_0 = trace_result
        ###
        kl_div_1 = mu_diff[:,None].matmul(cov_1_inv).matmul(mu_diff[..., None]).squeeze()
        kl_div_2 = torch.log(torch.prod((scaling_1_diag/scaling_0_diag)**2, dim=1))
        kl_div = 0.5 * (kl_div_0 + kl_div_1 + kl_div_2 - 3)
        return kl_div

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def update_lbs_network(self):
    
        original_input_dim = self.lweight_offset_decoder.bw_fc.in_channels
        original_output_dim = self.lweight_offset_decoder.bw_fc.out_channels
        new_output_dim = original_output_dim + 1
        
        with torch.no_grad():
            original_weight = self.lweight_offset_decoder.bw_fc.weight.data
            original_bias = self.lweight_offset_decoder.bw_fc.bias.data
    
            # 创建新的权重和偏置矩阵
            new_weight = torch.zeros((new_output_dim, original_weight.size(1), original_weight.size(2)), device=self.device)
            new_bias = torch.zeros(new_output_dim, device=self.device)
    
            # 将原始权重和偏置复制到新的矩阵中
            new_weight[:original_weight.size(0), :, :] = original_weight
            new_bias[:original_bias.size(0)] = original_bias
    
            # 创建新的 Conv1d 层并加载新的权重和偏置
            self.lweight_offset_decoder.bw_fc = nn.Conv1d(original_input_dim, new_output_dim, 1)
            self.lweight_offset_decoder.bw_fc.weight.data = new_weight
            self.lweight_offset_decoder.bw_fc.bias.data = new_bias

            nn.init.kaiming_uniform_(self.lweight_offset_decoder.bw_fc.weight[original_weight.size(0):], a=math.sqrt(5))

            # Kaiming Uniform 初始化 bias (如果需要)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lweight_offset_decoder.bw_fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.lweight_offset_decoder.bw_fc.bias[original_bias.size(0):], -bound, bound)

            self.lweight_offset_decoder.to(self.device)
        
    def update_pose_decoder(self):
    
        # Get original dimensions
        original_input_dim = self.pose_decoder.block_mlps[0].in_features
        original_output_dim = self.pose_decoder.block_mlps[-1].out_features

        # New dimensions
        new_input_dim = original_input_dim + 3
        new_output_dim = original_output_dim + 3
        
        # Create new input layer
        new_input_layer = nn.Linear(new_input_dim, self.pose_decoder.block_mlps[0].out_features).to(self.device)
        with torch.no_grad():
            new_input_layer.weight[:, :original_input_dim] = self.pose_decoder.block_mlps[0].weight
            new_input_layer.weight[:, original_input_dim:] = nn.init.xavier_uniform_(
                torch.empty(new_input_layer.weight.size(0), 3, device=self.device))
            new_input_layer.bias = self.pose_decoder.block_mlps[0].bias

        # Create new output layer
        new_output_layer = nn.Linear(self.pose_decoder.block_mlps[-1].in_features, new_output_dim).to(self.device)
        with torch.no_grad():
            init_val = 1e-5
            new_output_layer.weight[:original_output_dim, :] = self.pose_decoder.block_mlps[-1].weight
            new_output_layer.weight[original_output_dim:, :] = torch.empty(3,
                                        new_output_layer.weight.size(1),
                                        device=self.device).uniform_(-init_val, init_val)
            new_output_layer.bias[:original_output_dim] = self.pose_decoder.block_mlps[-1].bias
            new_output_layer.bias[original_output_dim:] = nn.init.zeros_(
                torch.empty(3, device=self.device))

        # Replace layers in the model
        self.pose_decoder.block_mlps = nn.Sequential(
            new_input_layer,
            *self.pose_decoder.block_mlps[1:-1],
            new_output_layer
        ).to(self.device)

    def create_new_joint(self, joint_index, new_joint_init):
    
        num_main_joints = self.SMPL_NEUTRAL['weights'].shape[-1]
        num_extra_joints =  self.num_extra_joints
        print('before', self.lweight_offset_decoder.bw_fc.out_channels)
        self.update_lbs_network()
        print('update', self.lweight_offset_decoder.bw_fc.out_channels)
        print('before', self.pose_decoder.block_mlps[0].in_features, self.pose_decoder.block_mlps[-1].out_features)
        self.update_pose_decoder()
        print('update', self.pose_decoder.block_mlps[0].in_features, self.pose_decoder.block_mlps[-1].out_features)
        print(num_main_joints, num_extra_joints)
        if self.num_extra_joints == 0:
            self.extra_joints = new_joint_init.unsqueeze(0).to(self.device)
            self.extra_parents = torch.tensor([joint_index], device=self.device)
            self.num_extra_joints+=1
        else:
            self.extra_joints = torch.cat((self.extra_joints, \
                                           new_joint_init.unsqueeze(0).to(self.device)), dim=1)
            self.extra_parents = torch.cat((self.extra_parents, \
                                           torch.tensor([joint_index], device=self.device)))
            self.num_extra_joints+=1
        self.update_std_dist()
        
    def create_new_joint_v2(self, joint_index, new_joint_init):
    
        num_main_joints = self.SMPL_NEUTRAL['weights'].shape[-1]
        num_extra_joints = self.num_extra_joints
        print('num main joints:', num_main_joints, ', num extra joints:', num_extra_joints)
        print('before', self.lweight_offset_decoder.bw_fc.out_channels)
        self.update_lbs_network()
        print('update', self.lweight_offset_decoder.bw_fc.out_channels)
        if self.num_extra_joints == 0:
            self.extra_joints = new_joint_init.unsqueeze(0).to(self.device)
            self.extra_parents = torch.tensor([joint_index], device=self.device)
            self.num_extra_joints+=1
        else:
            self.extra_joints = torch.cat((self.extra_joints, \
                                           new_joint_init.unsqueeze(0).to(self.device)), dim=0)
            self.extra_parents = torch.cat((self.extra_parents, \
                                           torch.tensor([joint_index], device=self.device)))
            self.num_extra_joints+=1
        print(self.extra_joints.shape, self.extra_parents.shape, self.num_extra_joints)
        # if self.num_extra_joints == 0:
        #     self.extra_joints = nn.Parameter(new_joint_init.unsqueeze(0).\
        #                                      to(self.device).requires_grad_(True))
        #     self.extra_parents = torch.tensor([joint_index], device=self.device)
        #     self.num_extra_joints += 1
        #
        #     self.optimizer.add_param_group({'params': [self.extra_joints], \
        #                                     'lr': 0.01, \
        #                                     "name": "extra_joints"})
        # else:
        #     extra_joints = torch.cat((self.extra_joints, \
        #                               new_joint_init.unsqueeze(0).to(self.device)), dim=1)
        #     self.extra_joints = nn.Parameter(extra_joints.requires_grad_(True))
        #     self.extra_parents = torch.cat((self.extra_parents, \
        #                                 torch.tensor([joint_index], device=self.device)))
        #     self.num_extra_joints += 1

        self.update_std_dist()

    
    
    def coarse_deform_c2source(self, query_pts, params, t_params, t_vertices, \
                               lbs_weights=None, correct_Rs=None, return_transl=False, \
                               pc_grad=None):
        
        bs = query_pts.shape[0]   # 1
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]   #55
        vertices_num = t_vertices.shape[1]     #10475
        # Find nearest smpl vertex
        smpl_pts = t_vertices        # [1, 10475, 3]
        
        #this is not right cause some points can not be simply judged by KNN for belonging
        distance, vert_ids = self.knn(smpl_pts.float(), query_pts.float())  #[1, N, 1]
        # print(smpl_pts.float().shape, query_pts.float().shape)
        # assert False
        # print(torch.mean(distance))
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            # also considering optimizing the weight
            
            pretrained_bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num) # [1, N, 55]
            
            if self.num_extra_joints !=0:
                # extra_bweights = torch.zeros((*vert_ids.shape[:2], self.num_extra_joints), device=self.device)
                # extra_bweights = torch.ones((*vert_ids.shape[:2], self.num_extra_joints), device=self.device)
                extra_bweights = pretrained_bweights[:, :, self.extra_parents]
                # print(pretrained_bweights.shape)
                # print(self.extra_parents, self.extra_parents.shape)
                # print(torch.max(extra_bweights))
                # assert False
                
                pretrained_bweights = torch.cat((pretrained_bweights, extra_bweights), dim=-1)
            bweights = torch.log(pretrained_bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)
            # if self.num_extra_joints !=0:
            #     max_values, _ = torch.max(bweights, dim=1)
            #     print('max_bweight', max_values)
          
        extra_joints = self.extra_joints
        if self.num_extra_joints !=0:
            if self.joints_opt_flag:
                p = torch.arange(1, self.num_extra_joints + 1, device='cuda').\
                        view(self.num_extra_joints, 1).float()
                joints_opt_out = self.joints_deformer(p)
                d_joints = joints_opt_out['d_joints']
                # print(p.shape, extra_joints.shape, d_joints.shape)
                extra_joints = extra_joints + d_joints
            extra_joints = extra_joints.unsqueeze(0)

        ### From Big To T Pose
        big_pose_params = t_params
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params, \
                                    extra_joints=extra_joints, extra_parents=self.extra_parents,\
                                    num_extra_joints=self.num_extra_joints)

        # A [1, 55, 4, 4]  R [3, 3]   Th [1, 3]  joints [1, 55, 3]
        # infact A is the transformation from T-pose to big-pose
        A = torch.matmul(bweights, A.reshape(bs, joints_num+self.num_extra_joints, -1))   # 加权后的转移矩阵
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from Big To T Pose
        transforms = R_inv

        # translation from Big To T Pose
        translation = None
        if return_transl: 
            translation = -A[..., :3, 3]
            translation = torch.matmul(R_inv, translation[..., None]).squeeze(-1)
        
        self.mean_shape = True
        if self.mean_shape:

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()

            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offsets

            if return_transl: 
                translation -= pose_offsets

            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]#.cuda()
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            if return_transl: 
                translation += shape_offset

            posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])

            if self.num_extra_joints > 0:
                extra_rot_mats = torch.eye(3).unsqueeze(0).repeat(1, self.num_extra_joints, 1, 1).to(rot_mats.device)
                rot_mats = torch.cat((rot_mats, extra_rot_mats), dim=1)
            
            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), \
                                                correct_Rs.reshape(-1, 3, 3)).reshape(-1, joints_num+self.num_extra_joints-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)
            
            pose_feature = (rot_mats[:, 1:joints_num, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offsets

            if return_transl: 
                translation += pose_offsets

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, \
                                       extra_joints=extra_joints, extra_parents=self.extra_parents, \
                                       rot_mats=rot_mats)
        self.s_A = A
        bweights_joints = torch.eye(joints_num+self.num_extra_joints).unsqueeze(0).cuda()
        
        A_j = torch.matmul(bweights_joints, self.s_A.reshape(bs, joints_num+self.num_extra_joints, -1))
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num+self.num_extra_joints, -1))
        A_j = torch.reshape(A_j, (bs, -1, 4, 4))
        A = torch.reshape(A, (bs, -1, 4, 4))

        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]

        can_joints = torch.matmul(A_j[..., :3, :3], joints[..., None]).squeeze(-1)
        smpl_src_joints = can_joints + A_j[..., :3, 3]

        transforms = torch.matmul(A[..., :3, :3], transforms)

        if return_transl: 
            translation = torch.matmul(A[..., :3, :3], translation[..., None]).squeeze(-1) + A[..., :3, 3]

        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th
        world_src_joints = torch.matmul(smpl_src_joints, R_inv) + Th
        world_src_query = torch.matmul(query_pts, R_inv) + Th
        transforms = torch.matmul(R, transforms)

        if return_transl: 
            translation = torch.matmul(translation, R_inv).squeeze(-1) + Th
        
        return world_src_query, world_src_pts, world_src_joints, bweights, pretrained_bweights,\
               transforms, translation, distance

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    #rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix from child p' to father p
    # p = transforms_mat x p'
    # the rel_joints keep the same for all poses because joint position won't move in father's coordinate
    # T1/2 T0/1
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)
    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    
    transforms = torch.stack(transform_chain, dim=1)
    
    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    # define the difference of joint between target pose and t pose, no need to change rot as t pose
    # rot is identity
    # transforms[..., 3] = transforms[..., 3] - rel_joints
    
    # transforms_final = transforms.clone()
    t_rel = transforms[..., 3] - rel_joints
    # transforms_final = torch.cat([transforms_final[..., :3], t_rel.unsqueeze(-1)], dim=-1)
    transforms = torch.cat([transforms[..., :3], t_rel.unsqueeze(-1)], dim=-1)

    return transforms

# @profile
def get_transform_params_torch(smpl, params, extra_joints=None, extra_parents=None, \
                               num_extra_joints=0, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']   # [10475, 3, 400]
    betas = params['shapes']    # [1, 20]
    
    # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
    v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()
    # num_extra_joints += 1
    # extra_joints = torch.ones(3).unsqueeze(0).repeat(1, num_extra_joints, 1).to(betas.device)
    # extra_parents = torch.tensor([21], device=betas.device)

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 24 x 3 x 3
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)
        
        if num_extra_joints > 0:
            extra_rot_mats = torch.eye(3).unsqueeze(0).repeat(1, num_extra_joints, 1, 1).to(rot_mats.device)
            rot_mats = torch.cat((rot_mats, extra_rot_mats), dim=1)
        
        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)
    
      
    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]
    
    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    
    if extra_joints is not None:
        joints = torch.cat((joints, extra_joints), dim=1)
        parents = torch.cat((parents, extra_parents))

    A = get_rigid_transformation_torch(rot_mats, joints, parents) # [1, 55, 4, 4]
    # apply global transformation
    R = params['R']  # [3, 3]
    Th = params['Th']   # [1, 3]

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat