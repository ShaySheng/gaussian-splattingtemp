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

# 导入PyTorch库
import torch

# 导入NumPy库
import numpy as np

# 从utils.general_utils模块导入inverse_sigmoid, get_expon_lr_func, build_rotation函数
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

# 从PyTorch库中导入nn模块
from torch import nn

# 导入Python的os模块，用于处理文件和目录
import os

# 从utils.system_utils模块导入mkdir_p函数
from utils.system_utils import mkdir_p

# 导入PlyData和PlyElement类，用于处理PLY格式的3D数据
from plyfile import PlyData, PlyElement

# 从utils.sh_utils模块导入RGB2SH函数
from utils.sh_utils import RGB2SH

# 从simple_knn._C模块导入distCUDA2函数
from simple_knn._C import distCUDA2

# 从utils.graphics_utils模块导入BasicPointCloud类
from utils.graphics_utils import BasicPointCloud

# 从utils.general_utils模块导入strip_symmetric, build_scaling_rotation函数
from utils.general_utils import strip_symmetric, build_scaling_rotation

# 定义一个名为GaussianModel的类
class GaussianModel:

    # 定义一个设置函数的方法
    def setup_functions(self):
        # 定义一个内部函数，用于根据缩放和旋转构建协方差
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 设置缩放的激活函数为指数函数
        self.scaling_activation = torch.exp
        # 设置缩放的逆激活函数为对数函数
        self.scaling_inverse_activation = torch.log

        # 设置协方差的激活函数为上面定义的build_covariance_from_scaling_rotation函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 设置不透明度的激活函数为sigmoid函数
        self.opacity_activation = torch.sigmoid
        # 设置不透明度的逆激活函数为inverse_sigmoid函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 设置旋转的激活函数为归一化函数
        self.rotation_activation = torch.nn.functional.normalize


    # 类的初始化方法，接受一个参数sh_degree（球谐函数的最高度数）
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0  # 初始化当前激活的球谐函数度数为0
        self.max_sh_degree = sh_degree  # 设置最大球谐函数度数为传入的sh_degree
        # 初始化以下变量为零大小的空Tensor
        self._xyz = torch.empty(0)  # 用于存储坐标
        self._features_dc = torch.empty(0)  # 用于存储直流（DC）特征
        self._features_rest = torch.empty(0)  # 用于存储其他特征
        self._scaling = torch.empty(0)  # 用于存储缩放参数
        self._rotation = torch.empty(0)  # 用于存储旋转参数
        self._opacity = torch.empty(0)  # 用于存储不透明度
        self.max_radii2D = torch.empty(0)  # 用于存储二维最大半径
        self.xyz_gradient_accum = torch.empty(0)  # 用于累积梯度信息
        self.denom = torch.empty(0)  # 用于存储分母信息
        self.optimizer = None  # 初始化优化器为None
        self.percent_dense = 0  # 初始化模型的密度百分比为0
        self.spatial_lr_scale = 0  # 初始化空间学习率缩放为0
        self.setup_functions()  # 调用setup_functions方法来设置类中的函数

    # 定义一个capture方法，用于捕获当前模型状态并返回
    def capture(self):
        return (
            self.active_sh_degree,  # 当前激活的球谐函数度数
            self._xyz,  # 坐标
            self._features_dc,  # 直流特征
            self._features_rest,  # 其他特征
            self._scaling,  # 缩放参数
            self._rotation,  # 旋转参数
            self._opacity,  # 不透明度
            self.max_radii2D,  # 二维最大半径
            self.xyz_gradient_accum,  # 梯度累积
            self.denom,  # 分母
            self.optimizer.state_dict(),  # 优化器的状态字典
            self.spatial_lr_scale,  # 空间学习率缩放
        )
    
    # 定义一个restore方法，用于恢复模型的状态
    def restore(self, model_args, training_args):
        # 将传入的model_args解包到相应的属性中
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)  # 调用training_setup方法，传入training_args
        self.xyz_gradient_accum = xyz_gradient_accum  # 设置xyz的梯度累积
        self.denom = denom  # 设置分母
        self.optimizer.load_state_dict(opt_dict)  # 从opt_dict中恢复优化器的状态

    # 定义一个属性装饰器，当调用get_scaling时，返回通过激活函数处理的_scaling属性
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    # 定义一个属性装饰器，当调用get_rotation时，返回通过激活函数处理的_rotation属性
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    # 定义一个属性装饰器，当调用get_xyz时，直接返回_xyz属性
    @property
    def get_xyz(self):
        return self._xyz
    
    # 定义一个属性装饰器，当调用get_features时，返回直流特征和其他特征的组合
    @property
    def get_features(self):
        features_dc = self._features_dc  # 获取直流特征
        features_rest = self._features_rest  # 获取其他特征
        return torch.cat((features_dc, features_rest), dim=1)  # 将两种特征合并并返回
    
    # 定义一个属性装饰器，当调用get_opacity时，返回经过激活函数处理的不透明度
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)  # 使用不透明度激活函数处理_opacity属性并返回
    
    # 定义一个方法get_covariance，用于获取协方差
    def get_covariance(self, scaling_modifier = 1):
        # 使用协方差激活函数计算并返回协方差
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 定义一个方法oneupSHdegree，用于提升球谐函数的度数
    def oneupSHdegree(self):
        # 如果当前激活的球谐函数度数小于最大度数，则将其增加1
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 定义一个方法，用于从点云数据创建模型
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale  # 设置空间学习率缩放
        # 将点云数据转换为Tensor并移至CUDA设备
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 将点云的颜色数据转换为球谐函数并移至CUDA设备
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 初始化特征Tensor
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color  # 设置直流部分的特征
        features[:, 3:, 1:] = 0.0  # 其他特征部分初始化为0

        # 打印初始化时点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算点间距离的平方，并保证值不小于一个很小的正数
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 根据距离计算缩放比例
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 初始化旋转参数，旋转向量的第一个元素设置为1，其余为0
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化不透明度参数
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将各个参数转换为神经网络的参数，并设置为可优化
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 初始化用于存储二维最大半径的Tensor
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 定义一个方法用于设置训练参数
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense  # 设置模型密度的百分比
        # 初始化xyz梯度累积和分母为零的Tensor
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 创建一个列表，包含各个模型参数及其学习率和名称
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 使用Adam优化器并设置其参数
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 设置xyz参数的学习率调度器
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)



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
            assert len(group["params"]) == 1
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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1