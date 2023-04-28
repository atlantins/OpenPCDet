#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xvhuangjian
# time:2022/10/16

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

'''
batch_dict:
points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
frame_id:(batch_size,) -->帧ID-->我们存放的是npy的绝对地址，batch_size个地址
gt_boxes:(batch_size,N,8)--> (x,y,z,dx,dy,dz,ry,class)，
use_lead_xyz:(batch_size,) --> (1,1,1,1)，batch_size个1
voxels:(M,32,4) --> (x,y,z,r)
voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
voxel_num_points:(M,):每个voxel内的点云
batch_size：batch_size大小
'''

# 31530个体素， pillar里32个点，一个点10个维度

class PFNlayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm = True, last_layer = False):
        """
        in_channels: 10
        out_channels: 64
        """

        super(PFNlayer, self).__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm

        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            # 根据论文中，这是是简化版pointnet网络层的初始化
            # 论文中使用的是 1x1 的卷积层完成这里的升维操作（理论上使用卷积的计算速度会更快）
            # 输入的通道数是刚刚经过数据增强过后的点云特征，每个点云有10个特征，
            # 输出的通道数是64
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        """
        inputs:（31530，32，10)
        """
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_parts*self.part:(num_parts+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0) # 在第一个维度上堆叠
        else:
            x = self.linear(inputs) # (31530，32，64)
        torch.backends.cudnn.enable = False

        # BatchNorm1d层:(31530, 64, 32) --> (31530, 32, 64)
        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        x = F.relu(x)

        # 按照维度取每个voxel中的最大值 --> (31530, 1, 64)
        # 这里的0是表示取数值，max的1表示索引
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            # torch的repeat在第几维度复制几遍
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # 在最后一个维度上拼接
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super(PillarVFE, self).__init__(model_cfg = model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # 如果use_absolute_xyz==True，则num_point_features=4+6，否则为3

        num_point_features += 6 if self.use_absolute_xyz else 3

        # 如果使用距离特征，即使用sqrt(x^2+y^2+z^2)，则使用特征加1
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS  # 64
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)  # [10,64]

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(PFNlayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters)-2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_padding_indicator(self, actual_num, max_num, axis = 0):
        """
        计算padding的指示
        Args:
            actual_num:每个voxel实际点的数量（31530，）
            max_num:voxel最大点的数量（32，）
        Returns:
            paddings_indicator:表明需要padding的位置(31530, 32)
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1) # 扩展一个维度，变为（31530，1）
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
        max_num_shape[axis + 1] = -1 # [1, -1]
        max_num = torch.arange(max_num, dtype=torch.int, device = actual_num.device).view() # (1,32)
        paddings_indicator = actual_num.int() > max_num # (31530, 32)
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict:
            points:(97687,5)
            frame_id:(4,) --> (2238,2148,673,593)
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels:(31530,32,4) --> (x,y,z,intensity)
            voxel_coords:(31530,4) --> (batch_index,z,y,x) 在dataset.collate_batch中增加了batch索引
            voxel_num_points:(31530,)
            image_shape:(4,2) [[375 1242],[374 1238],[375 1242],[375 1242]]
            batch_size:4
        """
        voxel_features , voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        # 求每个voxle的平均值(31530, 1, 3)--> (31530, 1, 3) / (31530, 1, 1)
        # 被求和的维度，在求和后会变为1，如果没有keepdim=True的设置，python会默认压缩该维度，比如变为[31530, 3]
        # view扩充维度
        point_mean = voxel_features[:,:,:3].sum(dim=1,keepdim=True)/voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # (31530, 1, 3)
        f_cluster = voxel_features[:,:,:3] - point_mean  # (31530,32,3)

        f_center = torch.zeros_like(voxel_features[:,:,:3]) # (31530, 32, 3)
        #  (31530, 32) - (31530, 1)[(31530,)-->(31530, 1)]
        #  coords是网格点坐标，不是实际坐标，乘以voxel大小再加上偏移量是恢复网格中心点实际坐标

        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[:,:,:3], f_cluster, f_center]

        if self.with_distance:

            points_dist = torch.norm(vo)













