import torch

from .vfe_template import VFETemplate

 # voxel特征编码，计算每个voxel里的平均值，当成voxel的特征
 # (Batch*16000, 5, 4) -->  (Batch*16000, 4)

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        #（x, y, z, r)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
 
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']

        # 求voxel内所有点的值
        #  shape (Batch*16000, 5, 4) -> (Batch*16000, 4) keppdim就变成了(b*16000, 1, 4)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)

        # 正则化项， 保证每个voxel中最少有一个点，防止除0,     .view变成(b*n,1)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)

        # 求均值
        points_mean = points_mean / normalizer

        # 之前的操作使得，内容变成语义相邻，内存不相邻，现在使用后在内存上相邻
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

'''
voxel_features torch.Size([32000, 5, 4])
tensor([[[24.6709, -1.1747, -1.6103,  0.3200],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 8.5358, -6.4309, -2.1651,  0.3400],
         [ 8.5159, -6.4369, -2.1630,  0.3400],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]],

        [[17.0407,  4.8221, -1.6395,  0.3000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]],

        ...,

        [[10.7062, -3.4055, -0.9014,  0.2100],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]],

        [[18.7864, -2.5729, -0.9689,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]],

        [[10.4682,  7.6055,  0.3398,  0.5000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000]]], device='cuda:0')'''