from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv

# sparse 用来res结构
# post_act_block用来升维

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    """
    后处理执行块，根据conv_type选择对应的卷积操作并和norm与激活函数封装为块
    """

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample  # 没定义，无法跳转？？？
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # 版本2是可以read的，但是进行修改原来的属性.features需要进行更改为
        # x.features = F.relu(x.features)->x = x.replace_feature(F.relu(x.features))
        out = replace_feature(out, self.bn1(out.features))  # wtf   .Features代表tensor
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 对grid_size进行反向排列，
        # POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] ：xmin,ymin,zmin,xmax,ymax,zmax
        # VOXEL_SIZE: [0.05, 0.05, 0.1]

        self.sparse_shape = grid_size[::-1] + [1, 0, 0] 
        # [41, 1600, 1408],在原始网格的高度方向上增加了一维,self.sparse_shape=[41, 1600, 1480],这个是由kitti决定的 

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),  # 这里bias是True？？
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        # voxel_features, voxel_coords  shape (Batch * 16000, 4)
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # 根据voxel坐标，并将每个voxel放置voxel_coor对应的位置，建立成稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            # (Batch * 16000, 4)
            features=voxel_features,
            # (Batch * 16000, 4) 其中4为 batch_idx, x, y, z
            indices=voxel_coords.int(),
            # [41,1600,1408] ZYX 每个voxel的长宽高为0.05，0.05，0.1 点云的范围为[0, -40, -3, 70.4, 40, 1]
            spatial_shape=self.sparse_shape,
            # 4
            batch_size=batch_size
        )
        """
               稀疏卷积的计算中，feature，channel，shape，index这几个内容都是分开存放的，
               在后面用out.dense才把这三个内容组合到一起了，变为密集型的张量
               spconv卷积的输入也是一样，输入和输出更像是一个  字典或者说元组
               注意卷积中pad与no_pad的区别
               
        （1）输入的特征是[voxels,3]通过索引indices[voxels，4],最后一个维度表示[b,w,h,l]的格子中。也就是其对应的坐标coor，作者在最开始就把坐标voxel化。
        （2）spatial_shape大小为[41,1280,1056]，空间的总的size，对应着空间所有voxels（包含着没有点和存在点的voxel）41 × 1280 × 1056 = 55418880 
                
         # 对voxel_features按照coors进行索引，coors在之前的处理中加入例如batch这个位置，变成了四维
        # 输出是一个【batch_size，channels, sparse_shape】的数据（2， 4， 40， 1600， 1408）
        # 就是让数据按照coors里的坐标进行了排列，成为了标准的体素空间
        x, point_misc = self.backbone(x, points_mean, is_test)
        # x是backbone的输出，体素维度缩小8倍后的64维特征，point_misc包括几部分（mean cls reg）是auxiliary的输出，即预测出来的Seg和Center
        
        
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图
        """


        # [batch_size, 4, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        x = self.conv_input(input_sp_tensor)  # （4, 4, 41, 1600, 1408）

        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        x_conv1 = self.conv1(x)
        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 32, [21, 800, 704]]
        x_conv2 = self.conv2(x_conv1)
        # [batch_size, 32, [21, 800, 704]] --> [batch_size, 64, [11, 400, 352]]
        x_conv3 = self.conv3(x_conv2)
        # [batch_size, 32, [21, 800, 704]] --> [batch_size, 64, [11, 400, 352]]
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # [batch_size, 64, [5, 200, 176]] --> [batch_size, 128, [2, 200, 176]]

        out = self.conv_out(x_conv4)

        # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords'] # (64000, 4), (64000, 4)
        batch_size = batch_dict['batch_size'] # 4
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            # (Batch * 16000, 4)
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
