import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class HeightCompression_ez_attention(nn.Module):
    def __init__(self, model_cfg, **kwargs):

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.output_channel = 256
        self.q_conv = nn.Conv2d(256,self.output_channel,1,1)
        self.k_conv = nn.Conv2d(22528,self.output_channel,1,1)
        self.v_conv = nn.Conv2d(22528,self.output_channel,1,1)
        self.attention = DotProductAttention()

    def forward(self, batch_dict):
        # batch_size，128，2，200，176
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features_bev = spatial_features.view(N, C * D, H ,W)
        spatial_features_rv =  spatial_features.view(N, C * W, H ,D)
        q = self.q_conv(spatial_features_bev).view(N,H*W,self.output_channel)
        k = self.k_conv(spatial_features_rv).view(N,H*D,self.output_channel)
        v = self.v_conv(spatial_features_rv).view(N,H*D,self.output_channel)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        spatial_features = self.attention(q,k,v).permute(0,2,1).view(N, C * D, H, W) # attention_out(N,)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.5, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = self.dropout(attn) @ v
        return output