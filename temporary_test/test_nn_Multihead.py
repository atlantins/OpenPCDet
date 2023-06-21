import torch
import torch.nn as nn
 
# 先决定参数
dims = 256 * 10 # 所有头总共需要的输入维度
heads = 10    # 单注意力头的总共个数
dropout_pro = 0.0 # 单注意力头
 
# 传入参数得到我们需要的多注意力头
layer = nn.MultiheadAttention(embed_dim = dims, num_heads = heads, dropout = dropout_pro)
layer(torch)