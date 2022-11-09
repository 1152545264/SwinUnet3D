import torch
import torch.nn as nn
import numpy as np
a = torch.ones(1, 4, 8, 8, 8)
b = torch.ones(1, 4, 8, 8, 8)
c = torch.ones(1, 4, 8, 8, 8)
# print(a.shape, b.transpose(-2, -1).shape)
# c = a * b
# print(c.shape)
# a = nn.ConvTranspose3d(in_channels=384, out_channels=384, kernel_size=2, stride=2)(a)
# print(a.shape)

# a = np.array([[1,2],[3,4],[5,6]])
# b = np.array([[5,6],[7,8],[1,2]])
# print(a)
# # print(a @ b.transpose(-2, -1))
# print(a*b)
# class Pooling(nn.Module):
#     """
#     Implementation of pooling for PoolFormer
#     --pool_size: pooling size
#     """
#     def __init__(self, pool_size=3):
#         super().__init__()
#         self.pool = nn.AvgPool3d(
#             pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
#
#     def forward(self, x):
#         print(x.shape)
#         return self.pool(x) - x

# a = torch.ones(1, 32, 32, 32, 96)
# a = Pooling()(a)

# class localattenblock(nn.Module):
#     def __init__(self, dim):
#         super(localattenblock, self).__init__()
#         self.conv3d_k = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
#         self.conv3d_q = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
#         self.conv3d_v = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.bn = nn.BatchNorm3d()
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         k = self.conv3d_k(x)
#         q = self.conv3d_q(x)
#         x = self.conv3d_v(x)
#         attn = k @ q
#         attn = self.softmax(attn)
#         print(attn.shape)
#         x = x * attn
#
#         return x
# a = a
# a = localattenblock(4)(a)
# print(a.shape)


# a = np.array([True, True, False, False], dtype=bool)
# a = np.where(a == True, False, a)
# # a = a == False
# # a[a==True]=False
# print(a)
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
# im = 224
# im = to_2tuple(im)
# print(im.type)
a = torch.ones(1, 24, 32, 32, 32)
conv = nn.ConvTranspose3d(24, 12, kernel_size=7, stride=2, padding=3, output_padding=1)
a = conv(a)
print(a.shape)
from monai.networks.blocks.dynunet_block import UnetOutBlock