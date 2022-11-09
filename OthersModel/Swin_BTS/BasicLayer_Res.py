from functools import reduce, lru_cache
from operator import mul

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from .BasicLayer import BasicLayer



"""
以下模块为TransBTSBottlenecktransformer部分
"""
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print(attn.type)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(x.shape)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 96, 512)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        # print(position_embeddings.shape)
        return x + position_embeddings


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)


class ChannelAttention(nn.Module):
    # 此模块为CBAM注意力CA
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class localattenblock(nn.Module):
    def __init__(self, dim):
        super(localattenblock, self).__init__()
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()


    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ELSA(nn.Module):
    """
    Implementation of enhanced local self-attention
    """
    def __init__(self, dim, num_heads, dim_qk=None, dim_v=None, kernel_size=7,
                 stride=1, dilation=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., group_width=8, groups=1, lam=1,
                 gamma=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_qk = dim_qk or self.dim // 3 * 2
        self.dim_v = dim_v or dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = kernel_size // 2 * dilation
        head_dim = self.dim_v // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.dim_qk % group_width != 0:
            self.dim_qk = math.ceil(float(self.dim_qk) / group_width) * group_width

        self.group_width = group_width
        self.groups = groups
        # self.lam = lam
        # self.gamma = gamma

        self.pre_proj = nn.Conv3d(dim, self.dim_qk * 2 + self.dim_v, 1, bias=qkv_bias)
        self.attn = nn.Sequential(
            nn.Conv3d(self.dim_qk, self.dim_qk, kernel_size, padding=(kernel_size // 2)*dilation,
                      dilation=dilation, groups=self.dim_qk // group_width),
            nn.GELU(),
            # nn.Conv3d(self.dim_qk, kernel_size ** 2 * num_heads, 1, groups=groups),
            nn.Conv3d(self.dim_qk, self.dim_v, 1, groups=groups)
        )


        self.attn_drop = nn.Dropout(attn_drop)
        self.post_proj = nn.Linear(self.dim_v, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, D, H, W, C = x.shape
        # C = self.dim_v
        # ks = self.kernel_size
        # G = self.num_heads
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W

        qkv = self.pre_proj(x)
        # print(qkv.shape)
        q, k, v = torch.split(qkv, (self.dim_qk, self.dim_qk, self.dim_v), dim=1)
        # print(q.shape, k.shape, v.shape)
        hadamard_product = q * k * self.scale
        # print(hadamard_product.shape)
        if self.stride > 1:
            hadamard_product = F.avg_pool3d(hadamard_product, self.stride)

        h_attn = self.attn(hadamard_product)
        # print(h_attn.shape)
        v = v.reshape(B * self.num_heads, C // self.num_heads, D, H, W)
        # print()
        Bv, Cv = v.shape[:2]
        h_attn = h_attn.reshape(B * self.num_heads, -1, D, H, W).softmax(1)
        h_attn = self.attn_drop(h_attn)
        # x = F.unfold(v, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.pad, stride=self.stride)\
        #     .reshape(Bv, Cv, self.kernel_size ** 3, D * H * W)

        # h_attn = h_attn.reshape(Bv, 1, self.kernel_size ** 3, D * H * W)
        # x = (x * h_attn).sum(2).reshape(Bv, Cv, D, H, W)
        x = (v * h_attn).reshape(Bv, Cv, D, H, W)
        # print(x.shape)
        x = x.reshape(B, C, D, H, W)
        x = self.post_proj(x.permute(0, 2, 3, 4, 1))  # B, D, H, W, C
        x = self.proj_drop(x)
        return x


class ELSABlock(nn.Module):
    """
    Implementation of ELSA block: including ELSA + MLP
    """
    def __init__(self, dim, kernel_size,
                 stride=1, num_heads=1, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=1, dim_qk=None, dim_v=None,
                 lam=1, gamma=1, dilation=1, group_width=8, groups=1,
                 **kwargs):
        super().__init__()
        assert stride == 1
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = ELSA(dim, num_heads, dim_qk=dim_qk, dim_v=dim_v, kernel_size=kernel_size,
                         stride=stride, dilation=dilation,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                         group_width=group_width, groups=groups, lam=lam, gamma=gamma, **kwargs)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim,
                       hidden_dim=mlp_hidden_dim,
                       dropout_rate=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Bottleneck_ELSA(nn.Module):
    def __init__(self, dim, kernel_size, depths, nums_head):
        super().__init__()
        layers = []
        for _ in range(depths):
            layers.extend(
                [ELSABlock(dim=dim, kernel_size=kernel_size, num_heads=nums_head)
            ])
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)


class Basiclayer_Bottleneck(nn.Module):
    def __init__(self, dim, depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, positional_encoding_type="learned", ):
        super().__init__()
        # self.embedding_dim = 512
        # self.transformer = TransformerModel(dim=self.embedding_dim, depth=4, heads=8, mlp_dim=96*4)
        # if positional_encoding_type == "learned":
        #     self.position_encoding = LearnedPositionalEncoding()
        # elif positional_encoding_type == "fixed":
        #     self.position_encoding = FixedPositionalEncoding(
        #         self.embedding_dim
        #     )
        # self.conv1 = nn.Conv3d(dim, self.embedding_dim, 1, 1)
        self.elsa = Bottleneck_ELSA(dim=dim, kernel_size=7, depths=2, nums_head=8)
        # self.conv2 = nn.Conv3d(self.embedding_dim, dim, 1, 1)
        # self.blc = localattenblock(dim=dim)
        # self.swintrans = BasicLayer(
        #     dim=dim,
        #     depth=depth,
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop,
        #     attn_drop=attn_drop,
        #     drop_path=drop_path,
        #     drop_path_rate=drop_path_rate,
        #     norm_layer=norm_layer,
        #     downsample=downsample,
        #     use_checkpoint=use_checkpoint
        # )

    def forward(self, x):
        x_res = x
        # x1 = x

        ## transBTS block
        # B, D, H, W, C = x.shape
        # x = x.view(x.size(0), -1, self.embedding_dim)
        # x = self.position_encoding(x)
        # x = self.transformer(x)
        # x = x.view(B, D, H, W, C)

        ## swintransformer block
        # x1 = self.swintrans(x1)
        # print(x1.shape)

        ## CBAM block
        # x1 = rearrange(x1, 'b d h w c -> b c d h w')
        # x1 = self.blc(x1)
        # x1 = rearrange(x1, 'b c d h w -> b d h w c')

        ## ELSA block
        x = self.elsa(x)
        x = x + x_res
        # # print(x.shape)
        return x



if __name__ == '__main__':
    a = torch.zeros(1, 4, 4, 4, 768)
    a = ELSABlock(dim=768, kernel_size=5, num_heads=8)(a)
    print(a.shape)