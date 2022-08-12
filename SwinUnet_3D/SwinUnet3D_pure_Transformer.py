# 二维参考自：https://github.com/berniwal/swin-transformer-pytorch
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from typing import Union, List, Optional
from torch.nn import functional as F


class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()

        assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
        if type(displacement) is int:
            displacement = np.array([displacement, displacement, displacement])
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))


class Residual3D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
                  x_shift: bool, y_shift: bool, z_shift: bool):
    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])

    assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
    if type(displacement) is int:
        displacement = np.array([displacement, displacement, displacement])

    assert len(window_size) == len(displacement)
    for i in range(len(window_size)):
        assert 0 < displacement[i] < window_size[i], \
            f'在第{i}轴的偏移量不正确，维度包括X轴(i=0)，Y(i=1)轴和Z轴(i=2)'

    mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
                       window_size[0] * window_size[1] * window_size[2])  # (wx*wy*wz, wx*wy*wz)
    mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
                     x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])

    x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]

    if x_shift:
        #      x1     y1 z1     x2     y2 z2
        mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
        mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')

    if y_shift:
        #   x1   y1       z1 x2  y2       z2
        mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
        mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')

    if z_shift:
        #   x1  y1  z1       x2 y2  z2
        mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
        mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')

    mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
    return mask


# 参考自video_swin_transformer:
# #https://github.com/MohammadRezaQaderi/Video-Swin-Transformer/blob/c3cd8639decf19a25303615db0b6c1195495f5bb/mmaction/models/backbones/swin_transformer.py#L119
def get_relative_distances(window_size):
    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])
    indices = torch.tensor(
        np.array(
            [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))

    distances = indices[None, :, :] - indices[:, None, :]
    # distance:(n,n,3) n =window_size[0]*window_size[1]*window_size[2]
    return distances


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool):
        super().__init__()

        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        else:
            window_size = np.array(window_size)

        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)
            self.x_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=True, y_shift=False, z_shift=False), requires_grad=False)
            self.y_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=True, z_shift=False), requires_grad=False)
            self.z_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=False, z_shift=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # QKV三个

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            # relative_indices的形状为 (n,n,3) n=window_size[0]*window_size[1]*window_size[2],

            for i in range(len(window_size)):  # 在每个维度上进行偏移
                self.relative_indices[:, :, i] += window_size[i] - 1

            self.pos_embedding = nn.Parameter(
                torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1, 2 * window_size[2] - 1)
            )
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1] * window_size[2],
                                                          window_size[0] * window_size[1] * window_size[2]))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_x, n_y, n_z, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_x = n_x // self.window_size[0]
        nw_y = n_y // self.window_size[1]
        nw_z = n_z // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
                                h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long(),
                                       self.relative_indices[:, :, 2].long()]
        else:
            dots += self.pos_embedding

        if self.shifted:
            # 将x轴的窗口数量移至尾部，便于和x轴上对应的mask叠加，下同
            dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
                             n_x=nw_x, n_y=nw_y)
            #   b   h n_y n_z n_x i j
            dots[:, :, :, :, -1] += self.x_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
            dots[:, :, :, :, -1] += self.y_mask

            dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
            dots[:, :, :, :, -1] += self.z_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)  # 进行attn和v的矩阵乘法

        # nw_x 表示in x axis, num of windows , nw_y 表示 in y axis, num of windows
        # w_x 表示 x_window_size, w_y 表示 y_window_size， w_z表示z_window_size
        #                     b 3  (8,8,8)         （7,  7,  7） 96 -> b  56          56          56        288
        out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
                        h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2],
                        nw_x=nw_x, nw_y=nw_y)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class PatchMerging3D(nn.Module):
    def __init__(self, in_dims, out_dims, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor  # 2 或者 4
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.patch_merge = nn.Conv3d(in_channels=in_dims, out_channels=out_dims,
                                     kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)

        # 用3D卷积代替窗口划分以及全连接层，再把输出通道当成编码长度C

    def forward(self, x):
        x = self.patch_merge(x)

        # 另一个交换维度的地方是在StageModule3D类的forward函数中，在该函数返回之前把维度交换到第二维，便于下一个StageModule进行窗口划分(即进行卷积计算)
        x = rearrange(x, 'b c x y z -> b x y z c')  # 交换维度
        return x  # b,  w_x //down_scaling, w_y//down_scaling, w_z//down_scaling, out_dim


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
                 relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
                                                                           heads=heads,
                                                                           head_dim=head_dim,
                                                                           shifted=shifted,
                                                                           window_size=window_size,
                                                                           relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class StageModule3D(nn.Module):
    def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging3D(in_dims=in_dims, out_dims=hidden_dimension,
                                              downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)

        # 另一个交换维度的地方是在PatchMerging3D类的forward函数中，在该函数返回之前把维度交换到最后便于进行mlp计算
        x = rearrange(x, 'b x y z c -> b c x y z')
        return x
        # return x.permute(0, 4, 1, 2, 3)


class SwinTransformer3D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=1, num_classes=1000, head_dim=32,
                 window_size: Union[int, List[int]] = 7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule3D(in_dims=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                    downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                    window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule3D(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                    downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                    window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule3D(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                    downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                    window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule3D(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                    downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                    window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3, 4])
        return self.mlp_head(x)


def swin_t_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s_3D(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b_3D(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l_3D(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


# TODO  2021.11.12 测试整个结构

def test():
    #                b   c   x    y    z
    x = torch.randn((10, 1, 224, 224, 224))
    model = swin_t_3D(window_size=[7, 7, 7])
    # 必须保证: x % window_size[0] == 0, y % window_size[1] == 0 , z % window_size[2] == 0
    # 如果window在此处是个整数在，则必须保证: x % window_size ==0 , y % window_size == 0, z % window_size == 0

    y = model(x)
    print(y.shape)
    print(y)


# test()