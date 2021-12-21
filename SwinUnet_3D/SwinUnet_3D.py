import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Union, List
import numpy as np


def window_partition_3D(x, window_size: Union[int, List[int]]):
    '''

    :param x: (B,X,Y,Z,C)
    :param window_size: int | (window_size_x, window_size_y, window_size_z)
    :return:（num_windows*B, window_size_x, window_y, window_z,C）
    '''
    B, x_s, y_s, z_s, C = x.shape

    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])

    x = rearrange(x, 'b (x_wn x_ws) (y_wn y_ws) (z_wn z_ws) c -> (b x_wn y_wn z_wn) x_ws y_ws z_ws c',
                  x_ws=window_size[0], y_ws=window_size[1], z_ws=window_size[2])
    return x


def window_reverse_3D(windows, window_size: Union[int, List[int]], X_S: int, Y_S: int, Z_S: int):
    '''
    :param windows: （num_windows*B, window_size_x, window_y, window_z,C）
    :param window_size: int | [ws_x, ws_y, ws_z]
    :param X_S: X轴上图像的尺寸， X_size
    :param Y_S: Y轴上图像的尺寸
    :param Z_S: Z轴上的图像尺寸
    :return: (B,X_S,Y_S,Z_S, C)
    '''

    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])

    x_wn, y_wn, z_wn = X_S // window_size[0], Y_S // window_size[1], Z_S // window_size[2]
    windows_num = x_wn * y_wn * z_wn
    B = int(windows.shape[0] / windows_num)
    x = rearrange(windows, '(b x_wn y_wn z_wn) x_ws y_ws z_ws c -> b (x_wn x_ws) (y_wn y_ws) (z_wn z_ws) c',
                  b=B, x_wn=x_wn, y_wn=y_wn, z_wn=z_wn)
    return x


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

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q和k的矩阵乘法

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


class PatchMerging3D(nn.Module):
    def __init__(self, in_dims, out_dims, downscaling_factor):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims

        # 用3D卷积代替窗口划分以及全连接层，再把输出通道当成编码长度out_dims.
        # self.patch_merge = nn.Conv3d(in_channels=in_dims, out_channels=out_dims,
        #                              kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)

        self.dsf = downscaling_factor  # 2 或者 4

        self.patch_merge = nn.Linear((downscaling_factor ** 3) * in_dims, out_dims)

    def forward(self, x):
        # x: B, C, x_s, y_s, z_s
        dsf = self.dsf
        x = rearrange(x, 'b c (x_ns x_dsf) (y_ns y_dsf) (z_ns z_dsf) -> b x_ns y_ns z_ns (x_dsf y_dsf z_dsf c)',
                      x_dsf=dsf, y_dsf=dsf, z_dsf=dsf)
        x = self.patch_merge(x)

        return x  # b,  w_x //down_scaling, w_y//down_scaling, w_z//down_scaling, out_dim


'''
降采样过程为：
X, Y ,Z, in_dims ->
X // down_scaling, Y // down_scaling,Z//down_scaling, (down_scaling**3)*in_dims -> 
X// down_scaling, Y // down_scaling,Z//down_scaling, out_dims 
一般来说 out_dims = 2 * in_dims
'''

'''
上采样过程为：
X, Y , Z , in_dims ->  
X , Y ,Z, out_dims ->
X , Y ,Z, (down_scaling**3)*out_dims ->  
X* down_scaling, Y*down_scaling,Z*down_scaling, out_dims 
'''


class PatchExpand3D(nn.Module):
    def __init__(self, in_dim, out_dim, up_scaling_factor):
        super(PatchExpand3D, self).__init__()
        self.in_dims = in_dim
        self.out_dims = out_dim

        self.usf = up_scaling_factor  # 2 或者 4

        # X, Y, Z, out_dims -> X, Y, Z, (down_scaling ** 3) * out_dims
        fc_dim = (up_scaling_factor ** 3) * out_dim
        self.c_up = nn.Linear(in_dim, fc_dim)

    def forward(self, x):
        # X: B,C,X,Y,Z
        x = rearrange(x, 'b c x_s y_s z_s -> b x_s y_s z_s c')
        x = self.c_up(x)

        x = rearrange(x, 'b x_s y_s z_s (fac1 fac2 fac3 c) -> b  (x_s fac1) (y_s fac2) (z_s fac3) c',
                      fac1=self.usf, fac2=self.usf, fac3=self.usf)
        return x


class StageModuleDownScaling3D(nn.Module):
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


class StageModuleUpScaling3D(nn.Module):
    def __init__(self, in_dims, hidden_dimension, layers, up_scaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_expand = PatchExpand3D(in_dim=in_dims, out_dim=hidden_dimension,
                                          up_scaling_factor=up_scaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_expand(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)

        # 另一个交换维度的地方是在PatchMerging3D类的forward函数中，在该函数返回之前把维度交换到最后便于进行mlp计算
        x = rearrange(x, 'b x y z c -> b c x y z')
        return x


class DimChange(nn.Module):
    def __init__(self, in_dim: int, out_dim):
        super(DimChange, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.GELU()

    def forward(self, x):
        # x: B,C,X,Y,Z
        x = rearrange(x, 'b c x y z -> b x y z c')
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)

        x = rearrange(x, 'b x y z c -> b c x y z')
        return x


class SwinUnet3D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, in_channel=1, num_classes=2, head_dim=32,
                 window_size: Union[int, List[int]] = 7, downscaling_factors=(4, 2, 2, 2),
                 relative_pos_embedding=True):
        super().__init__()

        self.dsf = downscaling_factors
        self.window_size = window_size

        self.down_stage1 = StageModuleDownScaling3D(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                                    downscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                                    head_dim=head_dim, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)
        self.down_stage2 = StageModuleDownScaling3D(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
                                                    layers=layers[1],
                                                    downscaling_factor=downscaling_factors[1], num_heads=heads[1],
                                                    head_dim=head_dim, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)
        self.down_stage3 = StageModuleDownScaling3D(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
                                                    layers=layers[2],
                                                    downscaling_factor=downscaling_factors[2], num_heads=heads[2],
                                                    head_dim=head_dim, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)
        self.down_stage4 = StageModuleDownScaling3D(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
                                                    layers=layers[3],
                                                    downscaling_factor=downscaling_factors[3], num_heads=heads[3],
                                                    head_dim=head_dim, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)

        self.up_stage1 = StageModuleUpScaling3D(in_dims=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                                                layers=layers[3],
                                                up_scaling_factor=downscaling_factors[3], num_heads=heads[3],
                                                head_dim=head_dim, window_size=window_size,
                                                relative_pos_embedding=relative_pos_embedding)

        self.up_stage2 = StageModuleUpScaling3D(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 2,
                                                layers=layers[2],
                                                up_scaling_factor=downscaling_factors[2], num_heads=heads[2],
                                                head_dim=head_dim, window_size=window_size,
                                                relative_pos_embedding=relative_pos_embedding)

        self.up_stage3 = StageModuleUpScaling3D(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim,
                                                layers=layers[1],
                                                up_scaling_factor=downscaling_factors[1], num_heads=heads[1],
                                                head_dim=head_dim, window_size=window_size,

                                                relative_pos_embedding=relative_pos_embedding)

        self.converge1 = DimChange(hidden_dim * 8, hidden_dim * 4)  # 用于融合upstage和对应的downstage输出的特征，下同
        self.converge2 = DimChange(hidden_dim * 4, hidden_dim * 2)
        self.converge3 = DimChange(hidden_dim * 2, hidden_dim)

        self.final_resume = PatchExpand3D(in_dim=hidden_dim, out_dim=num_classes,
                                          up_scaling_factor=downscaling_factors[0])
        # 参数初始化
        self.init_weight()

        '''
              工作流程为：
              down[i] <==== down_stage[i](x)  , 1<= i <= 4
              up[1]   <==== up_stage[1](down[4])
              up[i]   <==== up_stage[i]( cat(up[i-1], down[4-i]) ), 2 <= i <= 3
              img_embed = img2flatten_dim(img) , # b,flatten_dim,x,y,z  <- b,c,x,y,z
              up[4]   <==== up_stage[4]( cat(up[i], img_embed) ) 
          '''

    def forward(self, img):
        window_size = self.window_size
        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        _, _, x_s, y_s, z_s = img.shape
        x_ws, y_ws, z_ws = window_size

        assert x_s % (x_ws * 32) == 0, f'x轴上的尺寸必须能被x_window_size*32 整除'
        assert y_s % (y_ws * 32) == 0, f'y轴上的尺寸必须能被y_window_size*32 整除'
        assert z_s % (z_ws * 32) == 0, f'y轴上的尺寸必须能被z_window_size*32 整除'

        down1 = self.down_stage1(img)  # (B,C, X//4, Y//4, Z//4)
        down2 = self.down_stage2(down1)  # (B, 2C,X//8, Y//8, Z//8)
        down3 = self.down_stage3(down2)  # (B, 4C,X//16, Y//16, Z//16)
        down4 = self.down_stage4(down3)  # (B, 8C,X//32, Y//32, Z//32)

        up1 = self.up_stage1(down4)  # (B, 4C, X//16, Y//16, Z//16, )
        # fixme up1和 down3融合
        up1 = torch.cat([up1, down3], dim=1)  # (B, 8C,X//16, Y//16, Z//16)
        up1 = self.converge1(up1)  # (B, 4C, X//16, Y//16, Z//16, 4C)

        up2 = self.up_stage2(up1)  # ((B, 2C,X//8, Y//8, Z//8, 2C)
        #  fixme up2和 down2融合
        up2 = torch.cat([up2, down2], dim=1)  # (B, 4C, X//8, Y//8, Z//8)
        up2 = self.converge2(up2)  # (B,2C, X//8, Y//8, Z//8)

        up3 = self.up_stage3(up2)  # (B,C, X//4, Y//4, Z// 4,C)
        # TODO up3和 down1融合
        up3 = torch.cat([up3, down1], dim=1)  # (B,2C, X//4, Y//4, Z//4)
        up3 = self.converge3(up3)  # (B,C, X//4, Y//4, Z//4)

        out = self.final_resume(up3)  # (B,num_classes, X, Y, Z)
        out = rearrange(out, 'b x y z c -> b c x y z')
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.LayerNorm):
                pass


def swinUnet_t_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swinUnet_s_3D(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swinUnet_b_3D(hidden_dim=192, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swinUnet_l_3D(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
