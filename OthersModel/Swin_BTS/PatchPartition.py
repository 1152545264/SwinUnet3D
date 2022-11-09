import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class PatchEmbed3D1(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (4,4,4).
        in_chans (int): Number of input video channels. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, patch_size=4, in_chans=4, embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.inter_channel = embed_dim // 2
        self.out_channels = embed_dim
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_chans, self.inter_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(self.inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.inter_channel, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward function."""
        x = self.conv3(self.conv2(self.conv1(x)))
        x = rearrange(x, 'b c d h w -> b d h w c')
        # print(x.shape)
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (4,4,4).
        in_chans (int): Number of input video channels. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, patch_size=4, in_chans=4, embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        patches_resolution = [img_size // 4, img_size // 4, img_size // 4]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        # _, _, D, H, W = x.size()

        PD, Ph, Pw = self.patches_resolution
        # y = self.proj(x)
        # print(y.shape)
        # B, C, _, _, _ = y.shape
        x = self.proj(x)
        # print(x.shape)
        B, C, _, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)  # B  PD*Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = x.view(B, PD, Ph, Pw, C)
        # print(x.shape)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        # x = x.permute(0, 4, 1, 2, 3)
        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale ** 3))
        # x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


if __name__ == '__main__':
    a = torch.ones(1, 4, 128, 128, 128)
    print(a.shape)

    c = PatchEmbed3D()(a)
    print(c.shape)
