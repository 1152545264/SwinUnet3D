from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from .shuffleblock import Basicunit_shuffle
from .PatchPartition import PatchEmbed3D
from .BasicLayer import BasicLayer, PatchExpand_Up, Basiclayer_up, \
    FinalPatchExpand_X4, FinalPatchExpand_X41, SwinTransformerBlock3D
from .BasicLayer_Res import Basiclayer_Bottleneck
from einops import rearrange

"""
该代码主要用于Bottleneck模块的修改
"""


class swinunetr(nn.Module):
    def __init__(self, image_size=128, patch_size=4, in_chans=4, num_lables=3, embed_dim=96,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super(swinunetr, self).__init__()
        self.im_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_class = num_lables
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # patch Patition
        self.patch_embed3D = PatchEmbed3D(
            img_size=image_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
        )
        # patches_resolution = self.patch_embed3D.patches_resolution
        # self.patches_resolution = patches_resolution
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 64, 64, 64, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build encoder and bottleneck
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer < 3:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                    use_checkpoint=use_checkpoint)
            else:
                layer = Basiclayer_Bottleneck(dim=int(embed_dim * 2 ** i_layer),
                                              depth=depths[i_layer],
                                              num_heads=num_heads[i_layer],
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias,
                                              qk_scale=qk_scale,
                                              drop=drop_rate,
                                              attn_drop=attn_drop_rate,
                                              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                              drop_path_rate=drop_path_rate,
                                              norm_layer=norm_layer,
                                              downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                                              use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        # build decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        # self.layers_Res = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      bias=False) if i_layer > 0 else nn.Identity()
            # layer_Res = Basiclayer_Res(
            #     dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            # )
            if i_layer == 0:
                layer_up = PatchExpand_Up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          norm_layer=norm_layer)
            else:
                layer_up = Basiclayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand_Up if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
            # self.layers_Res.append(layer_Res)

        # self.up = FinalPatchExpand_X4(embed_dim, patch_size=patch_size)
        self.up = FinalPatchExpand_X41(
            input_resolution=(image_size // patch_size, image_size // patch_size, image_size // patch_size),
            dim_scale=4, dim=embed_dim)
        # self.up = FinalPatchExpand_X41(input_resolution=(64, 64, 64),
        #                 dim_scale=2, dim=embed_dim)

        self.output = nn.Conv3d(in_channels=embed_dim, out_channels=num_lables, kernel_size=1, bias=False)

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed3D(x)  # B C D H W
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for i, layer in enumerate(self.layers):
            if i < 3:
                x_downsample.append(x)
            # print(i)
            x = layer(x)
        x = self.norm(x)

        return x, x_downsample

    # Decoder and skip connection
    def forward_up_features(self, x, x_downsample):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:

                x = torch.cat((x, x_downsample[3 - i]), 4)  # B D H W C
                # x = torch.cat((x, x_downsample[3-i]), 4) # B D H W C
                B, D, H, W, _ = x.shape
                x = x.flatten(1, 3)
                x = self.concat_back_dim[i](x)
                _, _, C = x.shape
                x = x.view(B, D, H, W, C)
                x = layer_up(x)
                # print(x.shape, x_pool.shape)
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        # D, H, W = self.patches_resolution
        # B, D, H, W, C = x.shape
        x = self.up(x)
        # print(x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.output(x)
        return x

    def forward(self, x):
        # or_x = x
        # print(x.shape)
        x, x_downsample = self.forward_features(x)  # x, x_downsample shape: B D H W C
        # print(len(x_downsample))

        x = self.forward_up_features(x, x_downsample)
        # print(x.shape)
        x = self.up_x4(x)

        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,
                                                                                                          self.patch_size[
                                                                                                              0], 1,
                                                                                                          1) / \
                                                self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)

            print(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                # self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                self.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class PatchMerging1(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        # x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        # x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        # x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        # self.shuffle = Basicunit_shuffle(dim, dim, s_ratio=0.5, groups=2)
        self.dwconv = nn.Sequential(
            nn.ReLU(inplace=False),
            # nn.GELU(),
            # nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1),
            # nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            # nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm3d(dim),
            # nn.GroupNorm(num_groups=8, num_channels=dim)
        )
        self.conv = nn.Conv3d(dim, 2 * dim, kernel_size=2, stride=2)
        self.gelu = nn.GELU()
        # self.bn = nn.BatchNorm3d(2*dim)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.conv(x + self.dwconv(x))
        # x = self.conv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        # print(x.shape)
        x = self.gelu(self.norm(x))
        return x


if __name__ == '__main__':
    a = torch.ones(1, 4, 128, 128, 128)

    a = swinunetr()(a)
    print(a.shape)
