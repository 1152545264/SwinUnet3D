import SimpleITK
import torch
import torch.nn as nn


def channel_shuffle(x, groups=2):
    B, C, D, H, W = x.shape
    if C % groups:
        return x
    ch_per_group = C // groups
    x = x.view(B, groups, ch_per_group, D, H, W)
    # print(x.shape)
    x = x.transpose(1, 2).contiguous()
    # print(x.shape)
    x = x.view(B, -1, D, H, W)
    # print(x.shape)
    return x


class BN_Conv3d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, bias=False, activation=False):
        super(BN_Conv3d, self).__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  # nn.BatchNorm3d(out_channels),
                  nn.GroupNorm(num_groups=8, num_channels=out_channels)
                  ]
        if activation:
            layers.append(nn.GELU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Basicunit_shuffle(nn.Module):

    def __init__(self, in_ch, out_ch, s_ratio, groups=2):
        super(Basicunit_shuffle, self).__init__()
        self.l_ch = int(in_ch*s_ratio)
        self.r_ch = in_ch - self.l_ch
        self.ro_ch = out_ch - self.l_ch
        self.groups = groups

        # layer
        self.conv1 = BN_Conv3d(self.r_ch, self.ro_ch, 1, 1, 0)
        self.dwconv = BN_Conv3d(self.ro_ch, self.ro_ch, 3, 1, 1, groups=self.ro_ch, activation=False)
        self.conv2 = BN_Conv3d(self.ro_ch, self.ro_ch, 1, 1, 0, )

    def forward(self, x):
        x_l = x[:, :self.l_ch, :, :]
        x_r = x[:, self.l_ch:, :, :]

        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv(out_r)
        out_r = self.conv2(out_r)

        out = torch.cat((x_l, out_r), 1)
        return channel_shuffle(out, self.groups)


class Resblock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel=3, stride=1, padding=1):
        super(Resblock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel = kernel
        self.conv3D = nn.Conv3d(self.in_chans, self.out_chans, kernel_size=self.kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(self.in_chans),
        self.gn = nn.GroupNorm(num_groups=8, num_channels=self.in_chans),
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        # res_x = x
        # x = self.conv3D(self.relu(self.bn(x)))
        # x = self.conv3D(self.relu(self.bn(x)))
        x = self.gelu(self.gn(self.conv3D(x)))
        return x


