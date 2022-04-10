# -*-coding:utf-8-*-
import torch
from monai.networks.nets import UNet, VNet, UNETR
from SwinUnet_3D import swinUnet_t_3D
from thop import profile

in_channels = 4
n_classes = 3
RoiSize = (256, 256, 160)
window_size = (8, 8, 5)

ModelDict = {}
ArgsDict = {}
ModelDict['Unet3D'] = UNet
ArgsDict['Unet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                      'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2)}

ModelDict['VNet'] = VNet
ArgsDict['VNet'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes, 'dropout_prob': 0.0, }

ModelDict['UNetR'] = UNETR
ArgsDict['UNetR'] = {'in_channels': in_channels, 'out_channels': n_classes, 'img_size': RoiSize}

ModelDict['SwinUnet3D'] = swinUnet_t_3D
ArgsDict['SwinUnet3D'] = {'in_channel': in_channels, 'num_classes': n_classes, 'window_size': window_size}

inputs = torch.randn((1, 4, 256, 256, 160))
for model_name, model in ModelDict.items():
    model2 = model(**ArgsDict[model_name])
    print(model_name)
    macs, params = profile(model2, inputs=(inputs,), verbose=True)
    print(macs / 1e9, "*****************", params / 1e6)
