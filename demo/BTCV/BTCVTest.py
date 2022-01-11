# -*-coding:utf-8-*-
import os
import random

import torch
from torch import nn, functional as F, optim
import monai
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import Compose
from monai.transforms import LoadImaged, LoadImage
from monai.data import Dataset, SmartCacheDataset
from torch.utils.data import DataLoader, random_split
from glob import glob
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.config import KeysCollection
from torch.utils.data import random_split
from SwinUnet_3D import swinUnet_t_3D
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, list_data_collate
from monai.networks.utils import one_hot
from einops import rearrange
from torchmetrics.functional import dice_score
from torchmetrics import IoU, Accuracy
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClassesd,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld
)

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\BTCV_Abdomen\RawData\Training'
    img_path = os.path.join(data_path, 'img')
    mask_path = os.path.join(data_path, 'label')

    Lower = 0.5
    Upper = 99.5


cfg = Config()

count_transform = Compose([
    LoadImage(image_only=True),
    transforms.EnsureChannelFirst(),
    transforms.Spacing(pixdim=(1.0, 1.0, 1.0))
])

img_path = cfg.img_path


def getImageFiles():
    img_files = sorted(glob(os.path.join(cfg.img_path, '*.nii.gz')))
    img_dict = []
    for file in img_files:
        tmp = {'image': file}
        img_dict.append(tmp)
    return img_dict


resample_size = []


def countInfo(img_files: [dict], images_size=resample_size):
    org_size = []
    foreground_size = []
    loader = LoadImaged(keys=['image'], image_only=False)
    add_chan = transforms.AddChanneld(keys=['image'])
    spa_trans = transforms.Spacingd(keys='image', pixdim=(1.0, 1.0, 1.0))
    post_trans = Compose([
        transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.Lower, upper=cfg.Upper,
                                                   b_min=0.0, b_max=1.0, clip=True),
        transforms.CropForegroundd(keys=['image'], source_key='image')
    ])
    for file in img_files:
        img = loader(file)
        tmp1 = img['image'].shape
        org_size.append(tmp1)

        img = add_chan(img)
        img = spa_trans(img)
        tmp2 = img['image'].shape
        images_size.append(tmp2[1:])  # 去掉通道维度

        img = post_trans(img)
        tmp3 = img['image'].shape
        foreground_size.append(tmp3[1:])  # 去掉通道维度

    org_size = np.stack(org_size)
    org_median = np.median(org_size, axis=0)
    info = f'原始数据的尺寸中位数：{org_median}'
    print(info)
    with open('./datasetsOrgInfo.txt', 'a+') as f:
        for x in org_size:
            x = f'{x}'
            f.write(x + '\n')
        f.write(info + '\n')

    images_size = np.stack(images_size)
    median = np.median(images_size, axis=0)
    info = f'重采样之后的数据尺寸中位数为: {median}'
    print(info)
    with open('./datasetsResampleInfo.txt', 'a+') as f:
        for x in images_size:
            x = f'{x}'
            f.write(x + '\n')
        f.write(info + '\n')

    foreground_size = np.stack(foreground_size)
    fore_median = np.median(foreground_size, axis=0)
    info = f'前景的尺寸中位数为：{fore_median}'
    print(info)
    with open('./datasetForeInfo.txt', 'a+') as f:
        for x in foreground_size:
            x = f'{x}'
            f.write(x + '\n')
        f.write(info + '\n')


img_dicts = getImageFiles()
countInfo(img_dicts)

# -*-coding:utf-8-*-
