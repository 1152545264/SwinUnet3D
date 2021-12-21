# -*-coding:utf-8-*-
import os
import random

import torch
import tqdm
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
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric
from torchmetrics.functional import dice_score
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\Brats2021'
    # 脑组织窗宽设定为80Hu~100Hu, 窗位为30Hu~40Hu,
    PadShape = [240, 240, 160]
    FinalShape = PadShape
    window_size = [7, 7, 5]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 4

    train_ratio, val_ratio, test_ratio = [0.7, 0.2, 0.1]
    BatchSize = 1
    NumWorkers = 0
    '''
    标签中绿色为浮肿区域(ED,peritumoral edema) (标签2)、黄色为增强肿瘤区域(ET,enhancing tumor)(标签4)、
    红色为坏疽(NET,non-enhancing tumor)(标签1)、背景(标签0)
    WT = ED + ET + NET
    TC = ET+NET
    ET
    '''
    n_classes = 3  # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、ET（肿瘤增强)和WT（整个肿瘤）。

    lr = 3e-4  # 学习率
    back_bone_name = 'SwinUnet'

    # 滑动窗口推理时使用
    roi_size = FinalShape
    overlap = 0.5


def img_analysis():
    FP = os.path.join(Config.data_path, 'Brats2021Train')
    train_x, train_y = [], []
    for _, dirs, _ in os.walk(FP):
        for dr in dirs:
            tmp = os.path.join(FP, dr)
            flair_file = glob(os.path.join(tmp, '*flair.nii.gz'), recursive=True)
            t1_file = glob(os.path.join(tmp, '*t1.nii.gz'), recursive=True)
            t1_ce_file = glob(os.path.join(tmp, '*ce.nii.gz'), recursive=True)
            t2_file = glob(os.path.join(tmp, '*t2.nii.gz'), recursive=True)
            seg_file = glob(os.path.join(tmp, '*seg.nii.gz'), recursive=True)
            files = [*flair_file, *t1_file, *t1_ce_file, *t2_file]
            train_x.append(files)
            train_y.append(seg_file)

    i = 0
    for file, seg in zip(train_x, train_y):
        flair, t1, t1_ce, t2 = file[0], file[1], file[2], file[3]
        flair, t1, t1_ce, t2, seg = LoadImage(image_only=True)(flair), LoadImage(image_only=True)(t1), LoadImage(
            image_only=True)(t1_ce), LoadImage(image_only=True)(t2), LoadImage(image_only=True)(seg),
        img = np.stack([flair, t1, t1_ce, t2], axis=-1)
        tmp = img.shape[:-1]
        if list(tmp) != [240, 240, 155]:
            print(f'图像尺寸不为(240,240,155)，文件名为{file[0]}')
        M, m = np.max(seg), np.min(seg)
        res0 = (seg == 0).sum()
        res1 = (seg == 1).sum()
        res2 = (seg == 2).sum()
        res3 = (seg == 3).sum()
        res4 = (seg == 4).sum()
        i += 1
        if M != 4 or m != 0:
            print(f'第{i}个图像最大值为{M},最小值为{m}，文件名为{file[0]}，像素分布为：{res0},"**",{res1}"**",{res2}"**",{res3}"**",{res4}')

    print(len(train_x))


img_analysis()
