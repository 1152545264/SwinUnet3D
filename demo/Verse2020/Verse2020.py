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
    RandCropByPosNegLabeld,
    Resized
)

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\Verse2020'
    FinalShape = [192, 192, 224]
    window_size = [6, 6, 7]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 1

    # 数据集原始尺寸(体素间距为1.0时)中位数为(386. 399. 464.)
    # 体素间距为1时，z轴最小尺寸为222，最大为1731
    ResamplePixDim = (2.0, 2.0, 2.0)

    HuMin = -500
    HuMax = 1300
    # 或者试试-200和1000，源自官方GitHub仓库

    low_percent = 0.5
    upper_percent = 99.5

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0

    n_classes = 27  # 26,27在本数据集上没用到，故为29-2=27
    '''
    源自：https://github.com/anjany/verse
    1-7: cervical spine: C1-C7
    8-19: thoracic spine: T1-T12
    20-25: lumbar spine: L1-L6
    26: sacrum - not labeled in this dataset
    27: cocygis - not labeled in this dataset
    28: additional 13th thoracic vertebra, T13
    0：背景
    '''

    lr = 3e-4  # 学习率

    back_bone_name = 'SwinUnet_0'
    # back_bone_name = 'Unet3D_0'
    # back_bone_name = 'UnetR'

    # 滑动窗口推理时使用
    roi_size = FinalShape
    slid_window_overlap = 0.5


class ConvertLabel(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            img[img == 28] = 26
            d[key] = img
        return d


class VerseDataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(VerseDataSet, self).__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.image_path = os.path.join(cfg.data_path, 'image')
        self.label_path = os.path.join(cfg.data_path, 'label')

        self.train_dict = []
        self.val_dict = []
        self.test_dict = []

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_process = None
        self.val_process = None

    def prepare_data(self):
        images, labels = self.get_init()
        for x, y in zip(images, labels):
            info = {'image': x, 'label': y}
            self.train_dict.append(info)

        self.split_dataset()
        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.test_set = Dataset(self.test_dict, transform=self.val_process)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers,
                          collate_fn=list_data_collate,
                          shuffle=True)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        # cfg = self.cfg
        # return DataLoader(self.test_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)
        pass

    def get_preprocess(self):
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            ConvertLabel(keys='label'),

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            # Resized(keys=['image', 'label'], spatial_size=cfg.FinalShape),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),

            SpatialPadd(keys=['image', 'label'], spatial_size=cfg.FinalShape),
            RandSpatialCropd(keys=['image', 'label'], roi_size=cfg.FinalShape,
                             random_size=False, random_center=True),
            SpatialPadd(keys=['image', 'label'], spatial_size=cfg.FinalShape),
            # 当心RandSpatialCropd有bug，在verse2020中的verse581会出现y_hat和y的尺寸不一致，分辨率为2.0时，
            # seg581会出现z轴层数为223，和期望的224不一致，本人也不知道为什么

            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),

            EnsureTyped(keys=['image', 'label']),
        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            ConvertLabel(keys='label'),

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),
            # ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.low_percent,
            #                                 upper=cfg.upper_percent, b_max=1.0, b_min=0.0),

            EnsureTyped(keys=['image', 'label']),
        ])

    def get_init(self):
        images = []
        labels = []
        for FatherP, dirs, _ in os.walk(self.image_path):
            for dr in dirs:
                fullP = os.path.join(FatherP, dr, "*.nii.gz")
                nii_file = glob(fullP)
                images.append(nii_file)

        for FatherP, dirs, _ in os.walk(self.label_path):
            for dr in dirs:
                fullP = os.path.join(FatherP, dr, "*.nii.gz")
                nii_file = glob(fullP)
                labels.append(nii_file)

        images.sort()
        labels.sort()
        return images, labels

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_dict)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        test_num = int(num * cfg.test_ratio)
        if train_num + val_num + test_num != num:
            remain = num - train_num - test_num - val_num
            val_num += remain

        self.train_dict, self.val_dict, self.test_dict \
            = random_split(self.train_dict, [train_num, val_num, test_num])


class Verse(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(Verse, self).__init__()
        self.cfg = cfg
        if cfg.back_bone_name == 'SwinUnet_0':
            self.net = swinUnet_t_3D(window_size=cfg.window_size,
                                     num_classes=cfg.n_classes,
                                     in_channel=cfg.in_channels, )
        else:
            from monai.networks.nets import UNETR, UNet
            if cfg.back_bone_name == 'UnetR':
                self.net = UNETR(in_channels=cfg.in_channels,
                                 out_channels=cfg.n_classes,
                                 img_size=cfg.FinalShape)
            else:
                self.net = UNet(spatial_dims=3, in_channels=1,
                                out_channels=cfg.n_classes,
                                channels=(32, 64, 128, 256, 512),
                                strides=(2, 2, 2, 2))

        self.loss_func = DiceLoss(to_onehot_y=True, softmax=True)
        self.metrics = DiceMetric(include_background=False,  # 原此处为True
                                  reduction='mean_batch',
                                  get_not_nans=False)
        self.post_pred = Compose([
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=True, num_classes=cfg.n_classes)
        ])
        self.post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=True, num_classes=cfg.n_classes)
        ])

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7,
                          weight_decay=1e-5)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=30, )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_epoch_mean_loss'}

        # return opt

    def training_step(self, batch, batch_idx):

        x = batch['image']
        y = batch['label']
        y_hat = self.net(x)
        loss, mean_dice = self.shared_step(y_hat=y_hat, y=y)
        self.log('train_mean_dice', mean_dice, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=cfg.BatchSize, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        loss, mean_dice = self.shared_step(y_hat=y_hat, y=y)
        self.log('valid_mean_dice', mean_dice, prog_bar=True)
        self.log('valid_loss', loss, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=1, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)

        loss, mean_dice = self.shared_step(y_hat=y_hat, y=y)
        self.log('valid_mean_dice', mean_dice, prog_bar=True)
        self.log('valid_loss', loss, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'loss')
        self.log('train_epoch_mean_dice', mean_dice, prog_bar=True)
        self.log('train_epoch_mean_loss', losses, prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'loss')
        self.log('valid_epoch_mean_dice', mean_dice, prog_bar=True)
        self.log('valid_epoch_mean_loss', losses, prog_bar=True)

    def test_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'loss')
        self.log('test_epoch_mean_dice', mean_dice, prog_bar=True)
        self.log('test_epoch_mean_loss', losses, prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key):
        losses = []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key].item()
            losses.append(loss)

        losses = np.array(losses)
        losses = np.mean(losses)

        dice = self.metrics.aggregate()
        self.metrics.reset()
        dice = torch.mean(dice)

        dice = dice.detach().cpu().numpy()
        return losses, dice

    def shared_step(self, y_hat, y):
        assert y_hat.shape[2:] == y.shape[2:]
        loss = self.loss_func(y_hat, y)

        y_hat = [self.post_pred(it) for it in decollate_batch(y_hat)]
        y = [self.post_label(it) for it in decollate_batch(y)]

        dice = self.metrics(y_hat, y)

        dice = torch.nan_to_num(dice)
        loss = torch.nan_to_num(loss)

        dice = torch.mean(dice)
        return loss, dice


data = VerseDataSet()
model = Verse()

early_stop = EarlyStopping(
    monitor='valid_epoch_mean_loss',
    patience=10,
)

cfg = Config()
check_point = ModelCheckpoint(dirpath=f'./trained_models/{cfg.back_bone_name}',
                              save_last=False,
                              save_top_k=2, monitor='valid_epoch_mean_loss', verbose=True,
                              filename='{epoch}-{valid_loss:.2f}-{valid_epoch_mean_dice:.2f}')
trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    # max_epochs=400,
    min_epochs=30,
    gpus=1,
    # auto_select_gpus=True, # 这个参数针对混合精度训练时，不能使用

    # auto_lr_find=True,
    auto_scale_batch_size=True,
    logger=TensorBoardLogger(save_dir=f'./logs', name=f'{cfg.back_bone_name}'),
    callbacks=[early_stop, check_point],
    precision=16,
    accumulate_grad_batches=4,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    auto_lr_find=True,
    gradient_clip_val=0.5
)
trainer.fit(model, data)
