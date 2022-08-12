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
    data_path = r'D:\Caiyimin\Dataset\MSD\Pancreas'

    FinalShape = [160, 160, 160]
    window_size = [it // 32 for it in FinalShape]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 1

    # 数据集原始尺寸(体素间距为1.0时)中位数为(411,411,240)
    # 体素间距为1时，z轴最小尺寸为127，最大为499
    ResamplePixDim = (1.5, 1.5, 2.0)
    HuMax = -57
    HuMin = 164
    low_percent = 0.5
    upper_percent = 99.5

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0

    n_classes = 2  # 括pancreas和cancer这两个通道

    lr = 1e-4  # 学习率

    back_bone_name = 'SwinUnet_0'
    # back_bone_name = 'Unet3D_0'
    # back_bone_name = 'UnetR'

    # 滑动窗口推理时使用
    roi_size = FinalShape
    slid_window_overlap = 0.5


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


class ConvertLabeld(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ConvertLabeld, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            res = []
            # 将 tumor和pancreas合并成pancreas
            res.append(np.logical_or(img == 1, img == 2))
            res.append(img == 2)  # tumor通道

            res = np.stack(res, axis=0)
            # res = np.concatenate(res, axis=0)
            res = res.astype(np.float)
            d[key] = res
        return d


class LitsDataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(LitsDataSet, self).__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.train_path = os.path.join(cfg.data_path, 'imagesTr')
        self.label_tr_path = os.path.join(cfg.data_path, 'labelsTr')
        self.test_path = os.path.join(cfg.data_path, 'imagesTs')

        self.train_dict = []
        self.val_dict = []
        self.test_dict = []

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_process = None
        self.val_process = None

    def prepare_data(self):
        train_x, train_y, test_x = self.get_init()
        for x, y in zip(train_x, train_y):
            info = {'image': x, 'label': y}
            self.train_dict.append(info)

        for x in test_x:
            info = {'image': x}
            self.test_dict.append(info)
        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.split_dataset()
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.test_set = Dataset(self.test_dict, transform=self.val_process)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers,
                          collate_fn=list_data_collate)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.test_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def get_preprocess(self):
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image']),

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=['image', 'label'], spatial_size=cfg.FinalShape),
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label',
                                   spatial_size=cfg.FinalShape,
                                   pos=1, neg=1, num_samples=1, image_key='image', ),

            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),

            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            EnsureTyped(keys=['image', 'label']),

        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),

            EnsureChannelFirstd(keys=['image']),
            ConvertLabeld(keys='label'),

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),
            # CropForegroundd(keys=['image', 'label'], source_key='image'),

            EnsureTyped(keys=['image', 'label']),
        ])

    def get_init(self):
        train_x = sorted(glob(os.path.join(self.train_path, '*.nii.gz')))
        train_y = sorted(glob(os.path.join(self.label_tr_path, '*.nii.gz')))
        test_x = sorted(glob(os.path.join(self.test_path, '*.nii.gz')))

        return train_x, train_y, test_x

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


class Lung(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(Lung, self).__init__()
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

        self.loss_func = DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                                  squared_pred=False,
                                  sigmoid=True)

        self.metrics = DiceMetric(include_background=True,
                                  reduction='mean_batch')
        self.post_pred = Compose([
            EnsureType(), Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ])

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7,
                          weight_decay=1e-5)

        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     opt, T_0=5, T_mult=1, )
        # return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}

        return opt

    def training_step(self, batch, batch_idx):

        x = batch['image']
        y = batch['label']
        # y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape,
        #                                  sw_batch_size=cfg.BatchSize,
        #                                  predictor=self.net,
        #                                  overlap=cfg.slid_window_overlap)
        y_hat = self.net(x)

        loss, dice = self.shared_step(y_hat=y_hat, y=y)
        p_dice, t_dice = dice[0], dice[1]
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_pancreas_dice', p_dice, prog_bar=True)
        self.log('train_tumor_dice', t_dice, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=cfg.BatchSize, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        loss, dice = self.shared_step(y_hat=y_hat, y=y)
        p_dice, t_dice = dice[0], dice[1]
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_pancreas_dice', p_dice, prog_bar=True)
        self.log('valid_tumor_dice', t_dice, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=1, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)

        loss, dice = self.shared_step(y_hat=y_hat, y=y)
        p_dice, t_dice = dice[0], dice[1]
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_pancreas_dice', p_dice, prog_bar=True)
        self.log('test_tumor_dice', t_dice, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        p_dice, t_dice = dice[0], dice[1]
        self.log('train_mean_loss', losses, prog_bar=True)
        self.log('train_mean_pancreas_dice', p_dice, prog_bar=True)
        self.log('train_mean_tumor_dice', t_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        p_dice, t_dice = dice[0], dice[1]
        self.log('valid_mean_loss', losses, prog_bar=True)
        self.log('valid_mean_pancreas_dice', p_dice, prog_bar=True)
        self.log('valid_mean_tumor_dice', t_dice, prog_bar=True)

    def test_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        p_dice, t_dice = dice[0], dice[1]
        self.log('valid_mean_loss', losses, prog_bar=True)
        self.log('valid_mean_pancreas_dice', p_dice, prog_bar=True)
        self.log('valid_mean_tumor_dice', t_dice, prog_bar=True)

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

        dice = dice.detach().cpu().numpy()
        return losses, dice

    def shared_step(self, y_hat, y):
        loss = self.loss_func(y_hat, y)

        y_hat = [self.post_pred(it) for it in decollate_batch(y_hat)]
        y = decollate_batch(y)

        dice = self.metrics(y_hat, y)

        dice = torch.nan_to_num(dice)
        loss = torch.nan_to_num(loss)

        dice = torch.mean(dice, dim=0)
        return loss, dice


data = LitsDataSet()
model = Lung()

early_stop = EarlyStopping(
    monitor='valid_mean_loss',
    patience=10,
)

cfg = Config()
check_point = ModelCheckpoint(dirpath=f'./trained_models/{cfg.back_bone_name}',
                              save_last=False,
                              save_top_k=2, monitor='valid_mean_loss', verbose=True,
                              filename='{epoch}-{valid_loss:.2f}-{valid_mean_dice:.2f}')
trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=400,
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
    auto_lr_find=True
)
trainer.fit(model, data)
