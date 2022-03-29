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
from monai.networks.nets import UNETR, UNet, VNet
from timm.models.layers import trunc_normal_
from monai.data import NiftiSaver
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
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d
)

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\BTCV_Abdomen\RawData\Training'
    ValidSegDir = r'D:\Caiyimin\Dataset\BTCV_Abdomen\ValidSeg'
    Seed = 42
    ResamplePixDim = (1.5, 1.5, 1.5)  # 原始采样分辨率是(1.5,1.5,2.0)
    # 采样间隔为1mm,数据尺寸中位数为[388.  388.  441.5]
    FinalShape = [160, 160, 160]
    window_size = [i // 32 for i in FinalShape]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 1
    SampleN = 2

    low_percent = 0.5
    upper_percent = 99.5
    HuMin = -175.0
    HuMax = 250.0

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0

    n_classes = 14  #

    lr = 3e-4  # 学习率

    MinEpoch = 1000
    MaxEpoch = 5000

    model_name = 'SwinUnet3D'
    # model_name = 'Unet3D'
    # model_name = 'VNet'
    # model_name = 'UNetR'

    ModelDict = {}
    ArgsDict = {}

    ModelDict['Unet3D'] = UNet
    ArgsDict['Unet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                          'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2)}

    ModelDict['VNet'] = VNet
    ArgsDict['VNet'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes, 'dropout_prob': 0.0, }

    ModelDict['UNetR'] = UNETR
    ArgsDict['UNetR'] = {'in_channels': in_channels, 'out_channels': n_classes, 'img_size': FinalShape}

    ModelDict['SwinUnet3D'] = swinUnet_t_3D
    ArgsDict['SwinUnet3D'] = {'in_channel': in_channels, 'num_classes': n_classes, 'window_size': window_size}

    # 滑动窗口推理时使用
    roi_size = FinalShape
    slid_window_overlap = 0.25

    check_val_every_n_epoch = 5  # 多少个epoch进行验证集测试一次

    patience = 10


class BTCVDataset(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(BTCVDataset, self).__init__()
        self.cfg = cfg
        self.image_path = os.path.join(cfg.data_path, 'img')
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

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),
            # ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.low_percent,
            #                                 upper=cfg.upper_percent, b_max=1.0,
            #                                 b_min=0.0, clip=True),

            # CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=cfg.FinalShape, pos=1, neg=1,
                num_samples=cfg.SampleN, image_key="image", ),

            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
            RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3),

            RandScaleIntensityd(keys='image', factors=0.1, prob=0.1),
            RandShiftIntensityd(keys='image', offsets=0.1, prob=0.1),

            EnsureTyped(keys=['image', 'label']),
        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Spacingd(keys=['image', 'label'], pixdim=cfg.ResamplePixDim,
                     mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),

            EnsureTyped(keys=['image', 'label']),
        ])

    def get_init(self):
        images = glob(os.path.join(self.image_path, '*nii.gz'))
        labels = glob(os.path.join(self.label_path, '*nii.gz'))
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
            = random_split(self.train_dict, [train_num, val_num, test_num],
                           generator=torch.Generator().manual_seed(cfg.Seed)
                           )


class BTCV(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(BTCV, self).__init__()
        self.cfg = cfg

        model = cfg.ModelDict[cfg.model_name]
        args = cfg.ArgsDict[cfg.model_name]
        self.net = model(**args)
        if cfg.model_name != 'SwinUnet3D':  # Monai中的模型缺乏参数初始化，不加参数初始化容易导致某些通道的dice系数爆零
            self.net = ModelParamInit(self.net)

        self.loss_func = DiceLoss(to_onehot_y=True, softmax=True)
        self.metrics = DiceMetric(include_background=False,
                                  reduction='mean_batch',
                                  get_not_nans=False)
        self.post_pred = Compose([
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=cfg.n_classes)
        ])
        self.post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=cfg.n_classes)
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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=cfg.BatchSize,
                                         predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        loss, mean_dice = self.shared_step(y_hat=y_hat, y=y)
        # if mean_dice > 0.65:
        meta_dict = batch['image_meta_dict']  # 将meta_dict中的值转成cpu()向量，原来位于GPU上
        for k, v in meta_dict.items():
            if isinstance(v, torch.Tensor):
                meta_dict[k] = v.detach().cpu()
        saver = NiftiSaver(output_dir=self.cfg.ValidSegDir, mode="nearest", print_log=False)
        y_hat = torch.argmax(y_hat, dim=1, keepdim=True)
        saver.save_batch(y_hat, meta_dict)  # fixme 检查此处用法是否正确
        self.log('valid_mean_dice', mean_dice, prog_bar=True)
        self.log('valid_loss', loss, prog_bar=True)
        return {'loss': loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
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

        counts = torch.count_nonzero(dice)
        if counts == 0:
            counts = counts + 1e-5
        dice = torch.sum(dice) / counts
        return loss, dice


def ModelParamInit(model):
    assert isinstance(model, nn.Module)
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model


def main():
    cfg = Config()

    data = BTCVDataset()
    model = BTCV()

    early_stop = EarlyStopping(
        monitor='valid_epoch_mean_loss',
        patience=cfg.patience,
    )

    check_point = ModelCheckpoint(dirpath=f'./trained_models/{cfg.model_name}',
                                  save_last=False,
                                  save_top_k=2, monitor='valid_epoch_mean_loss', verbose=True,
                                  filename='{epoch}-{valid_loss:.2f}-{valid_epoch_mean_dice:.2f}')
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=cfg.MaxEpoch,
        min_epochs=cfg.MinEpoch,
        gpus=1,
        # auto_select_gpus=True, # 这个参数针对混合精度训练时，不能使用

        # auto_lr_find=True,
        auto_scale_batch_size=True,
        logger=TensorBoardLogger(save_dir=f'./logs', name=f'{cfg.model_name}'),
        callbacks=[early_stop, check_point],
        precision=16,
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        auto_lr_find=True,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )
    trainer.fit(model, data)


if __name__ == '__main__':
    main()
