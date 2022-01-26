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
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric
from torchmetrics.functional import dice_score
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
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
    SpatialPadd
)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\MSD\BrainTumor'
    # 脑组织窗宽设定为80Hu~100Hu, 窗位为30Hu~40Hu,
    HUMAX = 140
    HUMIN = -70
    FinalShape = [224, 224, 128]
    window_size = [7, 7, 4]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 4
    ResamplePixDim = [1.0, 1.0, 1.0]

    train_ratio, val_ratio, test_ratio = [0.7, 0.2, 0.1]
    BatchSize = 1
    NumWorkers = 0

    n_classes = 3  # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。

    lr = 3e-4  # 学习率
    back_bone_name = 'Unet3D_0'

    # 滑动窗口推理时使用
    roi_size = [224, 224, 128]
    overlap = 0.5


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


# 摘自：https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []

            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)

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
            info = {'image': x, 'seg': y}
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
        return DataLoader(self.train_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.test_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def get_preprocess(self):
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            EnsureChannelFirstd(keys=['image']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            Spacingd(keys=['image', 'seg'], pixdim=cfg.ResamplePixDim, mode=['bilinear', 'nearest']),
            Orientationd(keys=['image', 'seg'], axcodes='RAS'),
            RandSpatialCropd(keys=['image', 'seg'], roi_size=cfg.FinalShape, random_size=False),
            SpatialPadd(keys=['image', 'label'], spatial_size=cfg.FinalShape),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            EnsureTyped(keys=['image', 'seg']),
        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            EnsureChannelFirstd(keys=['image']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            Spacingd(keys=['image', 'seg'], pixdim=cfg.ResamplePixDim, mode=['bilinear', 'nearest']),
            Orientationd(keys=['image', 'seg'], axcodes='RAS'),

            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            EnsureTyped(keys=['image', 'seg']),
        ])

    def get_init(self):
        train_x = glob(os.path.join(self.train_path, '*.nii.gz'))
        train_y = glob(os.path.join(self.label_tr_path, '*.nii.gz'))
        test_x = glob(os.path.join(self.test_path, '*.nii.gz'))
        sorted(train_x)
        sorted(train_y)
        sorted(test_x)

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

        self.train_dict, self.val_dict, self.test_dict = random_split(self.train_dict, [train_num, val_num, test_num])


class LITSModel(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(LITSModel, self).__init__()
        self.cfg = cfg
        if cfg.back_bone_name == 'SwinUnet_0':
            self.net = swinUnet_t_3D(window_size=cfg.window_size, num_classes=cfg.n_classes, in_channel=cfg.in_channels,
                                     flatten_dim=256)
        else:
            from monai.networks.nets import UNETR, UNet
            self.net = UNETR(in_channels=cfg.in_channels, out_channels=cfg.n_classes, img_size=cfg.FinalShape)

        self.loss_func = DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True,
                                       include_background=True)
        self.post_transform = Compose([
            transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold_values=True)
        ])

        # 这个类当y_pred和y均为全零矩阵时，此处计算出来的dice系数为nan
        # self.metrics = DiceMetric(include_background=True, reduction="mean")
        self.metrics = dice_score

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['seg'].float()
        y_hat = self.net(x)

        losses, dices = self.shared_step(y_hat, y)
        tc_loss, wt_loss, et_loss = losses[0], losses[1], losses[2]
        tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

        self.log('train_tc_loss', tc_loss, prog_bar=False)
        self.log('train_tc_dice', tc_dice, prog_bar=False)

        self.log('train_wt_loss', wt_loss, prog_bar=False)
        self.log('train_wt_dice', wt_dice, prog_bar=False)

        self.log('train_et_loss', et_loss, prog_bar=False)
        self.log('train_et_dice', et_dice, prog_bar=False)

        mean_loss = torch.mean(losses)
        return {'loss': mean_loss, 'train_losses': losses, 'train_dice': dices}

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['seg'].float()
        # 使用滑动窗口进行推理
        y_hat = sliding_window_inference(x, roi_size=cfg.roi_size, overlap=cfg.overlap,
                                         sw_batch_size=1, predictor=self.net)

        losses, dices = self.shared_step(y_hat, y)
        tc_loss, wt_loss, et_loss = losses[0], losses[1], losses[2]
        tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

        self.log('valid_tc_loss', tc_loss, prog_bar=False)
        self.log('valid_tc_dice', tc_dice, prog_bar=False)

        self.log('valid_wt_loss', wt_loss, prog_bar=False)
        self.log('valid_wt_dice', wt_dice, prog_bar=False)

        self.log('valid_et_loss', et_loss, prog_bar=False)
        self.log('valid_et_dice', et_dice, prog_bar=False)

        return {'valid_loss': losses, 'valid_dice': dices}

    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        losses, dices = self.shared_epoch_end(outputs, 'train_losses', 'train_dice')
        if len(losses) > 0:
            mean_loss = torch.mean(losses, dim=1)
            mean_dice = torch.mean(dices, dim=1)

            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_loss, wt_mean_loss, et_mean_loss = mean_loss[0], mean_loss[1], mean_loss[2]
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('tc_train_mean_loss', tc_mean_loss, prog_bar=True)
            self.log('tc_train_mean_dice', tc_mean_dice, prog_bar=True)

            self.log('wt_train_mean_loss', wt_mean_loss, prog_bar=True)
            self.log('wt_train_mean_dice', wt_mean_dice, prog_bar=True)

            self.log('et_train_mean_loss', et_mean_loss, prog_bar=True)
            self.log('et_train_mean_dice', et_mean_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, dices = self.shared_epoch_end(outputs, 'valid_loss', 'valid_dice')
        if len(losses) > 0:
            mean_loss = torch.mean(losses, dim=1)
            mean_dice = torch.mean(dices, dim=1)

            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_loss, wt_mean_loss, et_mean_loss = mean_loss[0], mean_loss[1], mean_loss[2]
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('tc_valid_mean_loss', tc_mean_loss, prog_bar=True)
            self.log('tc_valid_mean_dice', tc_mean_dice, prog_bar=True)

            self.log('wt_valid_mean_loss', wt_mean_loss, prog_bar=True)
            self.log('wt_valid_mean_dice', wt_mean_dice, prog_bar=True)

            self.log('et_valid_mean_loss', et_mean_loss, prog_bar=True)
            self.log('et_valid_mean_dice', et_mean_dice, prog_bar=True)

            valid_mean_loss = torch.mean(losses)
            self.log('valid_mean_loss', valid_mean_loss, prog_bar=True)

    def test_epoch_end(self, outputs):
        pass

    @staticmethod
    def shared_epoch_end(outputs, loss_key, dice_key):
        losses, dices = [], []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key]
            dice = output[dice_key]

            loss = loss.detach()
            dice = dice.detach()

            losses.append(loss)
            dices.append(dice)
        losses = torch.stack(losses)
        dices = torch.stack(dices)

        losses = torch.permute(losses, (1, 0))
        dices = torch.permute(dices, (1, 0))

        return losses, dices

    def shared_step(self, y_hat, y):
        # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。

        # y_hat: b,c,h,w,d
        y_tc, y_wt, y_et = y[:, 0, ::], y[:, 1, ::], y[:, 2, ::]
        y_tc_hat, y_wt_hat, y_et_hat = y_hat[:, 0, ::], y_hat[:, 1, ::], y_hat[:, 2, ::]

        tc_loss = self.loss_func(y_tc_hat, y_tc)
        wt_loss = self.loss_func(y_wt_hat, y_wt)
        et_loss = self.loss_func(y_et_hat, y_et)
        losses = torch.stack([tc_loss, wt_loss, et_loss], dim=0)

        y_hat = self.post_transform(y_hat)
        y_tc_hat, y_wt_hat, y_et_hat = y_hat[:, 0, ::], y_hat[:, 1, ::], y_hat[:, 2, ::]

        tc_dice = self.metrics(preds=y_tc_hat, target=y_tc, bg=True)
        wt_dice = self.metrics(preds=y_wt_hat, target=y_wt, bg=True)
        et_dice = self.metrics(preds=y_et_hat, target=y_et, bg=True)

        dices = torch.stack([tc_dice, wt_dice, et_dice], dim=0)

        return losses, dices


pl.seed_everything(42)

data = LitsDataSet()
model = LITSModel()

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
    max_epochs=1000,
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
    log_every_n_steps=400,
    auto_lr_find=True
)
trainer.fit(model, data)
